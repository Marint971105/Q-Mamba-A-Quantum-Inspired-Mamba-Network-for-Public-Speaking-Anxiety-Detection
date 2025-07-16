# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from layers.complexnn.multiply import ComplexMultiply
from layers.quantumnn.mixture import QMixture
from layers.quantumnn.measurement import QMeasurement
from layers.quantumnn.outer import QOuter
from models.SimpleNet import SimpleNet
from layers.complexnn.l2_norm import L2Norm
from layers.quantumnn.attention import ModalAttention
from layers.quantumnn.cnot import QCNOT
from layers.quantumnn.hadamard import Hadamard
from layers.realnn.mamba import MambaBlock
from layers.realnn.transformer import TransformerEncoder
from layers.realnn.rwkv import RWKVBlock
from layers.realnn.rnn import RNNBlock
from layers.realnn.cnn import CNNBlock


class QMamba(nn.Module):
    def __init__(self, opt):
        super(QMamba, self).__init__()
        self.device = opt.device    
        self.input_dims = opt.input_dims[:3]  # 只使用text, audio, visual
        self.total_input_dim = sum(self.input_dims)
        self.embed_dim = opt.embed_dim
        self.n_classes = opt.output_dim
        self.num_layers = opt.num_layers  # 添加QRNN层数参数
        self.sequence_model = opt.sequence_model  # 'mamba' 或 'transformer'
        # 新增：权重计算方式
        self.weight_method = getattr(opt, 'weight_method', 'attention')
        
        # 模态投影层
        self.projections = nn.ModuleList([
            nn.Linear(dim, self.embed_dim) for dim in self.input_dims
        ])
        
        # beats投影层
        self.beats_projection = nn.Linear(768, self.embed_dim)
        
        # 添加可训练的语言特征参数
        self.lang1_param = nn.Parameter(torch.randn(1, 1, 768))  # [1, 1, 768]
        self.lang2_param = nn.Parameter(torch.randn(1, 1, 768))  # [1, 1, 768]
        
        # 语言特征投影层
        self.lang_proj1 = nn.Linear(768, self.embed_dim)
        self.lang_proj2 = nn.Linear(768, self.embed_dim)
        
        # 其他组件
        self.multiply = ComplexMultiply()
        self.outer = QOuter()
        self.norm = L2Norm(dim=-1)
        self.mixture = QMixture(device=self.device)
        
        # 输出层
        self.out_dropout_rate = opt.out_dropout_rate
        self.measurement = QMeasurement(self.embed_dim)
        self.fc_out = SimpleNet(
            self.embed_dim, 
            opt.output_cell_dim,
            self.out_dropout_rate,
            self.n_classes,
            output_activation=nn.Tanh()
        )
        
        # 根据权重计算方式选择组件
        if self.weight_method == 'attention':
            # 添加模态注意力层
            self.modal_attention = ModalAttention(
                embed_dim=self.embed_dim,
                num_heads=5,
                device=self.device
            )
        elif self.weight_method == 'magnitude':
            # 使用L2Norm进行模态大小计算
            self.modal_norm = L2Norm(dim=-1)
        else:
            raise ValueError(
                f"Unsupported weight_method: {self.weight_method}"
            )
        
        # 根据选择初始化序列模型
        if self.sequence_model == 'mamba':
            self.seq_blocks = nn.ModuleList([
                MambaBlock(
                    d_model=self.embed_dim,
                    d_state=16,
                    bimamba_type="v2",
                    device=self.device
                ) for _ in range(self.num_layers)
            ])
        elif self.sequence_model == 'transformer':
            self.seq_blocks = nn.ModuleList([
                TransformerEncoder(
                    embed_dim=self.embed_dim,
                    num_heads=opt.num_heads,
                    layers=1,
                    attn_dropout=opt.attn_dropout,
                    relu_dropout=opt.relu_dropout,
                    res_dropout=opt.res_dropout,
                    embed_dropout=opt.embed_dropout
                ) for _ in range(self.num_layers)
            ])
        elif self.sequence_model == 'rwkv':
            self.seq_blocks = nn.ModuleList([
                RWKVBlock(
                    d_model=self.embed_dim,
                    d_state=16,
                    device=self.device
                ) for _ in range(self.num_layers)
            ])
        elif self.sequence_model == 'rnn':
            self.seq_blocks = nn.ModuleList([
                RNNBlock(
                    d_model=self.embed_dim,
                    rnn_type=opt.rnn_type,
                    num_layers=1,
                    bidirectional=opt.bidirectional,
                    dropout=opt.rnn_dropout,
                    device=self.device
                ) for _ in range(self.num_layers)
            ])
        elif self.sequence_model == 'cnn':
            self.seq_blocks = nn.ModuleList([
                CNNBlock(
                    d_model=self.embed_dim,
                    kernel_size=opt.kernel_size,
                    num_layers=1,
                    dropout=opt.cnn_dropout,
                    device=self.device
                ) for _ in range(self.num_layers)
            ])
        
        # 添加CNOT操作
        self.cnot = QCNOT(self.embed_dim, self.device)
        
        # 添加Hadamard门
        self.hadamard = Hadamard(self.embed_dim, self.device)

    def quantum_entangle(self, state1, state2):
        """
        使用Hadamard+CNOT实现两个量子态的纠缠
        
        Args:
            state1, state2: 两个量子态 [real_part, imag_part]
        """
        # 1. 对第一个语言状态应用Hadamard门
        state1_h = self.hadamard(state1)
        
        # 2. 使用CNOT门创建纠缠
        entangled_state = self.cnot(state1_h, state2)
        
        return entangled_state
        
    def forward(self, in_modalities):
        # 1. 提取特征
        text, audio, beats, visual = in_modalities[:4]  # 只取前4个模态
        modalities = [text, audio, visual]
        
        # 获取batch_size
        batch_size = text.shape[0]
        
        # 扩展语言特征参数到当前batch size
        lang1 = self.lang1_param.expand(
            batch_size, -1, -1
        )  # [batch_size, 1, 768]
        lang2 = self.lang2_param.expand(
            batch_size, -1, -1
        )  # [batch_size, 1, 768]
        
        # 2. 投影所有模态
        utterance_reps = [
            nn.ReLU()(projection(x)) 
            for x, projection in zip(modalities, self.projections)
        ]
        
        # 投影语种特征
        lang_rep1 = nn.ReLU()(self.lang_proj1(lang1))
        lang_rep2 = nn.ReLU()(self.lang_proj2(lang2))
        
        # 3. 生成相位
        phase = nn.ReLU()(self.beats_projection(beats))
        
        # 4. 计算振幅
        amplitudes = [F.normalize(rep, dim=-1) for rep in utterance_reps]
        lang_amp1 = F.normalize(lang_rep1, dim=-1)
        lang_amp2 = F.normalize(lang_rep2, dim=-1)
        
        # 5. 构造量子态
        quantum_states = [
            self.multiply([phase, amplitude]) 
            for amplitude in amplitudes
        ]
        
        # 构造语种量子态
        lang_state1 = self.multiply([phase, lang_amp1])
        lang_state2 = self.multiply([phase, lang_amp2])
        
        # Hadamard+CNOT量子纠缠
        lang_entangled = self.quantum_entangle(lang_state1, lang_state2)
        
        # 将纠缠后的语种特征添加到量子态列表
        quantum_states.append(lang_entangled)
        
        # 对相邻模态应用CNOT
        entangled_states = []
        for i in range(len(quantum_states)-1):
            entangled_state = self.cnot(quantum_states[i], quantum_states[i+1])
            entangled_states.append(entangled_state)
        
        # 使用纠缠后的状态计算外积
        unimodal_matrices = [
            self.outer(quantum_state) 
            for quantum_state in entangled_states
        ]
        
        # 7. 计算模态权重 - 根据选择的方式
        if self.weight_method == 'attention':
            # 使用多头注意力方式
            weights = self.modal_attention(utterance_reps)
        elif self.weight_method == 'magnitude':
            # 使用模态大小方式（类似QMN）
            weights = [self.modal_norm(rep) for rep in utterance_reps]
            weights = F.softmax(torch.cat(weights, dim=-1), dim=-1)
        else:
            raise ValueError(
                f"Unsupported weight_method: {self.weight_method}"
            )
        
        # 8. 混合更新后的量子态
        mixed_state = self.mixture([unimodal_matrices, weights])
        
        # 保存中间状态供分析
        self.quantum_states = quantum_states
        self.entangled_states = entangled_states
        
        # 9. 使用Mamba处理多层RNN
        batch_size = in_modalities[0].shape[0]
        time_stamps = in_modalities[0].shape[1]
        
        # 处理mixed_state
        in_states = mixed_state
        
        # 序列处理部分
        if self.sequence_model == 'mamba':
            # Mamba处理
            for block in self.seq_blocks:
                h_r = torch.stack(
                    batch_size*[torch.eye(self.embed_dim)/self.embed_dim], 
                    dim=0
                ).to(self.device)
                h_i = torch.zeros_like(h_r).to(self.device)
                h = [h_r, h_i]
                
                all_h = []
                for t in range(time_stamps):
                    current_state = in_states[t]
                    h = block(current_state)
                    all_h.append(h)
                
                in_states = all_h
        
        elif self.sequence_model == 'transformer':  # transformer
            for block in self.seq_blocks:
                h_r = torch.stack(
                    batch_size*[torch.eye(self.embed_dim)/self.embed_dim], 
                    dim=0
                ).to(self.device)
                h_i = torch.zeros_like(h_r).to(self.device)
                h = [h_r, h_i]
                
                all_h = []
                for t in range(time_stamps):
                    current_state = in_states[t]
                    # 直接使用原始维度，不添加序列维度
                    out_real = block(current_state[0])  # [batch, dim, dim]
                    out_imag = block(current_state[1])  # [batch, dim, dim]
                    h = [out_real, out_imag]
                    all_h.append(h)
                
                in_states = all_h
        
        elif self.sequence_model == 'rwkv':
            # RWKV处理
            for block in self.seq_blocks:
                h_r = torch.stack(
                    batch_size*[torch.eye(self.embed_dim)/self.embed_dim], 
                    dim=0
                ).to(self.device)
                h_i = torch.zeros_like(h_r).to(self.device)
                h = [h_r, h_i]
                
                all_h = []
                for t in range(time_stamps):
                    current_state = in_states[t]
                    # 直接使用原始维度
                    out_real = block(current_state[0])  # [batch, dim, dim]
                    out_imag = block(current_state[1])  # [batch, dim, dim]
                    h = [out_real, out_imag]
                    all_h.append(h)
                
                in_states = all_h
        
        elif self.sequence_model == 'rnn':
            # RNN处理
            for block in self.seq_blocks:
                h_r = torch.stack(
                    batch_size*[torch.eye(self.embed_dim)/self.embed_dim], 
                    dim=0
                ).to(self.device)
                h_i = torch.zeros_like(h_r).to(self.device)
                h = [h_r, h_i]
                
                all_h = []
                for t in range(time_stamps):
                    current_state = in_states[t]
                    # 直接使用原始维度
                    out_real = block(current_state[0])  # [batch, dim, dim]
                    out_imag = block(current_state[1])  # [batch, dim, dim]
                    h = [out_real, out_imag]
                    all_h.append(h)
                
                in_states = all_h
        
        elif self.sequence_model == 'cnn':
            # CNN处理
            for block in self.seq_blocks:
                h_r = torch.stack(
                    batch_size*[torch.eye(self.embed_dim)/self.embed_dim], 
                    dim=0
                ).to(self.device)
                h_i = torch.zeros_like(h_r).to(self.device)
                h = [h_r, h_i]
                
                all_h = []
                for t in range(time_stamps):
                    current_state = in_states[t]
                    # 直接使用原始维度
                    out_real = block(current_state[0])  # [batch, dim, dim]
                    out_imag = block(current_state[1])  # [batch, dim, dim]
                    h = [out_real, out_imag]
                    all_h.append(h)
                
                in_states = all_h
        
        # 对每个时间步进行测量和输出
        output = []
        for _h in in_states:
            measurement_probs = self.measurement(_h)
            _output = self.fc_out(measurement_probs)
            output.append(_output)
        
        # 合并时间步维度
        output = torch.stack(output, dim=1)  # [batch_size, seq_len, n_classes]
        log_prob = F.log_softmax(output, dim=-1)
        
        return log_prob


