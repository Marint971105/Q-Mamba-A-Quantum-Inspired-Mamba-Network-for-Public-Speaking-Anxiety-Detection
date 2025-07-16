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
from layers.quantumnn.embedding import PositionEmbedding


class QMambaPhaseComparison(nn.Module):
    """
    QMamba相位对比模型
    支持两种相位生成方式：
    - unified: 所有模态共享beats生成的相位（原始QMamba方式）
    - independent: 每个模态有独立的相位（类似QMN方式）
    """
    
    def __init__(self, opt):
        super(QMambaPhaseComparison, self).__init__()
        self.device = opt.device    
        self.input_dims = opt.input_dims[:3]  # 只使用text, audio, visual
        self.total_input_dim = sum(self.input_dims)
        self.embed_dim = opt.embed_dim
        self.n_classes = opt.output_dim
        self.num_layers = opt.num_layers
        self.sequence_model = opt.sequence_model
        self.weight_method = getattr(opt, 'weight_method', 'attention')
        
        # 新增：相位生成方式
        self.phase_method = getattr(opt, 'phase_method', 'unified')
        print(f"初始化QMambaPhaseComparison，相位方法: {self.phase_method}")
        
        # 模态投影层
        self.projections = nn.ModuleList([
            nn.Linear(dim, self.embed_dim) for dim in self.input_dims
        ])
        
        # beats投影层（用于统一相位）
        self.beats_projection = nn.Linear(768, self.embed_dim)
        
        # 独立相位嵌入层（用于独立相位）
        if self.phase_method == 'independent':
            self.phase_embeddings = nn.ModuleList([
                PositionEmbedding(self.embed_dim, input_dim=1, device=self.device)
                for _ in range(len(self.input_dims))
            ])
        
        # 添加可训练的语言特征参数
        self.lang1_param = nn.Parameter(torch.randn(1, 1, 768))
        self.lang2_param = nn.Parameter(torch.randn(1, 1, 768))
        
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
            self.modal_attention = ModalAttention(
                embed_dim=self.embed_dim,
                num_heads=4,  # 修改为能整除embed_dim的数
                device=self.device
            )
        elif self.weight_method == 'magnitude':
            self.modal_norm = L2Norm(dim=-1)
        else:
            raise ValueError(f"Unsupported weight_method: {self.weight_method}")
        
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
        """
        # 1. 对第一个语言状态应用Hadamard门
        state1_h = self.hadamard(state1)
        
        # 2. 使用CNOT门创建纠缠
        entangled_state = self.cnot(state1_h, state2)
        
        return entangled_state
        
    def forward(self, in_modalities):
        # 1. 提取特征
        text, audio, beats, visual = in_modalities[:4]
        modalities = [text, audio, visual]
        
        # 获取batch_size
        batch_size = text.shape[0]
        
        # 扩展语言特征参数到当前batch size
        lang1 = self.lang1_param.expand(batch_size, -1, -1)
        lang2 = self.lang2_param.expand(batch_size, -1, -1)
        
        # 2. 投影所有模态
        utterance_reps = [
            nn.ReLU()(projection(x)) 
            for x, projection in zip(modalities, self.projections)
        ]
        
        # 投影语种特征
        lang_rep1 = nn.ReLU()(self.lang_proj1(lang1))
        lang_rep2 = nn.ReLU()(self.lang_proj2(lang2))
        
        # 3. 生成相位 - 根据phase_method选择不同方式
        if self.phase_method == 'unified':
            # 相位统一性：所有模态共享beats生成的相位
            phase = nn.ReLU()(self.beats_projection(beats))
            phases = [phase] * len(utterance_reps)
            print(f"使用统一相位，相位形状: {phase.shape}")
        else:
            # 相位独立性：每个模态有独立的相位
            # 为每个模态生成正确形状的索引张量
            batch_size = text.shape[0]
            seq_len = text.shape[1]
            
            phases = []
            for i, phase_embed in enumerate(self.phase_embeddings):
                # 创建形状为 [batch_size, seq_len] 的索引张量
                modal_index = torch.full(
                    (batch_size, seq_len), i, 
                    dtype=torch.long, device=self.device
                )
                phase = phase_embed(modal_index)
                phases.append(phase)
            
            print(f"使用独立相位，相位数量: {len(phases)}")
        
        # 4. 计算振幅
        amplitudes = [F.normalize(rep, dim=-1) for rep in utterance_reps]
        lang_amp1 = F.normalize(lang_rep1, dim=-1)
        lang_amp2 = F.normalize(lang_rep2, dim=-1)
        
        # 5. 构造量子态
        quantum_states = [
            self.multiply([phase, amplitude]) 
            for phase, amplitude in zip(phases, amplitudes)
        ]
        
        # 构造语种量子态 - 使用第一个模态的相位
        lang_phase = phases[0] if self.phase_method == 'independent' else phase
        lang_state1 = self.multiply([lang_phase, lang_amp1])
        lang_state2 = self.multiply([lang_phase, lang_amp2])
        
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
        
        # 7. 计算模态权重
        if self.weight_method == 'attention':
            weights = self.modal_attention(utterance_reps)
        elif self.weight_method == 'magnitude':
            weights = [self.modal_norm(rep) for rep in utterance_reps]
            weights = F.softmax(torch.cat(weights, dim=-1), dim=-1)
        else:
            raise ValueError(f"Unsupported weight_method: {self.weight_method}")
        
        # 8. 混合更新后的量子态
        mixed_state = self.mixture([unimodal_matrices, weights])
        
        # 保存中间状态供分析
        self.quantum_states = quantum_states
        self.entangled_states = entangled_states
        self.phases = phases  # 保存相位信息用于分析
        
        # 9. 序列处理
        batch_size = in_modalities[0].shape[0]
        time_stamps = in_modalities[0].shape[1]
        
        in_states = mixed_state
        
        # 序列处理部分
        if self.sequence_model == 'mamba':
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
        
        elif self.sequence_model == 'transformer':
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
                    out_real = block(current_state[0])
                    out_imag = block(current_state[1])
                    h = [out_real, out_imag]
                    all_h.append(h)
                
                in_states = all_h
        
        elif self.sequence_model == 'rwkv':
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
                    out_real = block(current_state[0])
                    out_imag = block(current_state[1])
                    h = [out_real, out_imag]
                    all_h.append(h)
                
                in_states = all_h
        
        elif self.sequence_model == 'rnn':
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
                    out_real = block(current_state[0])
                    out_imag = block(current_state[1])
                    h = [out_real, out_imag]
                    all_h.append(h)
                
                in_states = all_h
        
        elif self.sequence_model == 'cnn':
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
                    out_real = block(current_state[0])
                    out_imag = block(current_state[1])
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
        output = torch.stack(output, dim=1)
        log_prob = F.log_softmax(output, dim=-1)
        
        return log_prob 