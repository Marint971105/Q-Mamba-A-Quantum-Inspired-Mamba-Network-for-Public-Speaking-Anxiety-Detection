# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from layers.quantumnn.embedding import PositionEmbedding
from layers.complexnn.multiply import ComplexMultiply
from layers.quantumnn.mixture import QMixture
from layers.quantumnn.measurement import QMeasurement
from layers.complexnn.measurement import ComplexMeasurement
from layers.quantumnn.outer import QOuter
from models.SimpleNet import SimpleNet
from layers.complexnn.l2_norm import L2Norm
from layers.realnn.mamba import MambaBlock
from layers.realnn.transformer import TransformerEncoder
from layers.realnn.rwkv import RWKVBlock
from layers.realnn.rnn import RNNBlock
from layers.realnn.cnn import CNNBlock
from layers.quantumnn.hadamard import Hadamard
from layers.quantumnn.cnot import QCNOT
from layers.quantumnn.attention import ModalAttention

class QMamba_meld(nn.Module):
    def __init__(self, opt):
        super(QMamba_meld, self).__init__()
        self.device = opt.device    
        self.input_dims = opt.input_dims  # [text:600, visual:300, acoustic:300]
        self.total_input_dim = sum(self.input_dims)
        self.embed_dim = opt.embed_dim
        self.n_classes = opt.output_dim  # 7分类
        self.num_layers = opt.num_layers
        self.sequence_model = opt.sequence_model
        
        # 模态投影层
        self.projections = nn.ModuleList([
            nn.Linear(dim, self.embed_dim) for dim in self.input_dims
        ])
        
        # 语言特征参数
        self.lang1_param = nn.Parameter(torch.randn(1, 1, self.input_dims[2]))  # 使用音频维度(300)
        self.lang2_param = nn.Parameter(torch.randn(1, 1, self.input_dims[2]))  # 使用音频维度(300)
        
        # 语言特征投影层
        self.lang_proj1 = nn.Linear(self.input_dims[2], self.embed_dim)  # 从音频维度投影
        self.lang_proj2 = nn.Linear(self.input_dims[2], self.embed_dim)  # 从音频维度投影
        
        # 量子门和其他组件
        self.multiply = ComplexMultiply()
        self.outer = QOuter()
        self.norm = L2Norm(dim=-1)
        self.mixture = QMixture(device=self.device)
        self.hadamard = Hadamard(embed_dim=self.embed_dim, device=self.device)
        self.cnot = QCNOT(self.embed_dim, device=self.device)
        
        # 序列模型
        if self.sequence_model == 'mamba':
            self.seq_blocks = nn.ModuleList([
                MambaBlock(self.embed_dim, device=self.device) 
                for _ in range(self.num_layers)
            ])
        elif self.sequence_model == 'transformer':
            self.seq_blocks = nn.ModuleList([
                TransformerEncoder(self.embed_dim, opt.num_heads, opt.dim_feedforward)
                for _ in range(self.num_layers)
            ])
        elif self.sequence_model == 'rwkv':
            self.seq_blocks = nn.ModuleList([
                RWKVBlock(self.embed_dim)
                for _ in range(self.num_layers)
            ])
        elif self.sequence_model == 'rnn':
            self.seq_blocks = nn.ModuleList([
                RNNBlock(self.embed_dim, opt.rnn_type, bidirectional=opt.bidirectional)
                for _ in range(self.num_layers)
            ])
        elif self.sequence_model == 'cnn':
            self.seq_blocks = nn.ModuleList([
                CNNBlock(self.embed_dim, opt.kernel_size)
                for _ in range(self.num_layers)
            ])
            
        # 测量和输出层
        self.measurement = QMeasurement(self.embed_dim)
        self.fc_out = SimpleNet(
            self.embed_dim, 
            opt.output_cell_dim,
            opt.out_dropout_rate,
            self.n_classes
        )
        
        # 模态注意力
        self.modal_attention = ModalAttention(
            embed_dim=self.embed_dim,
            num_heads=5,
            device=self.device
        )
        
        # 添加位置嵌入
        self.phase_embeddings = nn.ModuleList([
            PositionEmbedding(self.embed_dim, input_dim=1, device=self.device)
            for _ in range(len(self.input_dims))
        ])
        
        # 添加状态存储
        self.quantum_states = None  # 存储所有量子态
        self.entangled_states = None  # 存储纠缠态
        self.unimodal_matrices = None  # 存储模态矩阵
        self.mixed_state = None  # 存储混合态
        self.weights = None  # 存储权重

    def forward(self, in_modalities):
        batch_size = in_modalities[0].shape[0]
        time_stamps = in_modalities[0].shape[1]
        
        # 扩展语言特征参数到相同的时间步长
        lang1 = self.lang1_param.expand(batch_size, time_stamps, -1)  # [batch_size, time_stamps, acoustic_dim]
        lang2 = self.lang2_param.expand(batch_size, time_stamps, -1)  # [batch_size, time_stamps, acoustic_dim]
        
        # 投影所有模态
        utterance_reps = [
            nn.ReLU()(projection(x)) 
            for x, projection in zip(in_modalities, self.projections)
        ]
        
        # 投影语种特征
        lang_rep1 = nn.ReLU()(self.lang_proj1(lang1))  # [batch_size, time_stamps, embed_dim]
        lang_rep2 = nn.ReLU()(self.lang_proj2(lang2))  # [batch_size, time_stamps, embed_dim]
        
        # 计算振幅
        amplitudes = [F.normalize(rep, dim=-1) for rep in utterance_reps]
        lang_amp1 = F.normalize(lang_rep1, dim=-1)
        lang_amp2 = F.normalize(lang_rep2, dim=-1)
        
        # 生成相位
        smask = torch.ones(batch_size, time_stamps, 1).to(self.device)
        phases = [phase_embed(smask.argmax(dim=-1)) for phase_embed in self.phase_embeddings]
        
        # 构造基本量子态
        unimodal_pure = [
            self.multiply([phase, amplitude]) 
            for phase, amplitude in zip(phases, amplitudes)
        ]
        
        # 构造语言特征量子态并进行纠缠
        lang_state1 = self.multiply([phases[0], lang_amp1])
        lang_state2 = self.multiply([phases[0], lang_amp2])
        
        # 对语言特征应用Hadamard和CNOT
        h_lang1 = self.hadamard(lang_state1)
        h_lang2 = self.hadamard(lang_state2)
        entangled_lang = self.cnot(h_lang1, h_lang2)
        
        # 存储状态供测试使用
        self.quantum_states = unimodal_pure + [entangled_lang]
        self.entangled_states = [entangled_lang]
        
        # 计算外积
        unimodal_matrices = [self.outer(s) for s in unimodal_pure]
        lang_matrix = self.outer(entangled_lang)
        
        # 存储矩阵
        self.unimodal_matrices = unimodal_matrices + [lang_matrix]
        
        # 合并所有矩阵
        all_matrices = unimodal_matrices + [lang_matrix]
        
        # 计算权重
        all_reps = utterance_reps + [lang_rep1]
        weights = [self.norm(rep) for rep in all_reps]
        weights = F.softmax(torch.cat(weights, dim=-1), dim=-1)
        
        # 存储权重
        self.weights = weights
        
        # 混合量子态
        in_states = self.mixture([all_matrices, weights])
        self.mixed_state = in_states
        
        # 序列处理
        if self.sequence_model == 'mamba':
            # Mamba处理...
            for block in self.seq_blocks:
                h_r = torch.stack(batch_size*[torch.eye(self.embed_dim)/self.embed_dim],dim=0).to(self.device)
                h_i = torch.zeros_like(h_r).to(self.device)
                h = [h_r, h_i]
                
                all_h = []
                for t in range(time_stamps):
                    current_state = in_states[t]
                    h = block(current_state)
                    all_h.append(h)
                
                in_states = all_h
        elif self.sequence_model in ['transformer', 'rwkv', 'rnn', 'cnn']:
            for block in self.seq_blocks:
                h_r = torch.stack(batch_size*[torch.eye(self.embed_dim)/self.embed_dim],dim=0).to(self.device)
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
        
        # 测量和输出
        output = []
        for _h in in_states:
            measurement_probs = self.measurement(_h)
            _output = self.fc_out(measurement_probs)
            output.append(_output)
        
        # 合并时间步维度
        output = torch.stack(output, dim=1)  # [batch_size, seq_len, n_classes]
        log_prob = F.log_softmax(output, dim=-1)
        
        return log_prob 