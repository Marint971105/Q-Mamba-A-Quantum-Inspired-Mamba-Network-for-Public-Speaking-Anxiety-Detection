# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from models.SimpleNet import SimpleNet
from layers.realnn.mamba import MambaBlock
from layers.complexnn.multiply import ComplexMultiply
from layers.quantumnn.outer import QOuter
from layers.quantumnn.measurement import QMeasurement


class QuantumSuperposition(nn.Module):
    """量子叠加模型：基线 + 量子叠加状态建模"""
    
    def __init__(self, opt):
        super(QuantumSuperposition, self).__init__()
        self.device = opt.device    
        self.input_dims = opt.input_dims[:3]  # 只使用text, audio, visual
        self.total_input_dim = sum(self.input_dims)
        self.embed_dim = opt.embed_dim
        self.n_classes = opt.output_dim
        self.num_layers = opt.num_layers
        
        # 模态投影层
        self.projections = nn.ModuleList([
            nn.Linear(dim, self.embed_dim) for dim in self.input_dims
        ])
        
        # beats投影层（用于生成相位）
        self.beats_projection = nn.Linear(768, self.embed_dim)
        
        # 量子组件
        self.multiply = ComplexMultiply()
        self.outer = QOuter()
        self.measurement = QMeasurement(self.embed_dim)
        
        # 输出层
        self.out_dropout_rate = opt.out_dropout_rate
        self.fc_out = SimpleNet(
            self.embed_dim, 
            opt.output_cell_dim,
            self.out_dropout_rate,
            self.n_classes,
            output_activation=nn.Tanh()
        )
        
        # Mamba序列模型
        self.seq_blocks = nn.ModuleList([
            MambaBlock(
                d_model=self.embed_dim,
                d_state=16,
                bimamba_type="v2",
                device=self.device
            ) for _ in range(self.num_layers)
        ])
        
    def forward(self, in_modalities):
        # 1. 提取特征
        text, audio, beats, visual = in_modalities[:4]  # 只取前4个模态
        modalities = [text, audio, visual]
        
        # 获取batch_size和time_stamps
        batch_size = text.shape[0]
        time_stamps = text.shape[1]
        
        # 2. 投影所有模态
        utterance_reps = [
            nn.ReLU()(projection(x)) 
            for x, projection in zip(modalities, self.projections)
        ]
        
        # 3. 生成相位（使用beats）
        phase = nn.ReLU()(self.beats_projection(beats))
        
        # 4. 量子叠加状态建模
        # 计算振幅（L2归一化）
        amplitudes = [F.normalize(rep, dim=-1) for rep in utterance_reps]
        
        # 构造量子态（相位 + 振幅）
        quantum_states = [
            self.multiply([phase, amplitude]) 
            for amplitude in amplitudes
        ]
        
        # 计算外积（量子叠加）
        unimodal_matrices = [
            self.outer(quantum_state) 
            for quantum_state in quantum_states
        ]
        
        # 5. 传统权重计算（基于L2范数）
        # 计算每个模态的权重
        modal_weights = []
        for rep in utterance_reps:
            weight = torch.norm(rep, dim=-1, keepdim=True)  # [batch, seq_len, 1]
            modal_weights.append(weight)
        
        # 拼接并归一化权重
        weights = torch.cat(modal_weights, dim=-1)  # [batch, seq_len, num_modalities]
        weights = F.softmax(weights, dim=-1)  # [batch, seq_len, num_modalities]
        
        # 6. 量子叠加融合
        fused_quantum_states = []
        for t in range(time_stamps):
            # 在时间步t上融合所有量子态
            fused_real = torch.zeros(batch_size, self.embed_dim, self.embed_dim).to(self.device)
            fused_imag = torch.zeros(batch_size, self.embed_dim, self.embed_dim).to(self.device)
            
            for i, matrix_list in enumerate(unimodal_matrices):
                # matrix_list[t] 是 [real, imag] 格式
                matrix_real = matrix_list[t][0]  # [batch, dim, dim]
                matrix_imag = matrix_list[t][1]  # [batch, dim, dim]
                
                # 使用正确的权重索引
                weight_t = weights[:, t:t+1, i:i+1]  # [batch, 1, 1]
                fused_real += weight_t * matrix_real
                fused_imag += weight_t * matrix_imag
            
            # 构造融合后的量子态
            fused_quantum_states.append([fused_real, fused_imag])
        
        # 7. Mamba序列处理
        in_states = fused_quantum_states
        
        # Mamba处理
        for block in self.seq_blocks:
            all_h = []
            for t in range(time_stamps):
                current_state = in_states[t]  # [real, imag]
                # 将矩阵转换为向量（取对角线）
                vector_real = torch.diagonal(current_state[0], dim1=1, dim2=2)  # [batch, dim]
                vector_imag = torch.diagonal(current_state[1], dim1=1, dim2=2)  # [batch, dim]
                
                # 构造量子态格式 [real, imag]
                h_r = vector_real.unsqueeze(1)  # [batch, 1, dim]
                h_i = vector_imag.unsqueeze(1)  # [batch, 1, dim]
                quantum_state = [h_r, h_i]
                
                # 传入量子态格式给Mamba
                h = block(quantum_state)
                all_h.append(h)
            
            in_states = all_h
        
        # 8. 量子测量和输出处理
        output = []
        for _h in in_states:
            if isinstance(_h, list):
                # 量子态测量 - 需要构造矩阵格式
                # 从向量构造矩阵 [batch, 1, dim] -> [batch, dim, dim]
                h_r = _h[0].squeeze(1)  # [batch, dim]
                h_i = _h[1].squeeze(1)  # [batch, dim]
                
                # 构造矩阵格式 - 对每个batch单独计算外积
                batch_size = h_r.shape[0]
                matrix_r = torch.zeros(batch_size, self.embed_dim, self.embed_dim).to(self.device)
                matrix_i = torch.zeros(batch_size, self.embed_dim, self.embed_dim).to(self.device)
                
                for b in range(batch_size):
                    matrix_r[b] = torch.outer(h_r[b], h_r[b])  # [dim, dim]
                    matrix_i[b] = torch.outer(h_i[b], h_i[b])  # [dim, dim]
                
                # 传入矩阵格式给测量层
                measurement_input = [matrix_r, matrix_i]
                measurement_probs = self.measurement(measurement_input)
                _output = self.fc_out(measurement_probs)
            else:
                # 直接处理
                _output = self.fc_out(_h)
            output.append(_output)
        
        # 合并时间步维度
        output = torch.stack(output, dim=1)  # [batch_size, seq_len, n_classes]
        log_prob = F.log_softmax(output, dim=-1)
        
        return log_prob 