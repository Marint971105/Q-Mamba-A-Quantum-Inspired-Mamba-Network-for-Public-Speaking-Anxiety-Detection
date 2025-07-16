# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from models.SimpleNet import SimpleNet
from layers.realnn.mamba import MambaBlock


class BaselineMamba(nn.Module):
    """基线模型：纯Mamba + 传统L2Norm融合，不使用量子机制"""
    
    def __init__(self, opt):
        super(BaselineMamba, self).__init__()
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
        
        # 传统L2Norm用于模态融合
        self.modal_norm = nn.LayerNorm(self.embed_dim)
        
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
        
        # 3. 传统L2Norm融合（不使用量子机制）
        # 对每个模态进行L2归一化
        normalized_reps = [F.normalize(rep, dim=-1) for rep in utterance_reps]
        
        # 计算模态权重（基于L2范数）
        weights = [torch.norm(rep, dim=-1, keepdim=True) for rep in utterance_reps]
        weights = F.softmax(torch.cat(weights, dim=-1), dim=-1)
        
        # 加权融合
        fused_reps = []
        for t in range(time_stamps):
            # 在时间步t上融合所有模态
            fused = torch.zeros(batch_size, self.embed_dim).to(self.device)
            for i, rep in enumerate(normalized_reps):
                fused += weights[i][:, t:t+1] * rep[:, t, :]
            fused_reps.append(fused)
        
        # 4. Mamba序列处理
        in_states = fused_reps
        
        # Mamba处理
        for block in self.seq_blocks:
            all_h = []
            for t in range(time_stamps):
                current_state = in_states[t]
                # 构造量子态格式 [real, imag]
                # 实部使用当前状态，虚部使用零
                h_r = current_state.unsqueeze(1)  # [batch, 1, dim]
                h_i = torch.zeros_like(h_r)
                quantum_state = [h_r, h_i]
                
                # 传入量子态格式给Mamba
                h = block(quantum_state)
                all_h.append(h)
            
            in_states = all_h
        
        # 5. 输出处理
        output = []
        for _h in in_states:
            if isinstance(_h, list):
                # 如果是量子态，取实部
                _h = _h[0].squeeze(1)  # [batch, dim]
            _output = self.fc_out(_h)
            output.append(_output)
        
        # 合并时间步维度
        output = torch.stack(output, dim=1)  # [batch_size, seq_len, n_classes]
        log_prob = F.log_softmax(output, dim=-1)
        
        return log_prob 