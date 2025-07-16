# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.quantumnn.measurement import QMeasurement
class QAttention(torch.nn.Module):
    def __init__(self):
        super(QAttention, self).__init__()


    def forward(self, in_states, mea_states, mea_weights):
        

        out_states = []
        mea_r = mea_states[0]
        mea_i = mea_states[1]
        mea_mat_r = torch.matmul(mea_r.unsqueeze(dim = -1), mea_r.unsqueeze(dim = -2)) \
        + torch.matmul(mea_i.unsqueeze(dim = -1), mea_i.unsqueeze(dim = -2))
        mea_mat_i = torch.matmul(mea_r.unsqueeze(dim = -1), mea_r.unsqueeze(dim = -2)) \
        + torch.matmul(mea_i.unsqueeze(dim = -1), mea_i.unsqueeze(dim = -2))
        
        time_stamps = mea_r.shape[1]
        for s in in_states:
            s_r = s[0] # Real part
            s_i = s[1] # Imaginary part

            probs = []
            for i in range(time_stamps):
                m_r, m_i = mea_r[:,i,:].unsqueeze(dim = 1), mea_i[:,i,:].unsqueeze(dim = 1)
                prob = self.measurement(m_r, m_i, s_r, s_i)
                probs.append(prob)
                
            weights = torch.cat(probs,dim=-1)* mea_weights.squeeze(dim = -1)
            weights = F.softmax(weights, dim = -1)
            
            out_r = torch.sum(mea_mat_r * weights.unsqueeze(dim = -1).unsqueeze(dim = -1),dim = 1)
            out_i = torch.sum(mea_mat_i * weights.unsqueeze(dim = -1).unsqueeze(dim = -1),dim = 1)
            out_states.append([out_r, out_i])
        return out_states
            
    
    # p = (s_r+i*s_i)(rho_r +i*rho_i)(s_r' - i*s_i')
    def measurement(self, s_r, s_i, rho_r, rho_i):
        res_r = torch.matmul(s_r, rho_r) - torch.matmul(s_i,rho_i)
        res_i = torch.matmul(s_r, rho_i) + torch.matmul(s_i,rho_r)
         
        prob = torch.matmul(res_r, s_r.transpose(1,2)) + torch.matmul(res_i, s_i.transpose(1,2))
        #res_i_2 = - torch.matmul(res_r, rho_i.transpose(1,2)) + torch.matmul(res_i, rho_r.transpose(1,2))
    
        return prob.squeeze(dim = -1)

class ModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=5, dropout=0.1, device=torch.device('cpu')):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.device = device
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # 投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim).to(device)
        self.k_proj = nn.Linear(embed_dim, embed_dim).to(device)
        self.v_proj = nn.Linear(embed_dim, embed_dim).to(device)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, modal_reps):
        """
        计算模态间的注意力权重
        
        Args:
            modal_reps: List[Tensor], 每个tensor shape为[batch_size, seq_len, embed_dim]
            
        Returns:
            weights: [batch_size, seq_len, num_modalities]
        """
        batch_size, seq_len, _ = modal_reps[0].shape
        num_modalities = len(modal_reps)
        
        # 堆叠所有模态
        stacked_reps = torch.stack(modal_reps, dim=2)  # [batch_size, seq_len, num_modalities, embed_dim]
        
        # 投影并分头
        Q = self.q_proj(stacked_reps).view(batch_size, seq_len, num_modalities, self.num_heads, self.head_dim)
        K = self.k_proj(stacked_reps).view(batch_size, seq_len, num_modalities, self.num_heads, self.head_dim)
        V = self.v_proj(stacked_reps).view(batch_size, seq_len, num_modalities, self.num_heads, self.head_dim)
        
        # 转置以便进行注意力计算
        Q = Q.transpose(2, 3)  # [batch_size, seq_len, num_heads, num_modalities, head_dim]
        K = K.transpose(2, 3)
        V = V.transpose(2, 3)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 计算最终的模态权重 - 对所有头取平均，并对每个模态求和
        modal_weights = attn_weights.mean(dim=2).sum(dim=-2)  # [batch_size, seq_len, num_modalities]
        
        # 确保权重和为1
        modal_weights = F.softmax(modal_weights, dim=-1)
        
        return modal_weights

