import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RWKVBlock(nn.Module):
    def __init__(self, d_model, d_state=16, device=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.device = device
        
        # Time-mixing parameters
        self.time_decay = nn.Parameter(torch.ones(d_model))
        self.time_first = nn.Parameter(torch.ones(d_model))
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model))
        
        # Key, Value, Receptance projections
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.receptance = nn.Linear(d_model, d_model)
        
        # Output projection
        self.output = nn.Linear(d_model, d_model)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model * d_model)
        self.ln2 = nn.LayerNorm(d_model * d_model)
        
        # FFN
        self.ffn_time_mix = nn.Parameter(torch.ones(1, 1, d_model))
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        
    def forward(self, x):
        """
        Args:
            x: 输入张量 [batch_size, d_model, d_model]
        Returns:
            output: 输出张量 [batch_size, d_model, d_model]
        """
        B, H, W = x.shape
        assert H == W == self.d_model
        
        # 展平处理
        x_flat = x.view(B, -1)  # [batch_size, d_model*d_model]
        x_norm = self.ln1(x_flat)
        x_norm = x_norm.view(B, H, W)  # [batch_size, d_model, d_model]
        
        # 时间混合
        k = self.key(x_norm * self.time_mix_k)
        v = self.value(x_norm * self.time_mix_v)
        r = self.receptance(x_norm * self.time_mix_r)
        r = torch.sigmoid(r)
        
        # WKV注意力机制
        wkv = k * v
        rwkv = r * wkv
        
        # 第一个残差连接
        x = x + rwkv
        
        # FFN
        x_flat = x.view(B, -1)
        x_norm = self.ln2(x_flat)
        x_norm = x_norm.view(B, H, W)
        
        x = x + self.ffn(x_norm * self.ffn_time_mix)
        
        return x 