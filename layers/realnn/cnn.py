import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBlock(nn.Module):
    def __init__(self, d_model, kernel_size=3, num_layers=1, dropout=0.1, device=None):
        super().__init__()
        self.d_model = d_model
        self.device = device
        
        # 创建多层1D CNN
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=d_model if i == 0 else d_model,
                out_channels=d_model,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ) for i in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Position-wise Feed Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x):
        """
        Args:
            x: 输入张量 [batch_size, d_model, d_model]
        Returns:
            output: 输出张量 [batch_size, d_model, d_model]
        """
        batch_size, H, W = x.shape
        assert H == W == self.d_model
        
        # 输入已经是 [batch_size, d_model, d_model] 格式
        residual = x
        
        # CNN处理
        for conv in self.convs:
            # 卷积
            x = conv(x)
            # 激活函数
            x = F.relu(x)
            # Dropout
            x = self.dropout(x)
        
        # 残差连接
        x = residual + x
        
        # 转换维度用于层归一化
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        
        # Position-wise Feed Forward
        # 转换维度以适应全连接层
        x = x.transpose(1, 2)
        x = self.ff(x)
        x = x.transpose(1, 2)
        
        return x 