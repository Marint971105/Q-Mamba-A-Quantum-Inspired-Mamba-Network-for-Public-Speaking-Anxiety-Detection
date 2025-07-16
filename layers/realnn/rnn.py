import torch
import torch.nn as nn

class RNNBlock(nn.Module):
    def __init__(self, d_model, rnn_type='lstm', num_layers=1, bidirectional=False, dropout=0.1, device=None):
        super().__init__()
        self.d_model = d_model
        self.rnn_type = rnn_type.lower()
        self.device = device
        
        # 选择RNN类型
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            self.rnn = nn.RNN(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            
        # 如果是双向的,需要将输出转换回原始维度
        self.bidirectional = bidirectional
        if bidirectional:
            self.proj = nn.Linear(d_model * 2, d_model)
            
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: 输入张量 [batch_size, d_model, d_model]
        Returns:
            output: 输出张量 [batch_size, d_model, d_model]
        """
        batch_size, H, W = x.shape
        assert H == W == self.d_model
        
        # 重塑输入以适应RNN
        x = x.view(batch_size, self.d_model, -1)  # [batch, d_model, d_model]
        x = x.transpose(1, 2)  # [batch, d_model, d_model]
        
        # RNN处理
        output, _ = self.rnn(x)
        
        # 如果是双向的,将输出投影回原始维度
        if self.bidirectional:
            output = self.proj(output)
            
        # 重塑回原始维度
        output = output.transpose(1, 2)  # [batch, d_model, d_model]
        output = output.contiguous()
        
        # 残差连接和层归一化
        output = x + output
        output = self.layer_norm(output)
        
        return output 