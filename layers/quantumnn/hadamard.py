import torch
import torch.nn as nn

class Hadamard(nn.Module):
    def __init__(self, embed_dim, device):
        super(Hadamard, self).__init__()
        self.embed_dim = embed_dim
        self.device = device
        
        # Hadamard变换系数 1/√2
        self.h_coeff = 1.0 / torch.sqrt(torch.tensor(2.0))
        
    def forward(self, state):
        """
        实现Hadamard门操作: |0⟩ -> (|0⟩ + |1⟩)/√2, |1⟩ -> (|0⟩ - |1⟩)/√2
        
        Args:
            state: [real_part, imag_part] 每个part shape: [batch_size, embed_dim]
            
        Returns:
            transformed_state: [real_part, imag_part] Hadamard变换后的状态
        """
        real_part, imag_part = state
        
        # Hadamard变换
        transformed_real = self.h_coeff * (real_part + imag_part)
        transformed_imag = self.h_coeff * (real_part - imag_part)
        
        return [transformed_real, transformed_imag] 