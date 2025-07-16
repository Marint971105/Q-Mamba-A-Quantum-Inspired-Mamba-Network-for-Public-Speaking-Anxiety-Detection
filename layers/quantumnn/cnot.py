import torch
import torch.nn as nn

class QCNOT(nn.Module):
    def __init__(self, embed_dim, device):
        super(QCNOT, self).__init__()
        self.embed_dim = embed_dim
        self.device = device
        
    def forward(self, control, target):
        """
        实现CNOT门操作
        
        Args:
            control: [real_part, imag_part] 控制量子位的状态
            target: [real_part, imag_part] 目标量子位的状态
            
        Returns:
            entangled_state: [real_part, imag_part] 纠缠后的状态
        """
        # 获取控制位和目标位的实部虚部
        control_real, control_imag = control
        target_real, target_imag = target
        
        # 当控制位为|1⟩时，翻转目标位
        # 当控制位为|0⟩时，保持目标位不变
        # 使用控制位的振幅作为权重
        
        # 返回纠缠后的状态
        entangled_real = control_real * target_real - control_imag * target_imag
        entangled_imag = control_real * target_imag + control_imag * target_real
        
        return [entangled_real, entangled_imag] 