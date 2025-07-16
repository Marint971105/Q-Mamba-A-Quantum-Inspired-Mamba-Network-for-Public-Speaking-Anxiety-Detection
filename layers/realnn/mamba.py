import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba

class MambaBlock(nn.Module):
    def __init__(self, 
                 d_model,      # 输入维度
                 d_state=16,   # 状态维度
                 d_conv=4,     # 卷积核大小
                 bimamba_type="v2",  # 双向类型：none, v1, v2
                 device=None,
                 dtype=None):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.bimamba_type = bimamba_type
        
        # 分别为实部和虚部创建Mamba模块
        self.mamba_r = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            bimamba_type=bimamba_type,
            device=device,
            dtype=dtype
        )
        
        self.mamba_i = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            bimamba_type=bimamba_type,
            device=device,
            dtype=dtype
        )
        
        # 分别为实部和虚部创建Layer Norm
        self.norm_r = nn.LayerNorm(d_model)
        self.norm_i = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: 量子态输入 [h_r, h_i]，每个都是 [batch_size, seq_len, d_model]
        Returns:
            out: 量子态输出 [h_r, h_i]
        """
        h_r, h_i = x  # 分解量子态
        
        # Layer Norm
        h_r = self.norm_r(h_r)
        h_i = self.norm_i(h_i)
        
        # Mamba处理
        out_r = self.mamba_r(h_r)
        out_i = self.mamba_i(h_i)
        
        return [out_r, out_i]  # 返回量子态格式

def test_mamba():
    """测试Mamba模块的功能"""
    print("\n=== Testing Mamba Block ===")
    
    # 详细检查CUDA状态
    print("\nCUDA Status:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    if not torch.cuda.is_available():
        print("Warning: Mamba requires CUDA. Please run this on a GPU.")
        return
    
    # 设置测试参数
    batch_size = 2
    seq_len = 1
    d_model = 50
    
    try:
        # 创建测试数据并确保在GPU上
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        print("\n1. Input data:")
        print(f"- shape: {x.shape}")
        print(f"- range: [{x.min():.4f}, {x.max():.4f}]")
        print(f"- mean: {x.mean():.4f}")
        print(f"- std: {x.std():.4f}")
        print(f"- device: {x.device}")
        
        # 测试不同类型的Mamba
        for bimamba_type in ["none", "v1", "v2"]:
            print(f"\n2. Testing {bimamba_type} Mamba:")
            
            # 创建Mamba块并确保在GPU上
            mamba_block = MambaBlock(
                d_model=d_model,
                d_state=16,
                bimamba_type=bimamba_type,
                device=device
            ).cuda()
            
            # 验证模型是否在GPU上
            print(f"- Model device: {next(mamba_block.parameters()).device}")
            
            # 前向传播
            out = mamba_block(x)
            
            print(f"- Output shape: {out.shape}")
            print(f"- Output range: [{out.min():.4f}, {out.max():.4f}]")
            print(f"- Output mean: {out.mean():.4f}")
            print(f"- Output std: {out.std():.4f}")
            print(f"- Output device: {out.device}")
            
            # 测试梯度
            loss = out.mean()
            loss.backward()
            
            # 检查参数梯度
            total_params = sum(p.numel() for p in mamba_block.parameters())
            grad_params = sum(p.numel() for p in mamba_block.parameters() if p.grad is not None)
            
            print(f"- Total parameters: {total_params}")
            print(f"- Parameters with gradients: {grad_params}")
            
            # 清除梯度
            for p in mamba_block.parameters():
                p.grad = None
                
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
            
    print("\nMamba Block test completed!")

if __name__ == "__main__":
    test_mamba() 