import torch
from layers.realnn.transformer import TransformerEncoder

def test_transformer_real():
    # 使用真实数据的维度
    batch_size = 64
    seq_len = 1
    embed_dim = 50
    num_heads = 5  # 与QMamba中相同
    
    # 创建模型
    transformer = TransformerEncoder(
        embed_dim=embed_dim,
        num_heads=num_heads,
        layers=1,
        attn_dropout=0.1,
        relu_dropout=0.1,
        res_dropout=0.1,
        embed_dropout=0.1,
        attn_mask=False
    )
    
    # 创建模拟的实部和虚部数据
    real_states = torch.randn(seq_len, batch_size, embed_dim)  # [1, 64, 50]
    imag_states = torch.randn(seq_len, batch_size, embed_dim)  # [1, 64, 50]
    
    print("=== 输入维度 ===")
    print("real_states:", real_states.shape)
    print("imag_states:", imag_states.shape)
    
    try:
        # 分别处理实部和虚部
        transformed_real = transformer(real_states)
        transformed_imag = transformer(imag_states)
        
        print("\n=== 输出维度 ===")
        print("transformed_real:", transformed_real.shape)
        print("transformed_imag:", transformed_imag.shape)
        
        # 转换回QMamba需要的格式
        transformed_real = transformed_real.transpose(0, 1)  # [64, 1, 50]
        transformed_imag = transformed_imag.transpose(0, 1)  # [64, 1, 50]
        
        print("\n=== 转换后维度 ===")
        print("final_real:", transformed_real.shape)
        print("final_imag:", transformed_imag.shape)
        
        # 重新组合实部和虚部
        in_states = [(transformed_real[:, t, :], transformed_imag[:, t, :]) 
                    for t in range(transformed_real.size(1))]
        
        print("\n=== 最终状态 ===")
        print("状态数量:", len(in_states))
        print("每个状态的实部维度:", in_states[0][0].shape)
        print("每个状态的虚部维度:", in_states[0][1].shape)
        
        print("\n测试成功!")
        
    except Exception as e:
        print("\n测试失败:", str(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_transformer_real()