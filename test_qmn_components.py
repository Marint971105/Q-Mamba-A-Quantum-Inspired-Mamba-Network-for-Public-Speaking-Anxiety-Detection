import torch
from models.QMN import QMN

class Config:
    def __init__(self):
        # 基本参数
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dims = [768, 768, 768]  # text, audio, visual维度
        self.embed_dim = 50
        self.output_dim = 4  # 输出类别数
        self.output_cell_dim = 128
        self.out_dropout_rate = 0.1
        
        # QMN特有参数
        self.num_layers = 1  # QRNN层数
        self.speaker_num = 2  # 说话者数量
        self.dataset_name = 'IEMOCAP'  # 数据集名称
        self.use_bert = False
        self.use_wav2vec = False
        self.use_visual = True
        self.fusion_type = 'sum'
        self.activation = 'tanh'
        self.use_speaker = True
        self.use_pos = False
        self.attn_mask = True
        self.update_weights = True
        self.norm_first = True
        self.use_common_space = False
        self.common_space_dim = 50
        self.modality_weights = [1.0, 1.0, 1.0]  # text, audio, visual权重
        self.use_h = True
        self.use_local = False
        self.use_last = True
        self.use_residual = True

def test_qmn_components():
    print("\n=== QMN模型组件测试 ===\n")
    
    # 1. 配置参数
    opt = Config()
    
    # 2. 创建模型
    model = QMN(opt).to(opt.device)
    print('model',model)
    # 3. 准备测试数据
    batch_size = 2
    seq_len = 1
    
    # 创建模态数据
    text = torch.randn(batch_size, seq_len, 768).to(opt.device)
    audio = torch.randn(batch_size, seq_len, 768).to(opt.device)
    visual = torch.randn(batch_size, seq_len, 768).to(opt.device)
    
    # 创建speaker mask
    speaker_mask = torch.zeros(batch_size, seq_len, 2).to(opt.device)  # 假设有2个说话者
    speaker_mask[:, :, 0] = 1  # 设置第一个说话者为活跃
    
    # 打印输入数据信息
    print("1. 输入数据:\n")
    for name, data in [('text', text), ('audio', audio), ('visual', visual)]:
        print(f"{name}:")
        print(f"- shape: {data.shape}")
        print(f"- 数值范围: [{data.min():.4f}, {data.max():.4f}]")
        print(f"- 均值: {data.mean():.4f}")
        print(f"- 标准差: {data.std():.4f}\n")
    
    # 4. 前向传播
    in_modalities = [text, audio, visual, None, speaker_mask, None]
    output = model(in_modalities)
    
    # 5. 打印输出信息
    print("\n最终输出:")
    print(f"- shape: {output.shape}")
    print(f"- 数值范围: [{output.min():.4f}, {output.max():.4f}]")
    print(f"- 均值: {output.mean():.4f}")
    print(f"- 标准差: {output.std():.4f}")

if __name__ == "__main__":
    test_qmn_components()
