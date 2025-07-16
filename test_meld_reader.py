import torch
from types import SimpleNamespace
from dataset.meld_reader import MELDReader

def print_sample_shapes():
    # 配置参数
    opt = SimpleNamespace(
        pickle_dir_path = '/home/tjc/audio/Quantum/Data',  # 使用与run.ini相同的路径
        features = 'textual,acoustic,visual',  # 使用所有模态
        dialogue_context = False,  # 不使用对话上下文
        label = 'emotion',  # 情感标签
        batch_size = 1,  # 只看一个样本
        dataset_name = 'meld',  # 添加数据集名称
        embedding_enabled = False  # 添加embedding参数
    )

    # 创建数据读取器
    reader = MELDReader(opt)
    reader.read(opt)

    # 获取一个训练样本
    train_loader = reader.get_data(shuffle=False, split='train')
    batch = next(iter(train_loader))

    # 打印每个部分的shape
    print("\n=== MELD样本形状 ===")
    print(f"数据集大小: {reader.train_sample_num}")
    print(f"最大序列长度: {reader.max_seq_len}")
    print(f"说话者数量: {reader.speaker_num}")
    
    print("\n=== 输入特征 ===")
    print(f"文本特征: {batch[0].shape}")  # [batch_size, seq_len, text_dim]
    print(f"声学特征: {batch[1].shape}")  # [batch_size, seq_len, acoustic_dim]
    print(f"视觉特征: {batch[2].shape}")  # [batch_size, seq_len, visual_dim]
    print(f"说话者掩码: {batch[3].shape}")  # [batch_size, seq_len, speaker_num]
    print(f"对话掩码: {batch[4].shape}")  # [batch_size, seq_len]
    
    print("\n=== 标签 ===")
    print(f"情感标签: {batch[-1].shape}")  # [batch_size, seq_len, num_emotions]
    
    # 打印一个具体的样本内容
    print("\n=== 样本示例 ===")
    print("说话者掩码示例:")
    print(batch[3][0][:5])  # 打印前5个时间步的说话者信息
    print("\n对话掩码示例(1表示有效句子):")
    print(batch[4][0][:5])  # 打印前5个时间步的掩码

if __name__ == "__main__":
    print_sample_shapes()