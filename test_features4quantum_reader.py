import torch
from types import SimpleNamespace
from dataset.features4quantum_reader import Features4QuantumReader

def print_sample_shapes():
    # 配置参数
    opt = SimpleNamespace(
        pickle_dir_path = '/home/tjc/audio/QMamba/feature_extract',  # 特征文件根目录
        features = 'textual,acoustic,beats,visual',  # 使用所有特征
        batch_size = 1,
        dataset_name = 'features4quantum',
        embedding_enabled = False
    )

    # 创建数据读取器
    reader = Features4QuantumReader(opt)
    reader.read(opt)

    # 获取一个训练样本
    train_loader = reader.get_data(shuffle=False, split='train')
    batch = next(iter(train_loader))

    # 打印形状信息
    print("\n=== Features4Quantum样本形状 ===")
    print(f"数据集大小: {reader.train_sample_num}")
    
    print("\n=== 输入特征 ===")
    print(f"文本特征: {batch[0].shape}")  # [batch_size, 1, 768]
    print(f"声学特征: {batch[1].shape}")  # [batch_size, 1, 768]
    print(f"BEATs特征: {batch[2].shape}")  # [batch_size, 1, 768]
    print(f"视觉特征: {batch[3].shape}")  # [batch_size, 1, 768]
    
    print("\n=== 标签 ===")
    print(f"情感标签: {batch[-1].shape}")  # [batch_size, 4]
    
    # 打印特征维度信息
    print("\n=== 特征维度 ===")
    for i, dim in enumerate(reader.input_dims):
        feature_name = reader.all_feature_names[reader.feature_indexes[i]]
        print(f"{feature_name}: {dim}")

if __name__ == "__main__":
    print_sample_shapes() 