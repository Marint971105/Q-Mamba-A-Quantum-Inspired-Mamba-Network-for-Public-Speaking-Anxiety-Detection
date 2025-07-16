import pickle
import numpy as np
import os
from tqdm import tqdm

def print_sample_details(sample, sample_idx=None):
    """
    打印单个样本的详细信息
    
    Args:
        sample: 样本字典
        sample_idx: 样本索引（可选）
    """
    print("\n" + "="*50)
    if sample_idx is not None:
        print(f"样本索引: {sample_idx}")
    print("="*50)
    
    # 打印基本信息
    print("\n基本信息:")
    print(f"ID: {sample['id']}")
    print(f"路径: {sample['path']}")
    print(f"标签: {sample['label']}")
    
    # 打印各个特征的详细信息
    feature_keys = ['text_features', 'audio_features', 'beats_features', 'video_features']
    print("\n特征信息:")
    for key in feature_keys:
        if key in sample:
            feature = sample[key]
            print(f"\n{key}:")
            print(f"形状: {feature.shape}")
            print(f"类型: {feature.dtype}")
            print(f"最小值: {feature.min():.4f}")
            print(f"最大值: {feature.max():.4f}")
            print(f"均值: {feature.mean():.4f}")
            print(f"标准差: {feature.std():.4f}")
            print(f"前5个值: {feature[:5]}")
        else:
            print(f"\n{key}: 不存在")

def load_and_check_features(feature_dir, feature_file="features.pkl", sample_idx_to_check=None):
    """
    加载并检查特征文件
    
    Args:
        feature_dir: 特征文件所在的目录名称 (例如: "chinese_test")
        feature_file: 特征文件名称，默认为"features.pkl"
        sample_idx_to_check: 要详细查看的样本索引（可选）
    """
    # 构建完整的文件路径
    base_dir = "Features4Quantum"
    file_path = os.path.join(base_dir, feature_dir, feature_file)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"特征文件不存在: {file_path}")
    
    # 加载特征文件
    print(f"正在加载特征文件: {file_path}")
    with open(file_path, 'rb') as f:
        features = pickle.load(f)
    
    print(f"\n总样本数: {len(features)}")
    
    # 检查第一个样本的结构
    if len(features) > 0:
        first_sample = features[0]
        print("\n特征字典结构:")
        for key, value in first_sample.items():
            if isinstance(value, np.ndarray):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"{key}: {type(value)}")
    
    # 检查所有样本的完整性
    print("\n检查所有样本...")
    missing_features = []
    feature_shapes = {
        'text_features': None,
        'audio_features': None,
        'beats_features': None,
        'video_features': None
    }
    
    for idx, sample in enumerate(tqdm(features)):
        # 检查必要的键是否存在
        required_keys = ['id', 'path', 'label', 'text_features', 'audio_features', 
                        'beats_features', 'video_features']
        
        for key in required_keys:
            if key not in sample:
                missing_features.append((idx, f"缺失键: {key}"))
                continue
        
        # 检查特征的形状
        for key in feature_shapes.keys():
            if key in sample and isinstance(sample[key], np.ndarray):
                if feature_shapes[key] is None:
                    feature_shapes[key] = sample[key].shape
                elif sample[key].shape != feature_shapes[key]:
                    missing_features.append(
                        (idx, f"{key} 形状不匹配: 预期 {feature_shapes[key]}, 实际 {sample[key].shape}")
                    )
    
    # 打印检查结果
    if missing_features:
        print("\n发现以下问题:")
        for idx, problem in missing_features:
            print(f"样本 {idx}: {problem}")
    else:
        print("\n所有样本检查通过!")
    
    # 打印特征维度信息
    print("\n特征维度信息:")
    for key, shape in feature_shapes.items():
        print(f"{key}: {shape}")
    
    # 如果指定了样本索引，打印该样本的详细信息
    if sample_idx_to_check is not None:
        if 0 <= sample_idx_to_check < len(features):
            print_sample_details(features[sample_idx_to_check], sample_idx_to_check)
        else:
            print(f"\n错误：指定的样本索引 {sample_idx_to_check} 超出范围 [0, {len(features)-1}]")
    
    return features

if __name__ == "__main__":
    try:
        # 检查训练集特征
        # print("检查训练集特征...")
        # train_features = load_and_check_features("chinese_train", sample_idx_to_check=0)  # 查看第一个样本的详细信息
        
        # 检查测试集特征
        print("\n检查测试集特征...")
        test_features = load_and_check_features("chinese_test", sample_idx_to_check=0)  # 查看第一个样本的详细信息
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()