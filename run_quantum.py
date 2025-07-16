import torch
import os
from types import SimpleNamespace
from dataset.features4quantum_reader import Features4QuantumReader
from models.QMN import QMN
from utils.model import train, test, save_model, save_performance
import numpy as np

def setup_params():
    """设置参数"""
    opt = SimpleNamespace(
        # 数据相关
        pickle_dir_path = '/home/tjc/audio/QMamba/feature_extract',
        features = 'textual,acoustic,beats,visual',
        dataset_name = 'features4quantum',
        embedding_enabled = False,
        dialogue_context = False,
        
        # 模型相关
        network_type = 'qmn',
        embed_dim = 50,              # 量子态维度
        speaker_num = 1,             # 单帧数据,不需要说话者信息
        output_dim = 4,              # 4分类
        output_cell_dim = 50,        # 输出层维度
        out_dropout_rate = 0.1,      # dropout率
        num_layers = 1,              # RNN层数
        
        # 训练相关
        gpu = 0,                     # GPU设备ID
        batch_size = 64,             # 批次大小
        epochs = 50,                 # 训练轮数
        lr = 0.001,                  # 学习率
        unitary_lr = 0.001,          # 酉矩阵学习率
        clip = 0.9,                  # 梯度裁剪
        patience = 10,               # 早停耐心值
        min_lr = 1e-5,              # 最小学习率
        
        # 设备
        device = None,               # 在run函数中设置
        
        # 保存相关
        save_dir = 'results/quantum4features',
        dir_name = None,
        output_file = None,
        config_file = 'config.ini',
        
        # 模型特定参数
        input_dims = [768, 768, 768, 768],  # 四个模态的输入维度
        total_input_dim = 768 * 4,          # 总输入维度
        
        # 类别平衡相关
        class_balance_strategy = 'weighted_loss',
    )
    return opt

def run(opt):
    """主运行函数"""
    # 设置GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)
    opt.device = torch.device(f'cuda:{opt.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # 创建保存目录
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
        
    print("=== 初始化数据加载器 ===")
    reader = Features4QuantumReader(opt)
    reader.read(opt)
    opt.reader = reader
    
    print(f"\n数据集信息:")
    print(f"训练集大小: {reader.train_sample_num}")
    print(f"特征维度: {reader.input_dims}")
    print(f"输出维度: {reader.output_dim}")
    
    print("\n=== 数据维度检查 ===")
    train_loader = reader.get_data(shuffle=False, split='train')
    batch = next(iter(train_loader))
    print("输入特征维度:")
    for i, feature in enumerate(batch[:-1]):
        print(f"模态{i}: {feature.shape}")
    print(f"标签维度: {batch[-1].shape}")
    
    print("\n=== 创建模型 ===")
    model = QMN(opt).to(opt.device)
    print(model)
    
    # 测试前向传播
    with torch.no_grad():
        inputs = [x.to(opt.device) for x in batch[:-1]]
        outputs = model(inputs)
        print(f"模型输出维度: {outputs.shape}")
        
    print("\n=== 开始训练 ===")
    # 设置临时模型保存路径
    opt.best_model_file = os.path.join(opt.save_dir, 'best_model.pth')
    
    # 训练模型
    train(opt, model)
    
    # 加载最佳模型并测试
    model = torch.load(opt.best_model_file)
    performance_dict = test(model, opt)
    
    print("\n=== 测试结果 ===")
    
    # 计算总体指标
    total_precision = np.mean([performance_dict[str(i)]['precision'] for i in range(4)])
    total_recall = np.mean([performance_dict[str(i)]['recall'] for i in range(4)])
    total_f1 = np.mean([performance_dict[str(i)]['f1-score'] for i in range(4)])
    
    # 打印总体结果
    print("\n总体性能:")
    print(f"Accuracy: {performance_dict['acc']:.4f}")
    print(f"Precision: {total_precision:.4f}")
    print(f"Recall: {total_recall:.4f}")
    print(f"F1-score: {total_f1:.4f}")
    
    # 打印宏平均和加权平均
    print("\n宏平均性能:")
    print(f"Macro Precision: {performance_dict['macro avg']['precision']:.4f}")
    print(f"Macro Recall: {performance_dict['macro avg']['recall']:.4f}")
    print(f"Macro F1-score: {performance_dict['macro avg']['f1-score']:.4f}")
    
    print("\n加权平均性能:")
    print(f"Weighted Precision: {performance_dict['weighted avg']['precision']:.4f}")
    print(f"Weighted Recall: {performance_dict['weighted avg']['recall']:.4f}")
    print(f"Weighted F1-score: {performance_dict['weighted avg']['f1-score']:.4f}")
    
    # 打印每个类别的结果
    print("\n各类别性能:")
    for i in range(4):
        print(f"\n类别 {i}:")
        print(f"Accuracy: {performance_dict[str(i)]['acc']:.4f}")
        print(f"Precision: {performance_dict[str(i)]['precision']:.4f}")
        print(f"Recall: {performance_dict[str(i)]['recall']:.4f}")
        print(f"F1-score: {performance_dict[str(i)]['f1-score']:.4f}")
    
    # 保存结果
    save_model(model, opt, str(performance_dict))
    save_performance(opt, performance_dict)
    
    # 保存详细结果到文件
    result_file = os.path.join(opt.save_dir, 'test_results.txt')
    with open(result_file, 'w') as f:
        f.write("=== 测试结果 ===\n")
        f.write(f"Overall Accuracy: {performance_dict['acc']:.4f}\n")
        
        f.write("\n各类别性能:\n")
        for i in range(4):
            f.write(f"\n类别 {i}:\n")
            f.write(f"Accuracy: {performance_dict[str(i)]['acc']:.4f}\n")
            f.write(f"Precision: {performance_dict[str(i)]['precision']:.4f}\n")
            f.write(f"Recall: {performance_dict[str(i)]['recall']:.4f}\n")
            f.write(f"F1-score: {performance_dict[str(i)]['f1-score']:.4f}\n")
        
        f.write("\n平均性能:\n")
        f.write("Macro Average:\n")
        f.write(f"F1-score: {performance_dict['macro avg']['f1-score']:.4f}\n")
        f.write(f"Accuracy: {performance_dict['macro avg']['acc']:.4f}\n")
        f.write("\nWeighted Average:\n")
        f.write(f"F1-score: {performance_dict['weighted avg']['f1-score']:.4f}\n")
        f.write(f"Accuracy: {performance_dict['weighted avg']['acc']:.4f}\n")
    
    print(f"\n详细结果已保存到: {result_file}")
    
    return model, performance_dict

if __name__ == "__main__":
    opt = setup_params()
    model, results = run(opt)