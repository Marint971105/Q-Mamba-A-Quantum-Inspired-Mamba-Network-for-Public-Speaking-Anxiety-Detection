# -*- coding: utf-8 -*-
import os
import torch
from types import SimpleNamespace
from dataset.features4quantum_reader import Features4QuantumReader
from models.QMamba import QMamba
from utils.model import train, test, save_model, save_performance
import time
import json


def convert_tensor_to_serializable(obj):
    """将tensor转换为可序列化的格式"""
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        else:
            return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_tensor_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensor_to_serializable(item) for item in obj]
    else:
        return obj


def setup_params(weight_method='attention', sequence_model='mamba'):
    """设置参数
    
    Args:
        weight_method: 权重计算方式 ('attention' 或 'magnitude')
        sequence_model: 序列模型类型
    """
    opt = SimpleNamespace(
        # 数据相关
        pickle_dir_path='/home/tjc/audio/QMamba/feature_extract',
        features='textual,acoustic,beats,visual',
        dataset_name='features4quantum',  # CN数据集
        embedding_enabled=False,
        dialogue_context=False,

        # 模型架构
        network_type='qmamba',
        sequence_model=sequence_model,
        weight_method=weight_method,  # 新增：权重计算方式
        
        # CNN特有参数
        kernel_size=3,
        cnn_dropout=0.1,
        
        # RNN特有参数
        rnn_type='rnn',
        bidirectional=False,
        rnn_dropout=0.1,
        
        # Transformer特有参数
        num_heads=2,
        attn_dropout=0.1,
        relu_dropout=0.1,
        res_dropout=0.1,
        embed_dropout=0.1,
        
        # 模型参数
        embed_dim=50,
        num_layers=1,
        output_cell_dim=50,
        output_dim=4,
        speaker_num=1,

        # 训练参数
        epochs=50,
        patience=10,
        batch_size=64,
        lr=0.0005,
        unitary_lr=0.001,
        out_dropout_rate=0.2,
        clip=1.0,
        min_lr=1e-5,

        # 其他设置
        gpu=0,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        save_dir='results/qmamba_ablation',
        output_file=None,
        config_file='config.ini'
    )
    return opt


def run_experiment(weight_method, sequence_model='mamba'):
    """运行单个实验
    
    Args:
        weight_method: 权重计算方式
        sequence_model: 序列模型类型
    
    Returns:
        dict: 实验结果
    """
    print(f"\n{'='*50}")
    print(f"开始实验: weight_method={weight_method}, sequence_model={sequence_model}")
    print(f"{'='*50}")
    
    # 获取参数
    opt = setup_params(weight_method, sequence_model)
    
    # 设置输出文件名
    timestamp = int(time.time())
    opt.output_file = f"{opt.save_dir}/results_{weight_method}_{sequence_model}_{timestamp}.json"
    opt.best_model_file = f"{opt.save_dir}/best_model_{weight_method}_{sequence_model}_{timestamp}.pth"
    
    # 创建保存目录
    os.makedirs(opt.save_dir, exist_ok=True)
    
    print(f"实验配置:")
    print(f"  权重计算方式: {opt.weight_method}")
    print(f"  序列模型: {opt.sequence_model}")
    print(f"  设备: {opt.device}")
    print(f"  批大小: {opt.batch_size}")
    print(f"  学习率: {opt.lr}")
    print(f"  嵌入维度: {opt.embed_dim}")
    print(f"  输出文件: {opt.output_file}")
    
    # 初始化数据读取器
    reader = Features4QuantumReader(opt)
    opt.reader = reader

    # 设置输入维度
    opt.input_dims = reader.input_dims
    opt.total_input_dim = sum(opt.input_dims[:3])

    # 初始化模型
    model = QMamba(opt).to(opt.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总量: {total_params}")

    # 训练模型
    print("\n开始训练...")
    train_start_time = time.time()
    train(opt, model)
    train_time = time.time() - train_start_time

    # 加载最佳模型
    best_model = QMamba(opt).to(opt.device)
    best_model.load_state_dict(torch.load(opt.best_model_file))

    # 测试模型
    print("\n开始测试...")
    test_start_time = time.time()
    test_performance = test(best_model, opt)
    test_time = time.time() - test_start_time

    # 保存模型和性能结果
    save_model(best_model, opt, test_performance)
    save_performance(opt, test_performance)

    # 删除临时文件
    if os.path.exists(opt.best_model_file):
        os.remove(opt.best_model_file)

    # 整理实验结果
    result = {
        'weight_method': weight_method,
        'sequence_model': sequence_model,
        'total_params': total_params,
        'train_time': train_time,
        'test_time': test_time,
        'performance': convert_tensor_to_serializable(test_performance),
        'config': {
            'embed_dim': opt.embed_dim,
            'num_layers': opt.num_layers,
            'batch_size': opt.batch_size,
            'lr': float(opt.lr),  # 转换为Python标量
            'epochs': opt.epochs
        }
    }
    
    print(f"\n实验完成:")
    print(f"  训练时间: {train_time:.2f}秒")
    print(f"  测试时间: {test_time:.2f}秒")
    print(f"  测试性能: {test_performance}")
    print(f"  结果已保存到: {opt.output_file}")
    
    return result


def run_ablation_study():
    """运行消融实验"""
    print("开始QMamba权重计算方式消融实验")
    print("="*60)
    
    # 实验配置
    weight_methods = ['attention', 'magnitude']
    sequence_models = ['mamba', 'transformer', 'cnn']
    
    # 存储所有实验结果
    all_results = []
    
    # 运行所有实验组合
    for weight_method in weight_methods:
        for sequence_model in sequence_models:
            try:
                result = run_experiment(weight_method, sequence_model)
                all_results.append(result)
            except Exception as e:
                print(f"实验失败: weight_method={weight_method}, sequence_model={sequence_model}")
                print(f"错误信息: {str(e)}")
                continue
    
    # 保存汇总结果
    summary_file = f"results/qmamba_ablation/summary_{int(time.time())}.json"
    os.makedirs("results/qmamba_ablation", exist_ok=True)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(
            convert_tensor_to_serializable(all_results), 
            f, indent=2, ensure_ascii=False
        )
    
    # 打印汇总结果
    print("\n" + "="*60)
    print("消融实验汇总结果")
    print("="*60)
    
    for result in all_results:
        print(f"\n{result['weight_method']} + {result['sequence_model']}:")
        print(f"  参数数量: {result['total_params']}")
        print(f"  训练时间: {result['train_time']:.2f}秒")
        print(f"  测试时间: {result['test_time']:.2f}秒")
        print(f"  性能: {result['performance']}")
    
    print(f"\n汇总结果已保存到: {summary_file}")
    print("="*60)


def run_single_comparison():
    """运行单一比较实验（只比较权重计算方式）"""
    print("开始权重计算方式比较实验")
    print("="*50)
    
    # 只比较权重计算方式，使用mamba作为序列模型
    weight_methods = ['attention', 'magnitude']
    all_results = []
    
    for weight_method in weight_methods:
        try:
            result = run_experiment(weight_method, 'mamba')
            all_results.append(result)
        except Exception as e:
            print(f"实验失败: weight_method={weight_method}")
            print(f"错误信息: {str(e)}")
            continue
    
    # 保存比较结果
    comparison_file = f"results/qmamba_ablation/weight_comparison_{int(time.time())}.json"
    os.makedirs("results/qmamba_ablation", exist_ok=True)
    
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(
            convert_tensor_to_serializable(all_results), 
            f, indent=2, ensure_ascii=False
        )
    
    # 打印比较结果
    print("\n" + "="*50)
    print("权重计算方式比较结果")
    print("="*50)
    
    for result in all_results:
        print(f"\n{result['weight_method']}:")
        print(f"  参数数量: {result['total_params']}")
        print(f"  训练时间: {result['train_time']:.2f}秒")
        print(f"  测试时间: {result['test_time']:.2f}秒")
        print(f"  性能: {result['performance']}")
    
    print(f"\n比较结果已保存到: {comparison_file}")
    print("="*50)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='QMamba消融实验')
    parser.add_argument('--mode', type=str, default='comparison', 
                       choices=['comparison', 'ablation'],
                       help='实验模式: comparison(权重比较) 或 ablation(完整消融)')
    parser.add_argument('--weight_method', type=str, default=None,
                       choices=['attention', 'magnitude'],
                       help='指定权重计算方式（仅在comparison模式下有效）')
    parser.add_argument('--sequence_model', type=str, default='mamba',
                       choices=['mamba', 'transformer', 'cnn', 'rnn', 'rwkv'],
                       help='指定序列模型（仅在comparison模式下有效）')
    
    args = parser.parse_args()
    
    if args.mode == 'comparison':
        if args.weight_method:
            # 运行单个实验
            run_experiment(args.weight_method, args.sequence_model)
        else:
            # 运行权重计算方式比较
            run_single_comparison()
    elif args.mode == 'ablation':
        # 运行完整消融实验
        run_ablation_study()


if __name__ == "__main__":
    main() 