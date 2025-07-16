# -*- coding: utf-8 -*-
import os
import torch
from types import SimpleNamespace
from dataset.features4quantum_fudan_reader import Features4QuantumFudanReader
from models.QMambaPhaseComparison import QMambaPhaseComparison
from utils.model import train, test, save_model, save_performance
import time
import json
from datetime import datetime


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


def setup_params(phase_method='unified', sequence_model='mamba'):
    """设置Fudan数据集的参数
    
    Args:
        phase_method: 相位方法 ('unified' 或 'independent')
        sequence_model: 序列模型类型
    """
    opt = SimpleNamespace(
        # 数据相关
        pickle_dir_path='/home/tjc/audio/QMamba/feature_extract',
        features='textual,acoustic,beats,visual',
        dataset_name='features4quantum_fudan',  # Fudan数据集
        embedding_enabled=False,
        dialogue_context=False,

        # 模型架构
        network_type='qmamba_phase_comparison',
        sequence_model=sequence_model,
        phase_method=phase_method,  # 相位方法
        weight_method='attention',  # 权重计算方式
        num_layers=1,  # 层数
        
        # RNN特有参数
        rnn_type='rnn',
        bidirectional=False,
        rnn_dropout=0.1,
        
        # CNN特有参数
        kernel_size=3,      # CNN的卷积核大小
        cnn_dropout=0.1,    # CNN的dropout率
        
        # 模型参数
        embed_dim=52,  # 修改为能被4整除的数
        output_cell_dim=52,
        output_dim=4,
        speaker_num=1,
        
        # Transformer特定参数
        num_heads=4,  # 注意力头数
        dropout=0.1,  # Transformer中的dropout率
        attn_dropout=0.1,  # 注意力的dropout
        relu_dropout=0.1,  # 前馈网络中relu后的dropout
        res_dropout=0.1,  # 残差连接的dropout
        embed_dropout=0.1,  # 嵌入层的dropout
        attn_mask=False,  # 是否使用注意力掩码
        dim_feedforward=52,  # 前馈网络维度
        normalize_before=True,  # 是否在attention和FFN之前进行norm
        
        # 数据维度
        input_dims=[768, 768, 768, 768],  # 各个模态的输入维度
        
        # 添加类别标签字典
        emotion_dic=['0', '1', '2', '3'],
        
        # 训练参数
        epochs=50,
        patience=10,
        batch_size=64,
        lr=0.0001,
        unitary_lr=0.0001,
        out_dropout_rate=0.2,
        clip=1.0,
        min_lr=1e-5,
        
        # 类别权重
        loss_weights=torch.FloatTensor([1.0, 1.0, 1.0, 1.0]),
        
        # 其他设置
        gpu=0,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        save_dir='results/phase_comparison_fudan',
        output_file=None,
        config_file='config.ini'
    )
    
    # 将loss_weights移到指定设备
    if hasattr(opt, 'loss_weights'):
        opt.loss_weights = opt.loss_weights.to(opt.device)
    
    return opt


def run_single_experiment(phase_method, sequence_model='mamba'):
    """运行单个相位方法的实验
    
    Args:
        phase_method: 相位方法 ('unified' 或 'independent')
        sequence_model: 序列模型类型
    
    Returns:
        dict: 实验结果
    """
    print(f"\n{'='*50}")
    print(f"开始Fudan相位实验: phase_method={phase_method}, sequence_model={sequence_model}")
    print(f"{'='*50}")
    
    # 获取参数
    opt = setup_params(phase_method, sequence_model)
    
    # 设置输出文件名
    timestamp = int(time.time())
    opt.output_file = f"{opt.save_dir}/results_{phase_method}_{sequence_model}_{timestamp}.json"
    opt.best_model_file = f"{opt.save_dir}/best_model_{phase_method}_{sequence_model}_{timestamp}.pth"
    
    # 创建保存目录
    os.makedirs(opt.save_dir, exist_ok=True)
    
    print(f"实验配置:")
    print(f"  数据集: {opt.dataset_name}")
    print(f"  相位方法: {opt.phase_method}")
    print(f"  权重计算方式: {opt.weight_method}")
    print(f"  序列模型: {opt.sequence_model}")
    print(f"  设备: {opt.device}")
    print(f"  批大小: {opt.batch_size}")
    print(f"  学习率: {opt.lr}")
    print(f"  嵌入维度: {opt.embed_dim}")
    print(f"  输出文件: {opt.output_file}")
    
    # 初始化数据读取器
    reader = Features4QuantumFudanReader(opt)
    opt.reader = reader

    # 设置输入维度
    opt.input_dims = reader.input_dims
    opt.total_input_dim = sum(opt.input_dims[:3])

    # 初始化模型
    model = QMambaPhaseComparison(opt).to(opt.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总量: {total_params}")

    # 训练模型
    print("\n开始训练...")
    train_start_time = time.time()
    train(opt, model)
    train_time = time.time() - train_start_time

    # 加载最佳模型
    best_model = QMambaPhaseComparison(opt).to(opt.device)
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
        'dataset': 'fudan',
        'phase_method': phase_method,
        'weight_method': opt.weight_method,
        'sequence_model': sequence_model,
        'total_params': total_params,
        'train_time': train_time,
        'test_time': test_time,
        'performance': convert_tensor_to_serializable(test_performance),
        'config': {
            'embed_dim': opt.embed_dim,
            'num_layers': opt.num_layers,
            'batch_size': opt.batch_size,
            'lr': float(opt.lr),
            'epochs': opt.epochs,
            'dataset_name': opt.dataset_name
        }
    }
    
    print(f"\n实验完成:")
    print(f"  训练时间: {train_time:.2f}秒")
    print(f"  测试时间: {test_time:.2f}秒")
    print(f"  测试性能: {test_performance}")
    print(f"  结果已保存到: {opt.output_file}")
    
    return result


def compare_phase_methods(sequence_model='mamba'):
    """比较两种相位方法的性能"""
    print("开始Fudan数据集相位方法对比实验")
    print("="*60)
    
    # 比较两种相位方法
    phase_methods = ['unified', 'independent']
    all_results = []
    
    for phase_method in phase_methods:
        try:
            result = run_single_experiment(phase_method, sequence_model)
            all_results.append(result)
        except Exception as e:
            print(f"实验失败: phase_method={phase_method}")
            print(f"错误信息: {str(e)}")
            continue
    
    # 保存对比结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = f"results/phase_comparison_fudan/phase_comparison_{sequence_model}_{timestamp}.json"
    os.makedirs("results/phase_comparison_fudan", exist_ok=True)
    
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(
            convert_tensor_to_serializable(all_results), 
            f, indent=2, ensure_ascii=False
        )
    
    # 打印对比结果
    print("\n" + "="*60)
    print("Fudan数据集相位方法对比结果")
    print("="*60)
    
    if len(all_results) == 2:
        unified_result = next((r for r in all_results if r['phase_method'] == 'unified'), None)
        independent_result = next((r for r in all_results if r['phase_method'] == 'independent'), None)
        
        if unified_result and independent_result:
            print(f"\n统一相位方法:")
            print(f"  参数数量: {unified_result['total_params']}")
            print(f"  训练时间: {unified_result['train_time']:.2f}秒")
            print(f"  测试时间: {unified_result['test_time']:.2f}秒")
            print(f"  性能: {unified_result['performance']}")
            
            print(f"\n独立相位方法:")
            print(f"  参数数量: {independent_result['total_params']}")
            print(f"  训练时间: {independent_result['train_time']:.2f}秒")
            print(f"  测试时间: {independent_result['test_time']:.2f}秒")
            print(f"  性能: {independent_result['performance']}")
            
            # 性能对比
            print(f"\n性能对比:")
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            for metric in metrics:
                unified_val = unified_result['performance'].get(metric, 0)
                independent_val = independent_result['performance'].get(metric, 0)
                
                print(f"  {metric.upper()}:")
                print(f"    统一相位: {unified_val:.4f}")
                print(f"    独立相位: {independent_val:.4f}")
                
                if unified_val > independent_val:
                    improvement = ((unified_val - independent_val) / independent_val) * 100
                    print(f"    统一相位优于独立相位: +{improvement:.2f}%")
                elif independent_val > unified_val:
                    improvement = ((independent_val - unified_val) / unified_val) * 100
                    print(f"    独立相位优于统一相位: +{improvement:.2f}%")
                else:
                    print("    两种方法性能相同")
    
    print(f"\n对比结果已保存到: {comparison_file}")
    print("="*60)
    
    return all_results


def run_ablation_study():
    """运行完整消融实验（不同序列模型）"""
    print("开始Fudan数据集相位方法消融实验")
    print("="*60)
    
    # 实验配置
    phase_methods = ['unified', 'independent']
    sequence_models = ['mamba', 'transformer', 'cnn']
    
    # 存储所有实验结果
    all_results = []
    
    # 运行所有实验组合
    for phase_method in phase_methods:
        for sequence_model in sequence_models:
            try:
                result = run_single_experiment(phase_method, sequence_model)
                all_results.append(result)
            except Exception as e:
                print(f"实验失败: phase_method={phase_method}, sequence_model={sequence_model}")
                print(f"错误信息: {str(e)}")
                continue
    
    # 保存汇总结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"results/phase_comparison_fudan/ablation_summary_{timestamp}.json"
    os.makedirs("results/phase_comparison_fudan", exist_ok=True)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(
            convert_tensor_to_serializable(all_results), 
            f, indent=2, ensure_ascii=False
        )
    
    # 打印汇总结果
    print("\n" + "="*60)
    print("Fudan数据集相位方法消融实验汇总结果")
    print("="*60)
    
    for result in all_results:
        print(f"\n{result['phase_method']} + {result['sequence_model']}:")
        print(f"  参数数量: {result['total_params']}")
        print(f"  训练时间: {result['train_time']:.2f}秒")
        print(f"  测试时间: {result['test_time']:.2f}秒")
        print(f"  性能: {result['performance']}")
    
    print(f"\n汇总结果已保存到: {summary_file}")
    print("="*60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fudan数据集相位方法对比实验')
    parser.add_argument('--mode', type=str, default='comparison', 
                       choices=['comparison', 'ablation', 'single'],
                       help='实验模式: comparison(相位对比) 或 ablation(完整消融) 或 single(单个实验)')
    parser.add_argument('--phase_method', type=str, default=None,
                       choices=['unified', 'independent'],
                       help='指定相位方法（仅在single模式下有效）')
    parser.add_argument('--sequence_model', type=str, default='mamba',
                       choices=['mamba', 'transformer', 'cnn', 'rnn', 'rwkv'],
                       help='指定序列模型')
    
    args = parser.parse_args()
    
    if args.mode == 'comparison':
        # 运行相位方法对比
        compare_phase_methods(args.sequence_model)
    elif args.mode == 'ablation':
        # 运行完整消融实验
        run_ablation_study()
    elif args.mode == 'single':
        if args.phase_method:
            # 运行单个实验
            run_single_experiment(args.phase_method, args.sequence_model)
        else:
            print("single模式需要指定--phase_method参数")
            return


if __name__ == "__main__":
    main() 