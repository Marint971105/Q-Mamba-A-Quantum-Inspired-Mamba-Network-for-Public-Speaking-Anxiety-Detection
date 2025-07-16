# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from types import SimpleNamespace
from dataset.features4quantum_reader import Features4QuantumReader
from models.QMambaPhaseComparison import QMambaPhaseComparison
from utils.model import train, test, save_model, save_performance
import random
import json
from datetime import datetime


def setup_params(phase_method='unified'):
    """设置参数"""
    opt = SimpleNamespace(
        # 数据相关
        pickle_dir_path='/home/tjc/audio/QMamba/feature_extract',
        features='textual,acoustic,beats,visual',
        dataset_name='features4quantum',  # CN数据集
        embedding_enabled=False,
        dialogue_context=False,

        # 模型架构
        network_type='qmamba_phase_comparison',
        sequence_model='mamba',
        # Mamba特有参数
        kernel_size=3,
        mamba_dropout=0.1,
        # 相位方法参数
        phase_method=phase_method,  # 'unified' 或 'independent'
        weight_method='attention',  # 'attention' 或 'magnitude'
        # 模型参数
        embed_dim=52,  # 修改为能被4整除的数
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
        save_dir='results/phase_comparison',
        output_file=None,
        config_file='config.ini'
    )
    return opt


def run_single_experiment(phase_method, opt_base):
    """运行单个相位方法的实验"""
    print(f"\n{'='*50}")
    print(f"开始 {phase_method} 相位方法实验")
    print(f"{'='*50}")
    
    # 创建实验特定的参数
    opt = setup_params(phase_method)
    opt.save_dir = f"{opt_base.save_dir}/{phase_method}"
    opt.output_file = f"{opt.save_dir}/results_{phase_method}.json"
    
    # 确保保存目录存在
    os.makedirs(opt.save_dir, exist_ok=True)
    
    print(f"相位方法: {opt.phase_method}")
    print(f"权重方法: {opt.weight_method}")
    print(f"保存目录: {opt.save_dir}")
    
    # 初始化数据读取器
    reader = Features4QuantumReader(opt)
    opt.reader = reader

    # 设置输入维度
    opt.input_dims = reader.input_dims
    opt.total_input_dim = sum(opt.input_dims[:3])

    # 初始化模型
    model = QMambaPhaseComparison(opt).to(opt.device)
    print(f"模型参数总量: {sum(p.numel() for p in model.parameters())}")

    # 训练模型
    print(f"\n开始训练 {phase_method} 方法...")
    train(opt, model)

    # 加载最佳模型
    best_model = QMambaPhaseComparison(opt).to(opt.device)
    best_model.load_state_dict(torch.load(opt.best_model_file))

    # 测试模型
    print(f"\n开始测试 {phase_method} 方法...")
    test_performance = test(best_model, opt)

    # 保存模型和性能结果
    save_model(best_model, opt, test_performance)
    save_performance(opt, test_performance)

    # 删除临时文件
    if os.path.exists(opt.best_model_file):
        os.remove(opt.best_model_file)
    
    return test_performance


def compare_performance(results):
    """比较两种方法的性能"""
    print(f"\n{'='*60}")
    print("相位方法性能对比")
    print(f"{'='*60}")
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for metric in metrics:
        print(f"\n{metric.upper()} 对比:")
        unified_val = results['unified'].get(metric, 0)
        independent_val = results['independent'].get(metric, 0)
        
        print(f"  统一相位: {unified_val:.4f}")
        print(f"  独立相位: {independent_val:.4f}")
        
        if unified_val > independent_val:
            improvement = ((unified_val - independent_val) / independent_val) * 100
            print(f"  统一相位优于独立相位: +{improvement:.2f}%")
        elif independent_val > unified_val:
            improvement = ((independent_val - unified_val) / unified_val) * 100
            print(f"  独立相位优于统一相位: +{improvement:.2f}%")
        else:
            print("  两种方法性能相同")


def save_comparison_results(results, opt_base):
    """保存对比结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = f"{opt_base.save_dir}/comparison_results_{timestamp}.json"
    
    comparison_data = {
        'timestamp': timestamp,
        'experiment_config': {
            'dataset': opt_base.dataset_name,
            'sequence_model': opt_base.sequence_model,
            'embed_dim': opt_base.embed_dim,
            'num_layers': opt_base.num_layers,
            'batch_size': opt_base.batch_size,
            'epochs': opt_base.epochs,
            'lr': opt_base.lr
        },
        'results': results,
        'summary': {
            'unified_better_metrics': [],
            'independent_better_metrics': [],
            'equal_metrics': []
        }
    }
    
    # 分析哪些指标哪种方法更好
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for metric in metrics:
        unified_val = results['unified'].get(metric, 0)
        independent_val = results['independent'].get(metric, 0)
        
        if unified_val > independent_val:
            comparison_data['summary']['unified_better_metrics'].append(metric)
        elif independent_val > unified_val:
            comparison_data['summary']['independent_better_metrics'].append(metric)
        else:
            comparison_data['summary']['equal_metrics'].append(metric)
    
    # 保存结果
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n对比结果已保存到: {comparison_file}")
    
    # 打印总结
    print(f"\n{'='*60}")
    print("实验总结")
    print(f"{'='*60}")
    print(f"统一相位更优的指标: {comparison_data['summary']['unified_better_metrics']}")
    print(f"独立相位更优的指标: {comparison_data['summary']['independent_better_metrics']}")
    print(f"性能相同的指标: {comparison_data['summary']['equal_metrics']}")


def main():
    """主函数"""
    print("相位方法对比实验")
    print("="*60)
    print("本实验将比较统一相位和独立相位两种方法的性能")
    print("="*60)
    
    # 基础参数设置
    opt_base = setup_params()
    
    # 确保主保存目录存在
    os.makedirs(opt_base.save_dir, exist_ok=True)
    
    # 运行两种方法的实验
    results = {}
    
    # 运行统一相位实验
    try:
        results['unified'] = run_single_experiment('unified', opt_base)
        print(f"\n统一相位实验完成")
    except Exception as e:
        print(f"统一相位实验失败: {e}")
        results['unified'] = {}
    
    # 运行独立相位实验
    try:
        results['independent'] = run_single_experiment('independent', opt_base)
        print(f"\n独立相位实验完成")
    except Exception as e:
        print(f"独立相位实验失败: {e}")
        results['independent'] = {}
    
    # 比较性能
    if results['unified'] and results['independent']:
        compare_performance(results)
        save_comparison_results(results, opt_base)
    else:
        print("\n部分实验失败，无法进行完整对比")
    
    print(f"\n{'='*60}")
    print("实验完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main() 