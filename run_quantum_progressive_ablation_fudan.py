# -*- coding: utf-8 -*-
import os
import torch
import json
import time
from types import SimpleNamespace
from dataset.features4quantum_fudan_reader import Features4QuantumFudanReader
from models.BaselineMamba import BaselineMamba
from models.QuantumSuperposition import QuantumSuperposition
from models.QuantumEntanglement import QuantumEntanglement
from models.QMamba import QMamba
from utils.model import train, test, save_model, save_performance


def setup_params():
    """设置Fudan数据集的参数"""
    opt = SimpleNamespace(
        # 数据相关
        pickle_dir_path='/home/tjc/audio/QMamba/feature_extract',
        features='textual,acoustic,beats,visual',
        dataset_name='features4quantum_fudan',  # Fudan数据集
        embedding_enabled=False,
        dialogue_context=False,

        # 模型架构
        network_type='qmamba',
        sequence_model='mamba',  # 只使用mamba
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
        lr=0.0001,
        unitary_lr=0.0001,
        out_dropout_rate=0.2,
        clip=1.0,
        min_lr=1e-5,

        # 添加类别标签字典
        emotion_dic=['0', '1', '2', '3'],
        
        # 其他设置
        gpu=0,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        save_dir='results/quantum_progressive_fudan',
        output_file=None,
        config_file='config.ini'
    )
    return opt


def convert_tensor_to_list(obj):
    """将Tensor转换为可序列化的格式"""
    if isinstance(obj, torch.Tensor):
        return obj.cpu().detach().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensor_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensor_to_list(item) for item in obj]
    else:
        return obj


def run_single_experiment(model_class, model_name, opt):
    """运行单个实验"""
    print(f"\n=== 运行 {model_name} 实验 ===")
    
    # 初始化数据读取器
    reader = Features4QuantumFudanReader(opt)
    opt.reader = reader
    
    # 设置输入维度
    opt.input_dims = reader.input_dims
    opt.total_input_dim = sum(opt.input_dims[:3])
    
    # 初始化模型
    model = model_class(opt).to(opt.device)
    print(f"模型参数总量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练模型
    print(f"开始训练 {model_name}...")
    train(opt, model)
    
    # 加载最佳模型
    best_model = model_class(opt).to(opt.device)
    best_model.load_state_dict(torch.load(opt.best_model_file))
    
    # 测试模型
    print(f"开始测试 {model_name}...")
    test_performance = test(best_model, opt)
    
    # 保存模型和性能结果
    save_model(best_model, opt, test_performance)
    save_performance(opt, test_performance)
    
    # 删除临时文件
    if os.path.exists(opt.best_model_file):
        os.remove(opt.best_model_file)
    
    print(f"=== {model_name} 实验完成 ===")
    return test_performance


def run_comparison_experiments():
    """运行比较实验"""
    print("\n=== 开始Fudan数据集渐进式量子组件消融实验 ===")
    
    # 设置参数
    opt = setup_params()
    
    # 定义实验配置
    experiments = [
        (BaselineMamba, "BaselineMamba", "基线模型：纯Mamba + 传统L2Norm融合"),
        (QuantumSuperposition, "QuantumSuperposition", "量子叠加：基线 + 量子叠加状态建模"),
        (QuantumEntanglement, "QuantumEntanglement", "量子纠缠：量子叠加 + 量子纠缠"),
        (QMamba, "QMamba", "完整Q-Mamba：量子纠缠 + 多头注意力融合")
    ]
    
    results = {}
    
    # 运行每个实验
    for model_class, model_name, description in experiments:
        print(f"\n{description}")
        try:
            performance = run_single_experiment(model_class, model_name, opt)
            results[model_name] = {
                'description': description,
                'performance': convert_tensor_to_list(performance)
            }
            print(f"✓ {model_name} 实验成功")
        except Exception as e:
            print(f"✗ {model_name} 实验失败: {str(e)}")
            results[model_name] = {
                'description': description,
                'error': str(e)
            }
    
    # 保存比较结果
    comparison_file = os.path.join(opt.save_dir, 'fudan_quantum_progressive_comparison.json')
    os.makedirs(opt.save_dir, exist_ok=True)
    
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n比较结果已保存到: {comparison_file}")
    
    # 计算增量贡献
    calculate_incremental_contributions(results)
    
    return results


def run_ablation_experiments():
    """运行消融实验"""
    print("\n=== 开始Fudan数据集量子组件消融实验 ===")
    
    # 设置参数
    opt = setup_params()
    
    # 定义消融配置
    ablation_configs = [
        {
            'name': 'baseline',
            'model': BaselineMamba,
            'description': '基线：无量子机制'
        },
        {
            'name': 'superposition',
            'model': QuantumSuperposition,
            'description': '+量子叠加'
        },
        {
            'name': 'entanglement',
            'model': QuantumEntanglement,
            'description': '+量子纠缠'
        },
        {
            'name': 'full_qmamba',
            'model': QMamba,
            'description': '+多头注意力融合'
        }
    ]
    
    ablation_results = {}
    
    # 运行消融实验
    for config in ablation_configs:
        print(f"\n消融实验: {config['description']}")
        try:
            performance = run_single_experiment(
                config['model'], 
                config['name'], 
                opt
            )
            ablation_results[config['name']] = {
                'description': config['description'],
                'performance': convert_tensor_to_list(performance)
            }
            print(f"✓ {config['name']} 消融实验成功")
        except Exception as e:
            print(f"✗ {config['name']} 消融实验失败: {str(e)}")
            ablation_results[config['name']] = {
                'description': config['description'],
                'error': str(e)
            }
    
    # 保存消融结果
    ablation_file = os.path.join(opt.save_dir, 'fudan_quantum_ablation_results.json')
    os.makedirs(opt.save_dir, exist_ok=True)
    
    with open(ablation_file, 'w', encoding='utf-8') as f:
        json.dump(ablation_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n消融结果已保存到: {ablation_file}")
    
    return ablation_results


def calculate_incremental_contributions(results):
    """计算量子组件的增量贡献"""
    print("\n=== Fudan数据集量子组件增量贡献分析 ===")
    
    # 提取性能指标
    performance_metrics = {}
    for model_name, result in results.items():
        if 'performance' in result and 'error' not in result:
            perf = result['performance']
            if isinstance(perf, dict) and 'accuracy' in perf:
                performance_metrics[model_name] = perf['accuracy']
    
    # 直接计算贡献，不需要比较
    models = ['BaselineMamba', 'QuantumSuperposition', 'QuantumEntanglement', 'QMamba']
    contributions = {}
    
    # 计算每个模型的绝对贡献
    for model_name in models:
        if model_name in performance_metrics:
            acc = performance_metrics[model_name]
            contributions[model_name] = {
                'accuracy': acc,
                'contribution': acc,  # 直接使用精度作为贡献
                'contribution_percentage': acc * 100  # 转换为百分比
            }
    
    # 计算相对增量贡献（如果有多个模型）
    if len(performance_metrics) >= 2:
        incremental_contributions = {}
        for i in range(1, len(models)):
            current_model = models[i]
            previous_model = models[i-1]
            
            if current_model in performance_metrics and previous_model in performance_metrics:
                current_acc = performance_metrics[current_model]
                previous_acc = performance_metrics[previous_model]
                increment = current_acc - previous_acc
                
                incremental_contributions[current_model] = {
                    'previous_model': previous_model,
                    'current_accuracy': current_acc,
                    'previous_accuracy': previous_acc,
                    'increment': increment,
                    'increment_percentage': (increment / previous_acc * 100) if previous_acc > 0 else 0
                }
        
        # 打印增量贡献
        print("\n量子组件增量贡献:")
        print("-" * 80)
        print(f"{'模型':<20} {'前一个模型':<20} {'当前精度':<10} {'增量':<10} {'增量%':<10}")
        print("-" * 80)
        
        for model_name, contrib in incremental_contributions.items():
            print(f"{model_name:<20} {contrib['previous_model']:<20} "
                  f"{contrib['current_accuracy']:<10.4f} "
                  f"{contrib['increment']:<10.4f} "
                  f"{contrib['increment_percentage']:<10.2f}%")
    
    # 打印绝对贡献
    print("\n量子组件绝对贡献:")
    print("-" * 60)
    print(f"{'模型':<20} {'精度':<10} {'贡献':<10} {'贡献%':<10}")
    print("-" * 60)
    
    for model_name, contrib in contributions.items():
        print(f"{model_name:<20} "
              f"{contrib['accuracy']:<10.4f} "
              f"{contrib['contribution']:<10.4f} "
              f"{contrib['contribution_percentage']:<10.2f}%")
    
    # 保存贡献结果
    opt = setup_params()
    contrib_file = os.path.join(opt.save_dir, 'fudan_quantum_contributions.json')
    
    all_contributions = {
        'absolute_contributions': contributions,
        'incremental_contributions': incremental_contributions if len(performance_metrics) >= 2 else {}
    }
    
    with open(contrib_file, 'w', encoding='utf-8') as f:
        json.dump(all_contributions, f, ensure_ascii=False, indent=2)
    
    print(f"\n贡献分析已保存到: {contrib_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fudan数据集量子渐进式消融实验')
    parser.add_argument('--mode', type=str, default='comparison', 
                       choices=['comparison', 'ablation', 'single'],
                       help='实验模式')
    parser.add_argument('--model', type=str, 
                       choices=['baseline', 'superposition', 'entanglement', 'qmamba'],
                       help='单个实验的模型类型')
    
    args = parser.parse_args()
    
    if args.mode == 'comparison':
        run_comparison_experiments()
    elif args.mode == 'ablation':
        run_ablation_experiments()
    elif args.mode == 'single':
        if not args.model:
            print("单个实验模式需要指定 --model 参数")
            return
        
        opt = setup_params()
        model_map = {
            'baseline': BaselineMamba,
            'superposition': QuantumSuperposition,
            'entanglement': QuantumEntanglement,
            'qmamba': QMamba
        }
        
        model_class = model_map[args.model]
        run_single_experiment(model_class, args.model, opt)
    
    print("\n=== 所有实验完成 ===")


if __name__ == "__main__":
    main() 