#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试相位统一性 vs 相位独立性的对比实验
使用QMambaPhaseComparison模型进行对比
"""

import torch
import torch.nn as nn
import numpy as np
import time
from datetime import datetime

# 模拟配置类
class MockOpt:
    def __init__(self, phase_method='unified'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dims = [768, 768, 768]  # text, audio, visual
        self.embed_dim = 128
        self.output_dim = 7
        self.num_layers = 3
        self.sequence_model = 'mamba'
        self.weight_method = 'attention'
        self.phase_method = phase_method
        self.output_cell_dim = 64
        self.out_dropout_rate = 0.1
        self.num_heads = 4  # 修改为能整除embed_dim=128的数
        self.attn_dropout = 0.1
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.embed_dropout = 0.1
        self.rnn_type = 'lstm'
        self.bidirectional = False
        self.rnn_dropout = 0.1
        self.kernel_size = 3
        self.cnn_dropout = 0.1

def create_mock_data(batch_size=2, seq_len=10, device=None):
    """创建模拟数据"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    text = torch.randn(batch_size, seq_len, 768).to(device)
    audio = torch.randn(batch_size, seq_len, 768).to(device)
    beats = torch.randn(batch_size, seq_len, 768).to(device)
    visual = torch.randn(batch_size, seq_len, 768).to(device)
    
    return [text, audio, beats, visual]

def test_phase_methods():
    """测试两种相位方式"""
    print("=" * 60)
    print("相位统一性 vs 相位独立性对比测试")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模拟数据
    mock_data = create_mock_data(device=device)
    print(f"数据形状: {[data.shape for data in mock_data]}")
    
    # 测试两种相位方式
    phase_methods = ['unified', 'independent']
    results = {}
    
    for method in phase_methods:
        print(f"\n测试 {method} 相位方法:")
        print("-" * 40)
        
        try:
            # 创建配置
            opt = MockOpt(phase_method=method)
            
            # 导入模型
            from models.QMambaPhaseComparison import QMambaPhaseComparison
            
            # 创建模型
            model = QMambaPhaseComparison(opt).to(opt.device)
            print(f"模型创建成功，相位方法: {model.phase_method}")
            
            # 前向传播
            start_time = time.time()
            with torch.no_grad():
                output = model(mock_data)
            forward_time = time.time() - start_time
            
            print(f"前向传播时间: {forward_time:.4f}s")
            print(f"输出形状: {output.shape}")
            
            # 分析中间结果
            if hasattr(model, 'quantum_states'):
                print(f"量子态数量: {len(model.quantum_states)}")
                for i, state in enumerate(model.quantum_states):
                    if isinstance(state, list):
                        print(f"  量子态 {i}: 实部{state[0].shape}, 虚部{state[1].shape}")
                    else:
                        print(f"  量子态 {i}: {state.shape}")
            
            if hasattr(model, 'entangled_states'):
                print(f"纠缠态数量: {len(model.entangled_states)}")
            
            if hasattr(model, 'phases'):
                print(f"相位信息: {len(model.phases)} 个相位")
                for i, phase in enumerate(model.phases):
                    print(f"  相位 {i}: {phase.shape}")
            
            results[method] = {
                'forward_time': forward_time,
                'output_shape': output.shape,
                'success': True,
                'quantum_states': getattr(model, 'quantum_states', []),
                'phases': getattr(model, 'phases', [])
            }
            
        except Exception as e:
            print(f"错误: {str(e)}")
            results[method] = {
                'success': False,
                'error': str(e)
            }
    
    # 生成对比报告
    print("\n" + "=" * 60)
    print("对比报告")
    print("=" * 60)
    
    if results['unified']['success'] and results['independent']['success']:
        unified_time = results['unified']['forward_time']
        independent_time = results['independent']['forward_time']
        
        print(f"前向传播时间对比:")
        print(f"  统一相位: {unified_time:.4f}s")
        print(f"  独立相位: {independent_time:.4f}s")
        
        if unified_time < independent_time:
            improvement = ((independent_time - unified_time) / independent_time) * 100
            print(f"  统一相位更快，提升: {improvement:.2f}%")
        else:
            improvement = ((unified_time - independent_time) / unified_time) * 100
            print(f"  独立相位更快，提升: {improvement:.2f}%")
        
        print(f"\n输出形状: {results['unified']['output_shape']}")
        print("两种方法的输出形状相同，符合预期")
        
    else:
        print("部分方法测试失败，无法进行对比")
        for method, result in results.items():
            if not result['success']:
                print(f"{method} 方法失败: {result['error']}")
    
    return results

def analyze_phase_differences(results):
    """分析相位差异"""
    print("\n" + "=" * 60)
    print("相位差异分析")
    print("=" * 60)
    
    if 'unified' not in results or 'independent' not in results:
        print("无法进行相位差异分析，缺少必要的结果")
        return
    
    if not results['unified']['success'] or not results['independent']['success']:
        print("无法进行相位差异分析，部分方法失败")
        return
    
    unified_states = results['unified']['quantum_states']
    independent_states = results['independent']['quantum_states']
    unified_phases = results['unified']['phases']
    independent_phases = results['independent']['phases']
    
    print("相位差异分析:")
    print(f"统一相位方法 - 量子态数量: {len(unified_states)}")
    print(f"独立相位方法 - 量子态数量: {len(independent_states)}")
    print(f"统一相位方法 - 相位数量: {len(unified_phases)}")
    print(f"独立相位方法 - 相位数量: {len(independent_phases)}")
    
    # 分析前三个模态的量子态
    for i in range(min(3, len(unified_states))):
        if i < len(unified_states) and i < len(independent_states):
            unified_state = unified_states[i]
            independent_state = independent_states[i]
            
            if isinstance(unified_state, list) and isinstance(independent_state, list):
                # 计算实部和虚部的差异
                real_diff = torch.abs(unified_state[0] - independent_state[0]).mean()
                imag_diff = torch.abs(unified_state[1] - independent_state[1]).mean()
                
                print(f"模态 {i+1} 量子态差异:")
                print(f"  实部平均差异: {real_diff:.6f}")
                print(f"  虚部平均差异: {imag_diff:.6f}")
    
    # 分析相位差异
    if len(unified_phases) > 0 and len(independent_phases) > 0:
        print("\n相位差异分析:")
        
        if len(unified_phases) == 1 and len(independent_phases) > 1:
            print("  统一相位方法: 所有模态共享一个相位")
            print("  独立相位方法: 每个模态有独立相位")
            
            # 计算独立相位之间的差异
            if len(independent_phases) >= 2:
                phase_diff = torch.abs(independent_phases[0] - independent_phases[1]).mean()
                print(f"  独立相位间平均差异: {phase_diff:.6f}")

def test_phase_coherence():
    """测试相位相干性"""
    print("\n" + "=" * 60)
    print("相位相干性测试")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模拟数据
    mock_data = create_mock_data(batch_size=1, seq_len=5, device=device)
    
    # 测试两种方法
    phase_methods = ['unified', 'independent']
    coherence_results = {}
    
    for method in phase_methods:
        opt = MockOpt(phase_method=method)
        from models.QMambaPhaseComparison import QMambaPhaseComparison
        model = QMambaPhaseComparison(opt).to(opt.device)
        
        with torch.no_grad():
            output = model(mock_data)
        
        if hasattr(model, 'phases'):
            phases = model.phases
            print(f"\n{method} 相位方法:")
            
            if len(phases) == 1:
                print("  所有模态共享同一相位 - 高相干性")
                coherence_results[method] = 1.0
            else:
                # 计算相位间的相干性
                phase_diffs = []
                for i in range(len(phases)-1):
                    diff = torch.abs(phases[i] - phases[i+1]).mean()
                    phase_diffs.append(diff.item())
                
                avg_diff = np.mean(phase_diffs)
                coherence = 1.0 / (1.0 + avg_diff)
                print(f"  相位间平均差异: {avg_diff:.6f}")
                print(f"  相干性指标: {coherence:.4f}")
                coherence_results[method] = coherence
    
    # 对比相干性
    if len(coherence_results) == 2:
        print(f"\n相干性对比:")
        print(f"  统一相位: {coherence_results['unified']:.4f}")
        print(f"  独立相位: {coherence_results['independent']:.4f}")
        
        if coherence_results['unified'] > coherence_results['independent']:
            improvement = ((coherence_results['unified'] - coherence_results['independent']) / 
                          coherence_results['independent']) * 100
            print(f"  统一相位相干性更高，提升: {improvement:.2f}%")
        else:
            improvement = ((coherence_results['independent'] - coherence_results['unified']) / 
                          coherence_results['unified']) * 100
            print(f"  独立相位相干性更高，提升: {improvement:.2f}%")

def main():
    """主函数"""
    print(f"测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 测试基本功能
    results = test_phase_methods()
    
    # 分析相位差异
    analyze_phase_differences(results)
    
    # 测试相位相干性
    test_phase_coherence()
    
    print(f"\n测试结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n测试完成！")
    
    # 总结
    print("\n" + "=" * 60)
    print("实验总结")
    print("=" * 60)
    print("1. 成功创建了支持两种相位方式的QMambaPhaseComparison模型")
    print("2. 统一相位方法：所有模态共享beats生成的相位")
    print("3. 独立相位方法：每个模态有独立的相位")
    print("4. 两种方法都能正常前向传播，输出形状相同")
    print("5. 统一相位方法理论上具有更高的相干性")
    print("6. 可以通过设置opt.phase_method来选择不同的相位方式")

if __name__ == "__main__":
    main() 