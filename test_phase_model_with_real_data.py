# -*- coding: utf-8 -*-
import torch
import numpy as np
from types import SimpleNamespace
from dataset.features4quantum_reader import Features4QuantumReader
from models.QMambaPhaseComparison import QMambaPhaseComparison
import time


def setup_test_params(phase_method='unified'):
    """设置测试参数"""
    opt = SimpleNamespace(
        # 数据相关
        pickle_dir_path='/home/tjc/audio/QMamba/feature_extract',
        features='textual,acoustic,beats,visual',
        dataset_name='features4quantum',
        embedding_enabled=False,
        dialogue_context=False,

        # 模型架构
        network_type='qmamba_phase_comparison',
        sequence_model='cnn',
        kernel_size=3,
        cnn_dropout=0.1,
        phase_method=phase_method,
        weight_method='attention',
        
        # 模型参数
        embed_dim=52,  # 修改为能被4整除的数
        num_layers=1,
        output_cell_dim=50,
        output_dim=4,
        speaker_num=1,

        # 训练参数
        batch_size=2,  # 添加batch_size参数
        epochs=1,
        patience=5,
        lr=0.0005,
        unitary_lr=0.001,
        out_dropout_rate=0.2,
        clip=1.0,
        min_lr=1e-5,

        # 设备设置
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    )
    return opt


def test_phase_method(phase_method):
    """测试单个相位方法"""
    print(f"\n{'='*50}")
    print(f"测试 {phase_method} 相位方法")
    print(f"{'='*50}")
    
    # 设置参数
    opt = setup_test_params(phase_method)
    
    try:
        # 初始化数据读取器
        print("初始化数据读取器...")
        reader = Features4QuantumReader(opt)
        opt.reader = reader
        opt.input_dims = reader.input_dims
        opt.total_input_dim = sum(opt.input_dims[:3])
        
        print(f"输入维度: {opt.input_dims}")
        print(f"总输入维度: {opt.total_input_dim}")
        
        # 获取一个批次的数据
        print("获取测试数据...")
        test_loader = reader.get_data(iterable=True, shuffle=False, split='test')
        test_data = next(iter(test_loader))
        print(f"测试数据形状: {[x.shape for x in test_data]}")
        
        # 将数据移动到设备上
        print("将数据移动到设备上...")
        test_data = [x.to(opt.device) for x in test_data]
        
        # 初始化模型
        print("初始化模型...")
        model = QMambaPhaseComparison(opt).to(opt.device)
        print(f"模型参数总量: {sum(p.numel() for p in model.parameters())}")
        
        # 前向传播测试
        print("执行前向传播...")
        start_time = time.time()
        
        with torch.no_grad():
            output = model(test_data)
        
        end_time = time.time()
        
        print(f"前向传播成功!")
        print(f"输出形状: {output.shape}")
        print(f"前向传播时间: {end_time - start_time:.4f}s")
        
        # 分析中间状态
        if hasattr(model, 'quantum_states'):
            print(f"量子态数量: {len(model.quantum_states)}")
            for i, state in enumerate(model.quantum_states):
                if isinstance(state, list) and len(state) == 2:
                    print(f"  量子态 {i}: 实部{state[0].shape}, 虚部{state[1].shape}")
        
        if hasattr(model, 'phases'):
            print(f"相位信息: {len(model.phases)} 个相位")
            for i, phase in enumerate(model.phases):
                print(f"  相位 {i}: {phase.shape}")
        
        return True, output.shape
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    """主测试函数"""
    print("相位方法真实数据测试")
    print("="*60)
    
    results = {}
    
    # 测试统一相位方法
    success, output_shape = test_phase_method('unified')
    results['unified'] = {'success': success, 'output_shape': output_shape}
    
    # 测试独立相位方法
    success, output_shape = test_phase_method('independent')
    results['independent'] = {'success': success, 'output_shape': output_shape}
    
    # 输出测试结果
    print(f"\n{'='*60}")
    print("测试结果总结")
    print(f"{'='*60}")
    
    for method, result in results.items():
        status = "成功" if result['success'] else "失败"
        shape_info = f", 输出形状: {result['output_shape']}" if result['output_shape'] else ""
        print(f"{method} 相位方法: {status}{shape_info}")
    
    # 检查是否都成功
    all_success = all(result['success'] for result in results.values())
    if all_success:
        print(f"\n✅ 所有相位方法测试成功！可以开始正式训练实验。")
    else:
        print(f"\n❌ 部分相位方法测试失败，需要修复问题后再进行训练实验。")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main() 