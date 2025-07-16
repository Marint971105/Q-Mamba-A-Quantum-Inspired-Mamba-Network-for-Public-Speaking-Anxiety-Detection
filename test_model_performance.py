import torch
from models.QMamba import QMamba
from utils.model_analysis import ModelAnalyzer
from types import SimpleNamespace
import thop
from thop import clever_format
import time
import numpy as np

def test_model_performance():
    # 配置参数
    opt = SimpleNamespace(
        # 数据相关
        pickle_dir_path = '/home/tjc/audio/QMamba/feature_extract',
        features = 'textual,acoustic,beats,visual',
        dataset_name = 'features4quantum',
        embedding_enabled = False,
        dialogue_context = False,
        
        # 模型相关
        network_type = 'qmamba',
        embed_dim = 50,
        speaker_num = 1,
        output_dim = 4,
        output_cell_dim = 50,
        out_dropout_rate = 0.2,
        num_layers = 2,
        
        # 训练相关
        gpu = 0,
        batch_size = 64,
        epochs = 50,
        lr = 0.001,
        unitary_lr = 0.001,
        clip = 0.8,
        patience = 10,
        min_lr = 1e-5,
        
        # 设备
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        
        # 保存相关
        save_dir = 'results/qmamba/performance_analysis',
        dir_name = None,
        output_file = None,
        config_file = 'config.ini',
        
        # 输入维度
        input_dims = [768, 768, 768, 768],
        total_input_dim = 768 * 4,
        
        # 类别平衡相关
        class_balance_strategy = 'weighted_loss',
        
        # 序列模型相关
        sequence_model = 'cnn',  # 用于初始化
        
        # Transformer特有参数
        num_heads = 5,
        attn_dropout = 0.1,
        relu_dropout = 0.1,
        res_dropout = 0.1,
        embed_dropout = 0.1,
        
        # RNN特有参数
        rnn_type = 'rnn',  # 'lstm', 'gru', 或 'rnn'
        bidirectional = False,
        rnn_dropout = 0.1,
        
        # CNN特有参数
        kernel_size = 128,
        cnn_dropout = 0.1,
    )
    
    # 创建测试数据
    batch_size = 64
    seq_len = 1
    text = torch.randn(batch_size, seq_len, 768).to(opt.device)
    audio = torch.randn(batch_size, seq_len, 768).to(opt.device)
    beats = torch.randn(batch_size, seq_len, 768).to(opt.device)
    visual = torch.randn(batch_size, seq_len, 768).to(opt.device)
    
    # 将输入数据作为列表传递
    test_inputs = [text, audio, beats, visual]
    
    # 创建不同的模型
    models = {}
    for model_type in ['mamba', 'transformer', 'rwkv', 'rnn', 'cnn']:
        opt.sequence_model = model_type
        
        # 为每个模型类型设置特定参数
        if model_type == 'cnn':
            opt.kernel_size = 3
            opt.cnn_dropout = 0.1
            opt.num_layers = 1  # CNN的层数
        elif model_type == 'transformer':
            opt.num_heads = 5
            opt.attn_dropout = 0.1
            opt.num_layers = 1
        elif model_type == 'rnn':
            opt.rnn_type = 'rnn'
            opt.bidirectional = False
            opt.rnn_dropout = 0.1
            opt.num_layers = 1
        
        models[model_type.capitalize()] = QMamba(opt).to(opt.device)
    
    # 创建分析器
    analyzer = ModelAnalyzer()
    
    # 比较模型性能
    print("\n=== 模型性能比较 ===")
    performance_metrics = {
        'params': [],
        'macs': [],
        'flops': [],
        'inference_time': [],  # 现在存储 (name, avg_time, std_time)
        'memory': []
    }
    
    for name, model in models.items():
        print(f"\n{name} 模型:")
        params_count = sum(p.numel() for p in model.parameters())
        print(f"参数总量: {params_count:,}")
        
        # 计算 FLOPs 和 MACs
        macs, params = thop.profile(model, inputs=(test_inputs,))
        flops = macs * 2  # FLOPs = MACs * 2
        macs, flops, params = clever_format([macs, flops, params], "%.3f")
        
        # 测量推理时间
        times = []
        model.eval()
        with torch.no_grad():
            # 预热
            for _ in range(10):
                _ = model(test_inputs)
            
            # 正式测量
            for _ in range(100):
                start_time = time.time()
                _ = model(test_inputs)
                torch.cuda.synchronize()
                times.append(time.time() - start_time)
        avg_time = np.mean(times) * 1000  # 转换为毫秒
        std_time = np.std(times) * 1000   # 标准差，转换为毫秒
        
        # 测量内存使用
        torch.cuda.reset_peak_memory_stats()
        _ = model(test_inputs)
        memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        
        # 存储指标
        performance_metrics['params'].append((name, params_count))
        performance_metrics['macs'].append((name, macs))
        performance_metrics['flops'].append((name, flops))
        performance_metrics['inference_time'].append((name, avg_time, std_time))
        performance_metrics['memory'].append((name, memory))
        
        print(f"MACs: {macs}")
        print(f"FLOPs: {flops}")
        print(f"参数量: {params}")
        print(f"推理时间: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"显存使用: {memory:.2f} MB")
    
    # 打印比较结果
    print("\n=== 详细性能比较 ===")
    
    # 参数量比较
    print("\n参数量比较:")
    sorted_params = sorted(performance_metrics['params'], key=lambda x: x[1])
    for name, value in sorted_params:
        print(f"{name}: {value:,} ({value/sorted_params[-1][1]*100:.1f}%)")
    
    # MACs比较
    print("\nMACs比较:")
    sorted_macs = sorted(performance_metrics['macs'], key=lambda x: float(x[1].replace('M','').replace('G','000')))
    for name, value in sorted_macs:
        print(f"{name}: {value} ({float(value.replace('M','').replace('G','000'))/float(sorted_macs[-1][1].replace('M','').replace('G','000'))*100:.1f}%)")
    
    # FLOPs比较
    print("\nFLOPs比较:")
    sorted_flops = sorted(performance_metrics['flops'], key=lambda x: float(x[1].replace('M','').replace('G','000')))
    for name, value in sorted_flops:
        print(f"{name}: {value} ({float(value.replace('M','').replace('G','000'))/float(sorted_flops[-1][1].replace('M','').replace('G','000'))*100:.1f}%)")
    
    # 推理时间比较
    print("\n推理时间比较:")
    sorted_time = sorted(performance_metrics['inference_time'], key=lambda x: x[1])  # 按平均时间排序
    for name, avg_time, std_time in sorted_time:
        print(f"{name}: {avg_time:.2f} ± {std_time:.2f} ms ({avg_time/sorted_time[-1][1]*100:.1f}%)")
    
    # 内存使用比较
    print("\n显存使用比较:")
    sorted_memory = sorted(performance_metrics['memory'], key=lambda x: x[1])
    for name, value in sorted_memory:
        print(f"{name}: {value:.2f} MB ({value/sorted_memory[-1][1]*100:.1f}%)")
    
    print("="*30)

if __name__ == "__main__":
    test_model_performance() 