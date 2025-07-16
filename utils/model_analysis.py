import torch
import time
import numpy as np
from thop import profile as thop_profile, clever_format
from torch.profiler import profile as torch_profile, record_function, ProfilerActivity

class ModelAnalyzer:
    def __init__(self):
        self.results = {}
        
    def compare_inference_speed(self, models, inputs, num_runs=100):
        """比较多个模型的推理速度"""
        print("\n推理速度比较:")
        for name, model in models.items():
            times = []
            model.eval()
            with torch.no_grad():
                # 预热
                for _ in range(10):
                    _ = model(inputs)
                
                # 正式测量
                for _ in range(num_runs):
                    start_time = time.time()
                    _ = model(inputs)
                    torch.cuda.synchronize()  # 确保GPU操作完成
                    times.append(time.time() - start_time)
            
            avg_time = np.mean(times) * 1000  # 转换为毫秒
            std_time = np.std(times) * 1000
            print(f"{name}: {avg_time:.2f} ms ± {std_time:.2f} ms")
    
    def compare_memory_usage(self, models, inputs):
        """比较多个模型的内存使用情况"""
        print("\n内存使用比较:")
        for name, model in models.items():
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                model.eval()
                with torch.no_grad():
                    _ = model(inputs)
                max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                print(f"{name}: {max_memory:.2f} MB")
            else:
                print(f"{name}: 无法在CPU模式下测量显存使用")
    
    def analyze_model(self, model, inputs, model_name):
        """分析模型的计算开销和性能"""
        print(f"\n=== 分析 {model_name} 模型性能 ===")
        
        # 1. 计算FLOPs和参数量
        try:
            # 创建一个示例输入
            example_inputs = []
            for x in inputs:
                if isinstance(x, torch.Tensor):
                    example_inputs.append(x)
                elif isinstance(x, (list, tuple)):
                    example_inputs.extend([t for t in x if isinstance(t, torch.Tensor)])
            
            macs, params = thop_profile(model, inputs=(example_inputs,))
            flops = macs * 2  # 转换 MACs 到 FLOPs
            macs, flops, params = clever_format([macs, flops, params], "%.3f")
            print(f"MACs: {macs}")
            print(f"FLOPs: {flops}")
            print(f"参数量: {params}")
        except Exception as e:
            print(f"Warning: 无法计算FLOPs: {str(e)}")
            print(f"错误详情: {e.__class__.__name__}")
            print(f"错误位置: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")
            macs, flops, params = "N/A", "N/A", "N/A"
        
        # 2. 测量推理时间
        times = []
        with torch.no_grad():
            # 预热
            for _ in range(10):
                _ = model(inputs)
            
            # 正式测量
            for _ in range(100):
                start_time = time.time()
                _ = model(inputs)
                torch.cuda.synchronize()  # 确保GPU操作完成
                times.append(time.time() - start_time)
        
        avg_time = np.mean(times) * 1000  # 转换为毫秒
        std_time = np.std(times) * 1000
        
        # 3. 使用PyTorch Profiler获取详细信息
        try:
            with torch_profile(
                activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA
                ],
                record_shapes=True,
                with_stack=True,
                profile_memory=True
            ) as prof:
                with record_function("model_inference"):
                    _ = model(inputs)
        except Exception as e:
            print(f"Warning: 性能分析失败: {str(e)}")
            prof = None
        
        # 4. 内存使用分析
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            _ = model(inputs)
            max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        else:
            max_memory = 0
            
        # 存储结果
        self.results[model_name] = {
            'macs': macs,
            'params': params,
            'avg_time': avg_time,
            'std_time': std_time,
            'max_memory': max_memory,
            'profile': prof
        }
        
        # 打印结果
        print(f"\n{model_name} 性能指标:")
        print(f"MACs: {macs}")
        print(f"FLOPs: {flops}")
        print(f"参数量: {params}")
        print(f"平均推理时间: {avg_time:.2f} ms ± {std_time:.2f} ms")
        print(f"峰值显存使用: {max_memory:.2f} MB")
        
        # 打印详细的性能分析
        if prof is not None:
            print("\n操作级别性能分析:")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
    def compare_models(self, models_dict, test_input):
        """比较多个模型的性能"""
        for name, model in models_dict.items():
            self.analyze_model(model, test_input, name)
            
        # 绘制比较图表
        self.plot_comparison()
        
    def plot_comparison(self):
        """绘制模型性能对比图"""
        # 暂时注释掉绘图功能
        pass 