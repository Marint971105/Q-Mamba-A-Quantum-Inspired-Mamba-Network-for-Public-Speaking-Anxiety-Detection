# 量子渐进式消融实验

本项目实现了量子机制的渐进式消融实验，通过逐步添加量子组件来验证每个组件的贡献。

## 实验设计

### 渐进式模型架构

1. **BaselineMamba** - 基线模型
   - 纯Mamba + 传统L2Norm融合
   - 不使用任何量子机制
   - 作为性能基准

2. **QuantumSuperposition** - 量子叠加模型
   - 基线 + 量子叠加状态建模
   - 使用相位和振幅构造量子态
   - 计算外积表示量子叠加

3. **QuantumEntanglement** - 量子纠缠模型
   - 量子叠加 + 量子纠缠
   - 使用Hadamard+CNOT实现量子纠缠
   - 对相邻模态应用CNOT操作

4. **QMamba** - 完整量子模型
   - 量子纠缠 + 多头注意力融合
   - 支持attention和magnitude两种权重计算方式
   - 包含所有量子组件

### 实验目标

- 验证量子组件的渐进贡献
- 分析每个量子机制的性能提升
- 提供量子机制有效性的实验证据

## 文件结构

```
models/
├── BaselineMamba.py              # 基线模型
├── QuantumSuperposition.py       # 量子叠加模型
├── QuantumEntanglement.py        # 量子纠缠模型
└── QMamba.py                     # 完整量子模型

run_quantum_progressive_ablation.py  # 主实验脚本
README_quantum_progressive.md        # 说明文档
```

## 使用方法

### 1. 比较实验（推荐）

运行所有模型的比较实验：

```bash
python run_quantum_progressive_ablation.py --mode comparison
```

这将依次运行：
- BaselineMamba
- QuantumSuperposition  
- QuantumEntanglement
- QMamba

### 2. 消融实验

运行消融实验：

```bash
python run_quantum_progressive_ablation.py --mode ablation
```

### 3. 单个实验

运行特定模型的实验：

```bash
# 基线模型
python run_quantum_progressive_ablation.py --mode single --model baseline

# 量子叠加模型
python run_quantum_progressive_ablation.py --mode single --model superposition

# 量子纠缠模型
python run_quantum_progressive_ablation.py --mode single --model entanglement

# 完整Q-Mamba
python run_quantum_progressive_ablation.py --mode single --model qmamba
```

## 实验配置

### 默认参数

- **数据集**: CN数据集 (features4quantum)
- **序列模型**: CNN
- **嵌入维度**: 50
- **层数**: 1
- **批大小**: 64
- **学习率**: 0.0005
- **训练轮数**: 50

### 支持的序列模型

所有模型都支持以下序列模型：
- `mamba` - Mamba块
- `transformer` - Transformer编码器
- `cnn` - CNN块
- `rnn` - RNN块
- `rwkv` - RWKV块

## 输出结果

### 结果文件

实验完成后会生成以下文件：

1. **比较结果**: `results/quantum_progressive/quantum_progressive_comparison.json`
2. **消融结果**: `results/quantum_progressive/quantum_ablation_results.json`
3. **增量贡献**: `results/quantum_progressive/incremental_contributions.json`

### 结果格式

```json
{
  "BaselineMamba": {
    "description": "基线模型：纯Mamba + 传统L2Norm融合",
    "performance": {
      "accuracy": 0.8234,
      "precision": 0.8156,
      "recall": 0.8234,
      "f1": 0.8194
    }
  },
  "QuantumSuperposition": {
    "description": "量子叠加：基线 + 量子叠加状态建模",
    "performance": {
      "accuracy": 0.8456,
      "precision": 0.8389,
      "recall": 0.8456,
      "f1": 0.8422
    }
  }
}
```

### 增量贡献分析

脚本会自动计算并显示每个量子组件的增量贡献：

```
量子组件增量贡献:
--------------------------------------------------------------------------------
模型                 前一个模型            当前精度    增量      增量%     
--------------------------------------------------------------------------------
QuantumSuperposition BaselineMamba        0.8456     0.0222    2.70%     
QuantumEntanglement   QuantumSuperposition 0.8567     0.0111    1.31%     
QMamba               QuantumEntanglement  0.8678     0.0111    1.30%     
```

## 模型修复说明

### 修复内容

1. **BaselineMamba**
   - 简化了序列处理逻辑
   - 使用传统L2Norm进行模态融合
   - 确保输出格式一致性

2. **QuantumSuperposition**
   - 修复了量子态处理逻辑
   - 添加了矩阵到向量的转换
   - 统一了量子态格式

3. **QuantumEntanglement**
   - 修复了量子纠缠操作
   - 简化了序列模型处理
   - 确保维度匹配

4. **QMamba**
   - 保持原有完整功能
   - 支持多种权重计算方式
   - 包含所有量子组件

### 兼容性

- 所有模型都使用相同的输入输出格式
- 支持相同的序列模型类型
- 使用统一的参数配置

## 故障排除

### 常见问题

1. **内存不足**
   - 减小batch_size
   - 减小embed_dim
   - 使用更少的层数

2. **训练不收敛**
   - 调整学习率
   - 增加训练轮数
   - 检查数据预处理

3. **模型加载失败**
   - 确保模型文件完整
   - 检查设备兼容性
   - 验证参数配置

### 调试模式

启用详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 扩展实验

### 自定义参数

可以修改`setup_params()`函数来调整实验参数：

```python
def setup_params():
    opt = SimpleNamespace(
        # 修改这些参数
        embed_dim=100,        # 更大的嵌入维度
        num_layers=2,         # 更多层数
        batch_size=32,        # 更小的批大小
        lr=0.001,            # 不同的学习率
        sequence_model='transformer',  # 不同的序列模型
        # ... 其他参数
    )
    return opt
```

### 添加新模型

1. 创建新的模型类
2. 继承相同的接口
3. 在实验脚本中添加模型映射
4. 更新文档

## 引用

如果使用本实验代码，请引用相关论文：

```bibtex
@article{qmamba2024,
  title={Quantum-Inspired Multi-Modal Fusion with Mamba},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件
- 参与讨论

---

**注意**: 本实验代码遵循不修改原始文件的原则，所有新功能都通过新增文件实现。 