# 多模态分歧识别与分析系统

这个系统实现了基于量子多模态数据集的分歧识别和分析功能，包括单模态模型训练、多模态融合和分歧识别分析。

## 系统概述

本系统借鉴了现有的QMamba项目的数据读取方法，实现了以下功能：

### 1. 单模态模型训练
- **文本模态模型**: 基于RoBERTa/DeBERTa风格的Transformer架构
- **音频模态模型**: 基于HuBERT/Wav2Vec2风格的卷积神经网络
- **视频模态模型**: 基于VideoMAE/TimesFormer风格的时空注意力机制

### 2. 多模态融合模型
- 提取最佳单模态模型的嵌入特征
- 训练多模态融合模型，将三个模态嵌入投射到共享潜在空间
- 使用跨模态注意力机制进行特征融合

### 3. 分歧识别方法
- 在测试集上评估所有模型，识别模态分歧
- 分歧定义：不同输入模态的模型给出不同预测标签的样本
- 创建置信度散点图，形成四个象限：
  - **红色**：多模态错误，单模态正确
  - **绿色**：多模态正确，单模态错误
  - **蓝色**：两者都正确
  - **黄色**：两者都错误

## 文件结构

```
├── multimodal_divergence_analysis.py  # 主要系统实现
├── test_multimodal_divergence.py      # 功能测试脚本
├── run_qmamba_CN.py                   # 修复后的原始脚本
└── README_multimodal_divergence.md    # 本说明文档
```

## 依赖要求

```python
torch>=1.9.0
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

## 使用方法

### 1. 快速功能测试

首先运行测试脚本验证系统功能：

```bash
python test_multimodal_divergence.py
```

这将测试：
- 数据加载功能
- 单模态模型前向传播
- 多模态融合模型
- 分歧分析器基本功能

### 2. 完整训练和分析

运行完整的多模态分歧分析：

```bash
python multimodal_divergence_analysis.py
```

### 3. 自定义参数

可以在`setup_params()`函数中修改以下参数：

```python
opt = SimpleNamespace(
    # 数据路径
    pickle_dir_path='/path/to/your/features',
    
    # 模型参数
    hidden_dim=256,        # 单模态隐藏层维度
    fusion_dim=512,        # 融合层维度
    dropout=0.1,           # Dropout率
    
    # 训练参数
    epochs=50,             # 训练轮数
    batch_size=64,         # 批大小
    lr=0.001,              # 学习率
    patience=10,           # 早停耐心值
    
    # 保存路径
    save_dir='results/multimodal_divergence',
    model_save_dir='models/multimodal',
)
```

## 模型架构详解

### 1. 文本模态模型 (TextModalModel)

```python
class TextModalModel(SingleModalModel):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=4):
        # 多头自注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=8, dropout=dropout
        )
        # 特征提取层 + LayerNorm
        # 分类头
```

**特点**：
- 使用多头自注意力捕获文本序列中的长距离依赖
- LayerNorm稳定训练过程
- 平均池化处理可变长度序列

### 2. 音频模态模型 (AudioModalModel)

```python
class AudioModalModel(SingleModalModel):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=4):
        # 1D卷积层 + BatchNorm
        self.conv_layers = nn.Sequential(...)
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
```

**特点**：
- 使用1D卷积捕获音频的时序特征
- BatchNorm加速收敛
- 全局平均池化提取全局特征

### 3. 视频模态模型 (VideoModalModel)

```python
class VideoModalModel(SingleModalModel):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=4):
        # 时序注意力机制
        self.temporal_attention = nn.MultiheadAttention(...)
        # 位置编码
        self.pos_encoding = nn.Parameter(...)
```

**特点**：
- 位置编码捕获时序信息
- 时序注意力机制处理视频帧间关系
- 适应可变长度视频序列

### 4. 多模态融合模型 (MultimodalFusionModel)

```python
class MultimodalFusionModel(nn.Module):
    def __init__(self, modal_hidden_dims, fusion_dim=512, output_dim=4):
        # 模态投影层（投射到共享潜在空间）
        self.modal_projections = nn.ModuleList(...)
        # 跨模态注意力
        self.cross_attention = nn.MultiheadAttention(...)
        # 可学习的模态权重
        self.modal_weights = nn.Parameter(...)
```

**特点**：
- 将不同模态特征投射到统一的潜在空间
- 跨模态注意力机制学习模态间交互
- 可学习的模态权重自适应调整模态重要性

## 分歧识别算法

### 1. 分歧定义

对于每个测试样本，如果满足以下任一条件，则认为存在模态分歧：

1. **单模态内部分歧**：文本、音频、视频三个单模态模型的预测结果不一致
2. **单模态与融合模型分歧**：融合模型的预测与任一单模态模型预测不同

### 2. 象限分类

根据模型预测的正确性，将样本分为四个象限：

```python
if fusion_correct and single_modal_correct:
    quadrant = 'blue'     # 两者都正确
elif fusion_correct and not single_modal_correct:
    quadrant = 'green'    # 融合正确，单模态错误
elif not fusion_correct and single_modal_correct:
    quadrant = 'red'      # 融合错误，单模态正确
else:
    quadrant = 'yellow'   # 两者都错误
```

### 3. 分析输出

系统会生成以下分析结果：

1. **分歧报告** (`divergence_report.json`)：
   - 总体分歧统计
   - 象限分布
   - 类别级分歧率

2. **置信度散点图** (`confidence_scatter_*.png`)：
   - 单模态 vs 融合模型置信度对比
   - 四象限颜色编码

3. **分歧样本详情** (`divergence_samples.csv`)：
   - 具体分歧样本信息
   - 各模型预测和置信度

## 输出结果解读

### 1. 分歧报告示例

```json
{
  "total_samples": 1000,
  "divergent_samples": 150,
  "divergence_rate": 0.15,
  "quadrant_distribution": {
    "blue": {"count": 800, "percentage": 80.0},
    "green": {"count": 100, "percentage": 10.0},
    "red": {"count": 80, "percentage": 8.0},
    "yellow": {"count": 20, "percentage": 2.0}
  },
  "class_wise_divergence": {
    "class_0": {"divergence_rate": 0.12},
    "class_1": {"divergence_rate": 0.18},
    "class_2": {"divergence_rate": 0.14},
    "class_3": {"divergence_rate": 0.16}
  }
}
```

### 2. 关键指标解释

- **分歧率 (divergence_rate)**：存在模态分歧的样本比例
- **象限分布**：
  - 蓝色：多模态和单模态都能正确预测的"简单"样本
  - 绿色：多模态融合带来性能提升的样本
  - 红色：多模态融合导致性能下降的样本
  - 黄色：所有模型都难以处理的"困难"样本

## 性能优化建议

### 1. 模型调优

- **增加模型容量**：可以增加`hidden_dim`和`fusion_dim`
- **调整学习率**：对不同模态使用不同的学习率
- **数据增强**：对音频和视频特征进行时序扰动

### 2. 训练策略

- **预训练**：可以先在大规模数据上预训练单模态模型
- **渐进训练**：先训练单模态，再训练融合模型
- **多任务学习**：同时训练分类和重构任务

### 3. 融合策略

- **注意力权重可视化**：分析跨模态注意力学习到的模态交互模式
- **动态权重**：让模态权重随输入样本动态变化
- **层次融合**：在多个层次进行特征融合

## 故障排除

### 1. 常见错误

**错误1**: `CUDA out of memory`
```
解决方案：
- 减少batch_size
- 减少hidden_dim和fusion_dim
- 使用梯度累积
```

**错误2**: 数据加载失败
```
解决方案：
- 检查pickle_dir_path路径是否正确
- 确认特征文件存在且格式正确
- 检查文件权限
```

**错误3**: 模型收敛慢
```
解决方案：
- 调整学习率
- 使用预训练权重
- 检查数据预处理和归一化
```

### 2. 调试技巧

1. **使用测试脚本**：先运行`test_multimodal_divergence.py`验证基本功能
2. **检查数据维度**：确认各模态特征维度匹配
3. **监控训练过程**：观察损失函数和准确率变化
4. **可视化特征**：使用t-SNE可视化学习到的特征表示

## 扩展功能

### 1. 支持更多模态

可以轻松扩展支持新的模态：

```python
class NewModalModel(SingleModalModel):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__(input_dim, hidden_dim, output_dim, 'new_modal')
        # 定义新模态特有的网络结构
```

### 2. 新的融合策略

可以实现不同的融合方法：

```python
class AdvancedFusionModel(nn.Module):
    def __init__(self):
        # 实现图神经网络、张量融合等方法
```

### 3. 更丰富的分析

可以添加更多分析维度：

- 模态重要性分析
- 错误样本聚类分析
- 时序分歧模式分析

## 联系与支持

如有问题或建议，请联系开发团队或提交Issue。

---

**注意**：本系统基于QMamba项目的数据读取框架开发，确保数据格式兼容性。 