# QMamba 权重计算方式消融实验

## 概述

本项目实现了QMamba模型中两种权重计算方式的消融实验：
1. **多头注意力方式** (`attention`): 使用ModalAttention计算模态权重
2. **模态大小方式** (`magnitude`): 使用L2Norm+softmax计算模态权重（类似QMN）

## 修改内容

### 1. QMamba模型修改 (`models/QMamba.py`)

- 添加了 `weight_method` 超参数
- 支持两种权重计算方式：
  - `attention`: 使用多头注意力
  - `magnitude`: 使用模态大小（L2Norm+softmax）

### 2. 新增消融实验脚本 (`run_qmamba_ablation.py`)

支持多种实验模式：
- 单一实验
- 权重计算方式比较
- 完整消融实验

## 使用方法

### CN数据集实验

#### 1. 运行权重计算方式比较实验

```bash
# 比较attention和magnitude两种方式
python run_qmamba_ablation.py --mode comparison
```

#### 2. 运行完整消融实验

```bash
# 运行所有组合的消融实验
python run_qmamba_ablation.py --mode ablation
```

#### 3. 运行单个实验

```bash
# 运行特定配置的实验
python run_qmamba_ablation.py --mode comparison --weight_method attention --sequence_model mamba
```

### Fudan数据集实验

#### 1. 运行权重计算方式比较实验

```bash
# 比较attention和magnitude两种方式
python run_qmamba_fudan_ablation.py --mode comparison
```

#### 2. 运行完整消融实验

```bash
# 运行所有组合的消融实验
python run_qmamba_fudan_ablation.py --mode ablation
```

#### 3. 运行单个实验

```bash
# 运行特定配置的实验
python run_qmamba_fudan_ablation.py --mode comparison --weight_method attention --sequence_model mamba
```

## 实验配置

### 权重计算方式

1. **attention方式**:
   - 使用ModalAttention（多头注意力）
   - 5个注意力头
   - 能够学习模态间的复杂交互关系

2. **magnitude方式**:
   - 使用L2Norm+softmax
   - 基于模态表示的"大小"计算权重
   - 计算简单高效

### 序列模型支持

- `mamba`: Mamba状态空间模型
- `transformer`: Transformer编码器
- `cnn`: 卷积神经网络
- `rnn`: 循环神经网络
- `rwkv`: RWKV模型

## 输出结果

### CN数据集结果
实验结果保存在 `results/qmamba_ablation/` 目录下：

1. **单个实验结果**: `results_{weight_method}_{sequence_model}_{timestamp}.json`
2. **比较结果**: `weight_comparison_{timestamp}.json`
3. **消融实验汇总**: `summary_{timestamp}.json`

### Fudan数据集结果
实验结果保存在 `results/qmamba_fudan_ablation/` 目录下：

1. **单个实验结果**: `results_{weight_method}_{sequence_model}_{timestamp}.json`
2. **比较结果**: `weight_comparison_{timestamp}.json`
3. **消融实验汇总**: `summary_{timestamp}.json`

### 结果格式

```json
{
  "weight_method": "attention",
  "sequence_model": "mamba",
  "total_params": 12345,
  "train_time": 120.5,
  "test_time": 5.2,
  "performance": {
    "accuracy": 0.85,
    "f1_score": 0.83
  },
  "config": {
    "embed_dim": 50,
    "num_layers": 1,
    "batch_size": 64,
    "lr": 0.0005,
    "epochs": 50
  }
}
```

## 实验分析

### 预期差异

1. **性能差异**:
   - attention方式可能在某些任务上表现更好
   - magnitude方式计算效率更高

2. **计算效率**:
   - attention方式参数量更多，训练时间更长
   - magnitude方式计算简单，收敛更快

3. **适用场景**:
   - attention方式适合模态间有复杂交互的任务
   - magnitude方式适合模态相对独立的任务

### 建议实验设置

1. **快速验证**: 使用 `--mode comparison` 比较两种权重计算方式
2. **完整分析**: 使用 `--mode ablation` 进行全面的消融实验
3. **特定场景**: 使用单个实验模式测试特定配置
4. **跨数据集比较**: 分别在CN和Fudan数据集上运行相同实验，比较不同数据集上的表现

## 注意事项

1. 确保数据集路径正确
2. 根据GPU内存调整batch_size
3. 实验结果会自动保存，避免重复运行
4. 可以通过修改 `setup_params` 函数调整实验参数 