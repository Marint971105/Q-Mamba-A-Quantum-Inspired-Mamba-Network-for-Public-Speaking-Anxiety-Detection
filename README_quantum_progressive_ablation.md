# 渐进式量子组件消融实验使用说明

本实验旨在分析量子组件在Q-Mamba模型中的增量贡献，通过逐步添加量子机制来评估每个组件的效果。

## 实验设计

### 渐进式模型变体

1. **BaselineMamba** - 基线模型：纯Mamba + 传统L2Norm融合
2. **QuantumSuperposition** - 量子叠加：基线 + 量子叠加状态建模
3. **QuantumEntanglement** - 量子纠缠：量子叠加 + 量子纠缠
4. **QMamba** - 完整Q-Mamba：量子纠缠 + 多头注意力融合

### 实验模式

- **comparison** - 比较实验：运行所有模型并比较性能
- **ablation** - 消融实验：逐步添加量子组件
- **single** - 单个实验：运行指定的单个模型

## 使用方法

### CN数据集实验

```bash
# 运行比较实验（所有模型）
python run_quantum_progressive_ablation.py --mode comparison

# 运行消融实验
python run_quantum_progressive_ablation.py --mode ablation

# 运行单个模型实验
python run_quantum_progressive_ablation.py --mode single --model baseline
python run_quantum_progressive_ablation.py --mode single --model superposition
python run_quantum_progressive_ablation.py --mode single --model entanglement
python run_quantum_progressive_ablation.py --mode single --model qmamba
```

### Fudan数据集实验

```bash
# 运行比较实验（所有模型）
python run_quantum_progressive_ablation_fudan.py --mode comparison

# 运行消融实验
python run_quantum_progressive_ablation_fudan.py --mode ablation

# 运行单个模型实验
python run_quantum_progressive_ablation_fudan.py --mode single --model baseline
python run_quantum_progressive_ablation_fudan.py --mode single --model superposition
python run_quantum_progressive_ablation_fudan.py --mode single --model entanglement
python run_quantum_progressive_ablation_fudan.py --mode single --model qmamba
```

## 输出结果

### 结果文件

- `results/quantum_progressive/` - CN数据集结果
- `results/quantum_progressive_fudan/` - Fudan数据集结果

### 主要输出文件

1. **比较结果**
   - `quantum_progressive_comparison.json` / `fudan_quantum_progressive_comparison.json`
   - 包含所有模型的性能对比

2. **消融结果**
   - `quantum_ablation_results.json` / `fudan_quantum_ablation_results.json`
   - 包含消融实验的详细结果

3. **贡献分析**
   - `quantum_contributions.json` / `fudan_quantum_contributions.json`
   - 包含绝对贡献和增量贡献分析

### 贡献分析内容

- **绝对贡献**：每个模型的直接性能表现
- **增量贡献**：相对于前一个模型的性能提升
- **贡献百分比**：以百分比形式表示的贡献度

## 实验配置

### CN数据集配置
- 数据集：features4quantum
- 学习率：0.0005
- 批大小：64
- 嵌入维度：50

### Fudan数据集配置
- 数据集：features4quantum_fudan
- 学习率：0.0001
- 批大小：64
- 嵌入维度：50

## 注意事项

1. **数据准备**：确保特征提取目录 `/home/tjc/audio/QMamba/feature_extract` 存在
2. **GPU内存**：实验需要足够的GPU内存，建议使用至少8GB显存
3. **时间估算**：每个模型实验约需30-60分钟（取决于硬件配置）
4. **错误处理**：脚本包含错误处理机制，失败的实验会记录错误信息

## 结果解读

### 性能指标
- **Accuracy**：整体准确率
- **Macro F1**：宏平均F1分数
- **Weighted F1**：加权平均F1分数

### 贡献分析
- **正贡献**：性能提升，表示该量子组件有效
- **负贡献**：性能下降，表示该组件可能不适合当前任务
- **零贡献**：无明显影响，表示该组件可能冗余

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小batch_size
   - 减小embed_dim
   - 使用CPU模式

2. **数据读取错误**
   - 检查特征提取路径
   - 确认数据集名称正确

3. **模型导入错误**
   - 确认所有模型文件存在
   - 检查Python路径设置

### 调试模式

可以修改脚本中的参数来启用调试模式：
- 减少epochs数量
- 使用更小的数据集
- 启用详细日志输出

## 扩展实验

### 自定义实验
可以修改脚本中的参数来运行自定义实验：
- 调整学习率
- 修改模型架构
- 更改训练策略

### 新数据集支持
要支持新数据集，需要：
1. 创建对应的数据读取器
2. 修改setup_params函数
3. 调整模型参数

## 联系信息

如有问题或建议，请联系开发团队。 