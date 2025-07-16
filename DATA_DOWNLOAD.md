# 数据文件下载说明

由于GitHub对文件大小有限制，以下大文件不包含在代码仓库中。请按照以下方式获取这些文件：

## 需要下载的文件

### 数据集文件 (Data目录)(自行下载)
- `Data/meld_data.pkl` (88.28 MB)
- `Data/meld_data_act.pkl` (71.55 MB)
- `Data/meld_data_original.pkl` (71.04 MB)
- `Data/meld_train.pkl` (279.26 MB)
- `Data/meld_test.pkl` (75.33 MB)
- `Data/meld_valid.pkl`
- `Data/meld_context_train.pkl` (1689.28 MB)
- `Data/meld_context_test.pkl` (441.39 MB)
- `Data/meld_context_valid.pkl` (187.55 MB)
- `Data/iemocap_context_train.pkl` (248.94 MB)
- `Data/iemocap_context_test.pkl` (94.00 MB)
- `Data/iemocap_context_valid.pkl` (87.57 MB)

### 模型文件 (feature_extract目录)(自行下载)
- `feature_extract/BEATs_iter3_plus_AS2M.pt` (344.75 MB)

### 特征文件 (feature_extract/Features4Quantum目录)（下面提供方式）
- `feature_extract/Features4Quantum/fudan_train/features.pkl` (109.50 MB)
- `feature_extract/Features4Quantum/fudan_test/features.pkl`
- `feature_extract/Features4Quantum/fudan_val/features.pkl`
- `feature_extract/Features4Quantum/chinese_train/features.pkl`
- `feature_extract/Features4Quantum/chinese_test/features.pkl`
- `feature_extract/Features4Quantum/chinese_val/features.pkl`

## 获取方式

### 方式: HuggingFace下载
请从以下链接下载所需文件：
https://huggingface.co/datasets/MarvinP/BPSAD

## 目录结构
```
Data/
├── meld_data.pkl
├── meld_data_act.pkl
├── meld_data_original.pkl
├── meld_train.pkl
├── meld_test.pkl
├── meld_valid.pkl
├── meld_context_train.pkl
├── meld_context_test.pkl
├── meld_context_valid.pkl
├── iemocap_context_train.pkl
├── iemocap_context_test.pkl
└── iemocap_context_valid.pkl

feature_extract/
├── BEATs_iter3_plus_AS2M.pt
└── Features4Quantum/
    ├── fudan_train/
    │   └── features.pkl
    ├── fudan_test/
    │   └── features.pkl
    ├── fudan_val/
    │   └── features.pkl
    ├── chinese_train/
    │   └── features.pkl
    ├── chinese_test/
    │   └── features.pkl
    └── chinese_val/
        └── features.pkl
```

## 注意事项
1. 确保下载的文件完整且未损坏
2. 文件大小应与上述标注的大小一致
3. 如果使用脚本下载，请确保网络连接稳定
4. 建议使用断点续传工具下载大文件

