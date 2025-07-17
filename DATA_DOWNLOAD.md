# Data File Download Instructions
Due to GitHub's file size limitations, the following large files are not included in the code repository. Please obtain these files as follows:
Files to Download

## Files to Download
### Dataset Files (Data directory) (Download required)
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

### Model Files (feature_extract directory) (Download required)
- `feature_extract/BEATs_iter3_plus_AS2M.pt` (344.75 MB)

### Feature Files (feature_extract/Features4Quantum directory) (Download methods provided below)
- `feature_extract/Features4Quantum/fudan_train/features.pkl` (109.50 MB)
- `feature_extract/Features4Quantum/fudan_test/features.pkl`
- `feature_extract/Features4Quantum/fudan_val/features.pkl`
- `feature_extract/Features4Quantum/chinese_train/features.pkl`
- `feature_extract/Features4Quantum/chinese_test/features.pkl`
- `feature_extract/Features4Quantum/chinese_val/features.pkl`

## Download Methods

### Method: HuggingFace Download
Please download the required files from the following link:
https://huggingface.co/datasets/MarvinP/BPSAD

## Directory Structure
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

## Important Notes

Ensure downloaded files are complete and uncorrupted
File sizes should match the sizes noted above
If using scripts for download, ensure stable network connection
Recommend using download tools with resume capability for large files

