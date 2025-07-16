import os
import time
import torch
import torch.nn as nn
import numpy as np
from models.QMamba_meld import QMamba_meld
from utils.model import train, test,save_model, save_performance, get_predictions
from utils.evaluation import evaluate
import pickle
from dataset.meld_reader import MELDReader

def load_meld_data(split, data_dir):
    """加载MELD数据集"""
    print(f"  - Loading {split} data...")
    features = {}
    for modality in ['text', 'visual', 'acoustic']:
        path = os.path.join(data_dir, f'meld_{modality}_{split}.pkl')
        with open(path, 'rb') as f:
            features[modality] = pickle.load(f)
            
    # 加载标签
    label_path = os.path.join(data_dir, f'meld_labels_{split}.pkl')
    with open(label_path, 'rb') as f:
        labels = pickle.load(f)
        
    # 转换为tensor
    X = [torch.tensor(features[m], dtype=torch.float32) for m in ['text', 'visual', 'acoustic']]
    y = torch.tensor(labels, dtype=torch.long)
    
    # 打印数据集统计信息
    print(f"\n=== {split}集统计 ===\n")
    print(f"对话数: {len(y)//33}")  # 每个对话33个话语
    print(f"话语数: {len(y)}")
    print(f"平均对话长度: {len(y)//(len(y)//33):.2f}")
    print(f"最短对话长度: {33}")
    print(f"最长对话长度: {33}\n")
    
    # 打印情感分布
    print("情感类别分布:")
    for i, emotion in enumerate(['neutral', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']):
        count = (y == i).sum().item()
        percentage = count / len(y) * 100
        print(f"{emotion}: {count} ({percentage:.2f}%)")
    print()
    
    return X, y

class Options:
    def __init__(self):
        # 数据参数
        self.dataset_name = 'MELD'
        self.pickle_dir_path = '/home/tjc/audio/QMamba/Data'
        self.features = 'textual,visual,acoustic'
        self.label = 'emotion'
        self.batch_size = 32
        self.num_workers = 4
        
        # 模型参数
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.input_dims = [600, 300, 300]  # 文本、视觉、音频维度
        self.embed_dim = 100
        self.output_dim = 7  # 7种情感类别
        self.num_layers = 1
        self.sequence_model = 'mamba'
        self.rnn_type = 'rnn'
        self.bidirectional = False
        self.output_cell_dim = 50
        self.out_dropout_rate = 0.1
        self.network_type = 'QMamba'
        
        # 训练参数
        self.epochs = 50  # 先用小的epoch测试
        self.lr = 1e-3
        self.min_lr = 1e-5
        self.unitary_lr = 1e-3
        self.weight_decay = 1e-4
        self.patience = 10
        self.clip = 0.8
        
        # 损失函数权重
        self.loss_weights = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(self.device)
        
        # 数据集相关
        self.emotion_dic = ['neutral', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
        self.dialogue_context = False
        self.embedding_enabled = False  # 添加这个参数

def print_test_results(performances):
    print("\n" + "="*50)
    print("测试结果详情")
    print("="*50)
    
    # 获取测试集中每个类别的样本数量
    test_target = performances['test_target']  # 从test函数传递过来的标签
    class_counts = torch.bincount(test_target, minlength=7)  # 7个情感类别
    
    # 打印整体性能
    print("\n整体性能:")
    print(f"Accuracy: {performances['acc']:.4f}")
    print(f"Macro F1: {performances['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1: {performances['weighted avg']['f1-score']:.4f}")
    
    # 打印每个情感类别的性能
    print("\n各情感类别性能:")
    emotions = ['neutral', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    for i, emotion in enumerate(emotions):
        count = class_counts[i].item()
        percentage = count / class_counts.sum().item() * 100
        print(f"\n{emotion.capitalize()} (样本数: {count}, {percentage:.2f}%):")
        if str(i) in performances:
            print(f"  Accuracy: {performances[str(i)]['acc']:.4f}")
            print(f"  Precision: {performances[str(i)]['precision']:.4f}")
            print(f"  Recall: {performances[str(i)]['recall']:.4f}")
            print(f"  F1-score: {performances[str(i)]['f1-score']:.4f}")
        else:
            print("  Accuracy: 0.0000")
            print("  Precision: 0.0000")
            print("  Recall: 0.0000")
            print("  F1-score: 0.0000")
            print("  (No predictions for this class)")
    
    # 打印总样本数
    print(f"\n总样本数: {class_counts.sum().item()}")
    
    # 打印平均性能
    print("\n平均性能:")
    print("Macro Average:")
    print(f"  Precision: {performances['macro avg']['precision']:.4f}")
    print(f"  Recall: {performances['macro avg']['recall']:.4f}")
    print(f"  F1-score: {performances['macro avg']['f1-score']:.4f}")
    
    print("\nWeighted Average:")
    print(f"  Precision: {performances['weighted avg']['precision']:.4f}")
    print(f"  Recall: {performances['weighted avg']['recall']:.4f}")
    print(f"  F1-score: {performances['weighted avg']['f1-score']:.4f}")
    
    print("\n" + "="*50)

def main():
    # 初始化配置
    params = Options()
    
    print("\n=== MELD数据集上的QMamba模型测试 ===\n")
    
    # 使用MELDReader加载数据
    print("加载数据...")
    reader = MELDReader(params)
    reader.read(params)
    
    # 打印数据集原始分布
    print("\n=== 数据集类别分布 ===")
    train_data = reader.datas['train']['y']
    test_data = reader.datas['test']['y']
    dev_data = reader.datas['dev']['y']
    
    for split, data in [('训练集', train_data), ('验证集', dev_data), ('测试集', test_data)]:
        print(f"\n{split}情感分布:")
        # 将数据reshape成2D，然后统计每个类别的样本数
        reshaped_data = data.reshape(-1, params.output_dim)
        total = reshaped_data.sum(dim=0)  # [7]
        for i, emotion in enumerate(params.emotion_dic):
            count = total[i].item()
            percentage = count / total.sum().item() * 100
            print(f"{emotion.capitalize()}: {count} ({percentage:.2f}%)")
        print(f"总样本数: {total.sum().item()}")
    print("="*50)
    
    # 将reader添加到params中
    params.reader = reader
    
    # 获取数据加载器
    train_loader = reader.get_data(shuffle=True, split='train')
    dev_loader = reader.get_data(shuffle=False, split='dev')
    test_loader = reader.get_data(shuffle=False, split='test')
    
    # 更新params
    params.train_loader = train_loader
    params.dev_loader = dev_loader
    params.test_loader = test_loader
    
    # 初始化模型
    print("\n初始化模型...")
    model = QMamba_meld(params).to(params.device)
    print(model)
    
    # 打印数据维度
    print("\n=== 输入数据维度 ===")
    print(f"文本特征: torch.Size([{params.batch_size}, 33, {params.input_dims[0]}])")
    print(f"视觉特征: torch.Size([{params.batch_size}, 33, {params.input_dims[1]}])")
    print(f"音频特征: torch.Size([{params.batch_size}, 33, {params.input_dims[2]}])")
    print(f"标签: torch.Size([{params.batch_size}, 33, {params.output_dim}])")
    
    # 设置模型保存路径
    model_dir = f'results/{params.dataset_name.lower()}/{params.network_type}'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    params.best_model_file = os.path.join(model_dir, 'best_model.pth')
    
    # 训练模型
    print("\n开始训练...")
    train(params, model)
    
    # 测试模型
    print("\n开始测试...")
    # 加载最佳模型
    model.load_state_dict(torch.load(params.best_model_file))
    model.eval()
    performances = test(model, params)
    
    # 打印详细测试结果
    print_test_results(performances)
    
    # 保存模型和结果
    print("\n保存模型和结果...")
    save_model(model, params, performances)
    save_performance(params, performances)
    
    print(f"\n模型已保存到: {model_dir}")
    print(f"详细结果已保存到: eval/{params.dataset_name.lower()}/{params.network_type}_results.csv")

if __name__ == '__main__':
    main()