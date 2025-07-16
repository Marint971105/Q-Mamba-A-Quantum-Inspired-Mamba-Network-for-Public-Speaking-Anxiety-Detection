# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

class Features4QuantumFudanReader:
    def __init__(self, opt):
        """初始化数据读取器
        
        Args:
            opt: 配置参数对象
        """
        self.opt = opt
        self.batch_size = opt.batch_size
        self.device = opt.device
        
        # 设置文件路径
        self.fp_prefix = os.path.join(opt.pickle_dir_path, 'Features4Quantum')
        self.train_path = os.path.join(self.fp_prefix, 'fudan_train/features.pkl')
        self.test_path = os.path.join(self.fp_prefix, 'fudan_test/features.pkl')
        self.dev_path = os.path.join(self.fp_prefix, 'fudan_val/features.pkl')
        
        # 初始化必要的属性
        self.train_sample_num = 0
        self.input_dims = None
        self.output_dim = 4  # Fudan数据集为4分类
        
        # 添加标签映射字典
        self.label_map = {
            'A': 0,
            'B': 1,
            'C': 2,
            'D': 3
        }
        
        # 设置情感标签字典（保持与CN数据集一致的数字标签）
        self.emotion_dic = ['0', '1', '2', '3']
        
        # 读取数据
        self.read(opt)
        
    def _get_features(self, data):
        """从数据中提取特征
        
        Args:
            data: 包含特征的数据列表
            
        Returns:
            features_list: 提取的特征列表
        """
        text_features = []
        audio_features = []
        beats_features = []
        visual_features = []
        
        for item in data:
            text_features.append(torch.FloatTensor(item['text_features']))
            audio_features.append(torch.FloatTensor(item['audio_features']))
            beats_features.append(torch.FloatTensor(item['beats_features']))
            visual_features.append(torch.FloatTensor(item['video_features']))
        
        # 填充序列到相同长度
        text_features = pad_sequence(text_features, batch_first=True)
        audio_features = pad_sequence(audio_features, batch_first=True)
        beats_features = pad_sequence(beats_features, batch_first=True)
        visual_features = pad_sequence(visual_features, batch_first=True)
        
        return [text_features, audio_features, beats_features, visual_features]
    
    def _get_labels(self, data):
        """从数据中提取标签并进行映射
        
        Args:
            data: 包含标签的数据列表
            
        Returns:
            labels: 标签张量
        """
        labels = []
        for item in data:
            # 将字母标签映射为数字
            label = self.label_map[item['label']]
            one_hot = torch.zeros(self.output_dim)
            one_hot[label] = 1
            labels.append(one_hot)
        return torch.stack(labels)
    
    def read(self, opt):
        """读取数据集
        
        Args:
            opt: 配置参数对象
        """
        # 读取训练集
        print("\n加载Fudan训练集...")
        with open(self.train_path, 'rb') as f:
            train_data = pickle.load(f)
        
        # 获取特征和标签
        train_x = self._get_features(train_data)
        train_y = self._get_labels(train_data)
        
        # 打印各个模态的shape
        print("\n=== 特征维度详情 ===")
        modality_names = ['文本特征', '音频特征', '节拍特征', '视觉特征']
        for name, features in zip(modality_names, train_x):
            print(f"{name} shape: {features.shape}")
        print(f"标签 shape: {train_y.shape}")
        print("="*30)
        
        # 更新训练样本数和特征维度
        self.train_sample_num = len(train_y)
        self.input_dims = [x.shape[-1] for x in train_x]
        
        # 打印数据集统计信息
        print(f"\n=== Fudan数据集统计 ===")
        print(f"训练集样本数: {self.train_sample_num}")
        print("特征维度:")
        feature_names = ['文本特征', '音频特征', '节拍特征', '视觉特征']
        for name, dim in zip(feature_names, self.input_dims):
            print(f"{name}: {dim}")
        
        # 统计类别分布
        class_counts = torch.zeros(self.output_dim)
        for label in train_y.argmax(dim=1):
            class_counts[label] += 1
        
        print("\n类别分布:")
        for i, count in enumerate(class_counts):
            print(f"类别 {i}: {int(count)} 样本 ({count/self.train_sample_num:.2%})")
        
        # 读取测试集和验证集
        print("\n加载测试集和验证集...")
        with open(self.test_path, 'rb') as f:
            test_data = pickle.load(f)
        test_x = self._get_features(test_data)
        test_y = self._get_labels(test_data)
        
        with open(self.dev_path, 'rb') as f:
            dev_data = pickle.load(f)
        dev_x = self._get_features(dev_data)
        dev_y = self._get_labels(dev_data)
        
        # 存储数据
        self.datas = {
            'train': {'X': train_x, 'y': train_y},
            'test': {'X': test_x, 'y': test_y},
            'dev': {'X': dev_x, 'y': dev_y}
        }
        
        print(f"测试集样本数: {len(test_y)}")
        print(f"验证集样本数: {len(dev_y)}")
    
    def get_data(self, iterable=True, shuffle=False, split='train'):
        """获取数据迭代器或数据集
        
        Args:
            iterable: 是否返回迭代器
            shuffle: 是否打乱数据
            split: 数据集划分('train', 'test', 'dev')
            
        Returns:
            data_iter: 数据迭代器或数据集
        """
        data = self.datas[split]
        dataset = Features4QuantumFudanDataset(data['X'], data['y'])
        
        if iterable:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=0
            )
        return dataset

class Features4QuantumFudanDataset(Dataset):
    def __init__(self, features, labels):
        """初始化数据集
        
        Args:
            features: 特征列表
            labels: 标签张量
        """
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        """获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            sample: 包含特征和标签的样本元组
        """
        features = [f[idx] for f in self.features]
        label = self.labels[idx]
        return (*features, label) 