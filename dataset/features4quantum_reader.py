# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

class Features4QuantumReader:
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
        
        # 设置CN数据集的路径
        self.train_path = os.path.join(self.fp_prefix, 'chinese_train/features.pkl')
        self.test_path = os.path.join(self.fp_prefix, 'chinese_test/features.pkl')
        self.dev_path = os.path.join(self.fp_prefix, 'chinese_val/features.pkl')
        
        # 初始化必要的属性
        self.train_sample_num = 0
        self.input_dims = None
        self.output_dim = 4  # Fudan数据集为4分类
        
        # 读取数据
        self.read(opt)
        
        # 设置情感标签字典
        self.emotion_dic = ['0', '1', '2', '3']
        
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
        """从数据中提取标签
        
        Args:
            data: 包含标签的数据列表
            
        Returns:
            labels: 标签张量
        """
        labels = []
        for item in data:
            label = item['label']
            one_hot = torch.zeros(self.output_dim)
            one_hot[label] = 1
            labels.append(one_hot)
        return torch.stack(labels)
    
    def read(self, opt):
        """读取数据"""
        # 读取训练集
        with open(self.train_path, 'rb') as f:
            train_data = pickle.load(f)
        
        # 获取特征和标签
        train_x = self._get_features(train_data)
        train_y = self._get_labels(train_data)
        
        # 获取各个类别的索引
        y_indices = train_y.argmax(dim=1)
        class0_indices = (y_indices == 0).nonzero(as_tuple=True)[0]
        class1_indices = (y_indices == 1).nonzero(as_tuple=True)[0]
        class2_indices = (y_indices == 2).nonzero(as_tuple=True)[0]
        class3_indices = (y_indices == 3).nonzero(as_tuple=True)[0]
        
        # 找出最大类别的样本数
        max_samples = max(len(class0_indices), len(class1_indices), 
                         len(class2_indices), len(class3_indices))
        
        # 计算每个类别需要重复的次数以达到最大类别数量
        class0_multiplier = int(np.ceil(max_samples / len(class0_indices)))
        class1_multiplier = int(np.ceil(max_samples / len(class1_indices)))
        class2_multiplier = int(np.ceil(max_samples / len(class2_indices)))
        class3_multiplier = int(np.ceil(max_samples / len(class3_indices)))
        
        # 计算平衡后的样本数
        balanced_samples = max_samples * 4  # 每个类别都达到最大数量
        
        # 再将总样本数翻倍
        total_samples = balanced_samples * 2
        
        # 创建新的数据列表
        new_x = [torch.empty((total_samples,) + x_modal.shape[1:], 
                           dtype=x_modal.dtype, 
                           device=x_modal.device) for x_modal in train_x]
        new_y = torch.empty((total_samples,) + train_y.shape[1:],
                          dtype=train_y.dtype,
                          device=train_y.device)
        
        current_idx = 0
        
        # 第一次采样：将所有类别采样到最大类别数量
        for indices, multiplier in [
            (class0_indices, class0_multiplier),
            (class1_indices, class1_multiplier),
            (class2_indices, class2_multiplier),
            (class3_indices, class3_multiplier)
        ]:
            # 计算需要的样本数
            target_count = max_samples
            samples_needed = target_count
            
            # 重复采样直到达到目标数量
            while samples_needed > 0:
                # 确定本轮采样数量
                samples_this_round = min(len(indices), samples_needed)
                # 如果需要的样本数大于可用的样本数，则重复使用
                idx_to_use = indices[:samples_this_round]
                
                # 复制数据
                for idx in idx_to_use:
                    for i in range(len(train_x)):
                        new_x[i][current_idx:current_idx+1] = train_x[i][idx:idx+1]
                    new_y[current_idx:current_idx+1] = train_y[idx:idx+1]
                    current_idx += 1
                    samples_needed -= 1
                
                # 如果已用完所有样本但还需要更多，重新开始使用
                if samples_needed > 0:
                    indices = indices.roll(1)
        
        # 第二次采样：复制一份平衡后的数据
        balanced_size = current_idx
        for i in range(len(train_x)):
            new_x[i][balanced_size:] = new_x[i][:balanced_size]
        new_y[balanced_size:] = new_y[:balanced_size]
        
        # 更新训练数据
        train_x = new_x
        train_y = new_y
        
        # 更新训练样本数和特征维度
        self.train_sample_num = len(train_y)
        self.input_dims = [x.shape[-1] for x in train_x]
        
        # 打印数据集统计信息
        print(f"\n=== 数据集统计 ===")
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
        dataset = Features4QuantumDataset(data['X'], data['y'])
        
        if iterable:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=0
            )
        return dataset

class Features4QuantumDataset(Dataset):
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