# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataset.features4quantum_reader import Features4QuantumReader


class MultiModalReader(Features4QuantumReader):
    def __init__(self, opt):
        """多模态数据读取器

        Args:
            opt: 配置参数对象
        """
        super().__init__(opt)

        # 设置模态特征维度
        self.text_dim = self.input_dims[0]
        self.audio_dim = self.input_dims[1]
        self.video_dim = self.input_dims[3]

        # 创建多模态数据加载器
        self.train_loader = self.get_data(split='train', shuffle=True)
        self.dev_loader = self.get_data(split='dev', shuffle=False)
        self.test_loader = self.get_data(split='test', shuffle=False)

        # 创建单模态数据加载器
        self.text_train_loader = self.get_single_modal_data(
            'text', split='train', shuffle=True)
        self.text_dev_loader = self.get_single_modal_data(
            'text', split='dev', shuffle=False)
        self.text_test_loader = self.get_single_modal_data(
            'text', split='test', shuffle=False)

        self.audio_train_loader = self.get_single_modal_data(
            'audio', split='train', shuffle=True)
        self.audio_dev_loader = self.get_single_modal_data(
            'audio', split='dev', shuffle=False)
        self.audio_test_loader = self.get_single_modal_data(
            'audio', split='test', shuffle=False)

        self.video_train_loader = self.get_single_modal_data(
            'video', split='train', shuffle=True)
        self.video_dev_loader = self.get_single_modal_data(
            'video', split='dev', shuffle=False)
        self.video_test_loader = self.get_single_modal_data(
            'video', split='test', shuffle=False)

        print("\n=== 多模态数据集信息 ===")
        print(f"文本特征维度: {self.text_dim}")
        print(f"音频特征维度: {self.audio_dim}")
        print(f"视频特征维度: {self.video_dim}")
        print("="*30)

    def get_single_modal_data(self, modal_type, split='train', shuffle=False):
        """获取单模态数据加载器

        Args:
            modal_type: 模态类型('text', 'audio', 'video')
            split: 数据集划分('train', 'dev', 'test')
            shuffle: 是否打乱数据

        Returns:
            data_loader: 数据加载器
        """
        data = self.datas[split]

        if modal_type == 'text':
            modal_idx = 0
        elif modal_type == 'audio':
            modal_idx = 1
        elif modal_type == 'video':
            modal_idx = 3
        elif modal_type == 'beats':
            modal_idx = 2  # 使用beats_features
        else:
            raise ValueError(f"不支持的模态类型: {modal_type}")

        dataset = SingleModalDataset(
            features=data['X'][modal_idx],
            labels=data['y']
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0
        )


class MultiModalDataset(Dataset):
    def __init__(self, text_features, audio_features, video_features, labels):
        """多模态数据集

        Args:
            text_features: 文本特征
            audio_features: 音频特征
            video_features: 视频特征
            labels: 标签
        """
        self.text_features = text_features
        self.audio_features = audio_features
        self.video_features = video_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """获取单个样本

        Args:
            idx: 样本索引

        Returns:
            sample: (文本特征, 音频特征, 视频特征, 标签)
        """
        text = self.text_features[idx]
        audio = self.audio_features[idx]
        video = self.video_features[idx]
        label = self.labels[idx]

        return text, audio, video, label


class SingleModalDataset(Dataset):
    def __init__(self, features, labels):
        """单模态数据集

        Args:
            features: 特征
            labels: 标签
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
            sample: (特征, 标签)
        """
        feature = self.features[idx]
        label = self.labels[idx]

        return {'feature': feature, 'label': label}
