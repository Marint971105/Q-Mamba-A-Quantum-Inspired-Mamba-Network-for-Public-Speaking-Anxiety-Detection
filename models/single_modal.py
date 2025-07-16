# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoConfig,
    RobertaModel,
    Wav2Vec2Model,
    VideoMAEModel
)


class TextModel(nn.Module):
    def __init__(self, opt):
        """文本模型(RoBERTa)

        Args:
            opt: 配置参数对象
        """
        super().__init__()

        # 加载预训练模型
        self.roberta = RobertaModel.from_pretrained(
            'hfl/chinese-roberta-wwm-ext')

        # 特征投影层
        self.projection = nn.Sequential(
            nn.Linear(768, opt.hidden_dim),
            nn.ReLU(),
            nn.Dropout(opt.dropout),
            nn.Linear(opt.hidden_dim, opt.hidden_dim)
        )

        # 分类头
        self.classifier = nn.Linear(opt.hidden_dim, opt.num_classes)

    def forward(self, x):
        """前向传播

        Args:
            x: 输入特征 [batch_size, seq_len, feature_dim]

        Returns:
            logits: 分类输出
            projected: 投影后的特征
        """
        # 获取RoBERTa特征
        # [batch_size, seq_len, 768]
        outputs = self.roberta(inputs_embeds=x)[0]

        # 使用[CLS]标记的输出
        pooled = outputs[:, 0, :]  # [batch_size, 768]

        # 特征投影
        projected = self.projection(pooled)  # [batch_size, hidden_dim]

        # 分类
        logits = self.classifier(projected)  # [batch_size, num_classes]

        return logits, projected


class AudioModel(nn.Module):
    def __init__(self, opt):
        """音频模型(Wav2Vec2)

        Args:
            opt: 配置参数对象
        """
        super().__init__()

        # 加载预训练模型
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            'facebook/wav2vec2-base-960h')

        # 特征投影层
        self.projection = nn.Sequential(
            nn.Linear(768, opt.hidden_dim),
            nn.ReLU(),
            nn.Dropout(opt.dropout),
            nn.Linear(opt.hidden_dim, opt.hidden_dim)
        )

        # 分类头
        self.classifier = nn.Linear(opt.hidden_dim, opt.num_classes)

    def forward(self, x):
        """前向传播

        Args:
            x: 输入特征 [batch_size, seq_len, feature_dim]

        Returns:
            logits: 分类输出
            projected: 投影后的特征
        """
        # 获取Wav2Vec2特征
        # [batch_size, seq_len, 768]
        outputs = self.wav2vec2(inputs_embeds=x)[0]

        # 平均池化
        pooled = torch.mean(outputs, dim=1)  # [batch_size, 768]

        # 特征投影
        projected = self.projection(pooled)  # [batch_size, hidden_dim]

        # 分类
        logits = self.classifier(projected)  # [batch_size, num_classes]

        return logits, projected


class VideoModel(nn.Module):
    def __init__(self, opt):
        """视频模型(VideoMAE)

        Args:
            opt: 配置参数对象
        """
        super().__init__()

        # 加载预训练模型
        self.videomae = VideoMAEModel.from_pretrained('MCG-NJU/videomae-base')

        # 特征投影层
        self.projection = nn.Sequential(
            nn.Linear(768, opt.hidden_dim),
            nn.ReLU(),
            nn.Dropout(opt.dropout),
            nn.Linear(opt.hidden_dim, opt.hidden_dim)
        )

        # 分类头
        self.classifier = nn.Linear(opt.hidden_dim, opt.num_classes)

    def forward(self, x):
        """前向传播

        Args:
            x: 输入特征 [batch_size, seq_len, feature_dim]

        Returns:
            logits: 分类输出
            projected: 投影后的特征
        """
        # 获取VideoMAE特征
        # [batch_size, seq_len, 768]
        outputs = self.videomae(pixel_values=x)[0]

        # 使用[CLS]标记的输出
        pooled = outputs[:, 0, :]  # [batch_size, 768]

        # 特征投影
        projected = self.projection(pooled)  # [batch_size, hidden_dim]

        # 分类
        logits = self.classifier(projected)  # [batch_size, num_classes]

        return logits, projected


class ModalMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        """单模态MLP模型

        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（类别数）
            dropout: Dropout比率
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        """前向传播

        Args:
            x: 输入特征 [batch_size, input_dim]

        Returns:
            logits: 分类输出
            features: 特征表示
        """
        features = self.mlp(x)
        logits = self.classifier(features)

        return logits, features
