# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from models.single_modal import TextModel, AudioModel, VideoModel, ModalMLP


class MultiModalFusion(nn.Module):
    def __init__(self, opt, text_model=None, audio_model=None, video_model=None):
        """多模态融合模型

        Args:
            opt: 配置参数对象
            text_model: 预训练的文本模型(可选)
            audio_model: 预训练的音频模型(可选)
            video_model: 预训练的视频模型(可选)
        """
        super().__init__()

        # 单模态模型
        if text_model is not None:
            self.text_model = text_model
        else:
            self.text_model = TextModel(opt)

        if audio_model is not None:
            self.audio_model = audio_model
        else:
            self.audio_model = AudioModel(opt)

        if video_model is not None:
            self.video_model = video_model
        else:
            self.video_model = VideoModel(opt)
        # 多模态融合层
        self.fusion = nn.Sequential(
            nn.Linear(opt.hidden_dim * 3, opt.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(opt.dropout),
            nn.Linear(opt.hidden_dim * 2, opt.hidden_dim)
        )

        # 分类头
        self.classifier = nn.Linear(opt.hidden_dim, opt.num_classes)

    def forward(self, text_x, audio_x, video_x):
        """前向传播

        Args:
            text_x: 文本特征 [batch_size, text_seq_len, text_dim]
            audio_x: 音频特征 [batch_size, audio_seq_len, audio_dim]
            video_x: 视频特征 [batch_size, video_seq_len, video_dim]

        Returns:
            logits: 分类输出
            modal_outputs: 各模态的分类输出
            fused_features: 融合后的特征
        """
        # 获取各模态的分类输出和投影特征
        text_logits, text_features = self.text_model(text_x)
        audio_logits, audio_features = self.audio_model(audio_x)
        video_logits, video_features = self.video_model(video_x)

        # 特征拼接
        concat_features = torch.cat([
            text_features,
            audio_features,
            video_features
        ], dim=1)  # [batch_size, hidden_dim * 3]

        # 特征融合
        # [batch_size, hidden_dim]
        fused_features = self.fusion(concat_features)

        # 分类
        logits = self.classifier(fused_features)  # [batch_size, num_classes]

        # 返回总体分类结果、各模态分类结果和融合特征
        modal_outputs = {
            'text': text_logits,
            'audio': audio_logits,
            'video': video_logits
        }

        return logits, modal_outputs, fused_features


class MultiModalMLP(nn.Module):
    def __init__(self, opt):
        """多模态融合MLP模型

        Args:
            opt: 配置参数对象
        """
        super().__init__()

        # 单模态MLP
        self.text_mlp = ModalMLP(
            opt.input_dims[0], opt.hidden_dim, opt.num_classes, opt.dropout)
        self.audio_mlp = ModalMLP(
            opt.input_dims[1], opt.hidden_dim, opt.num_classes, opt.dropout)
        self.video_mlp = ModalMLP(
            opt.input_dims[3], opt.hidden_dim, opt.num_classes, opt.dropout)  # 使用索引3的visual特征

        # 融合层
        fusion_input_dim = (opt.hidden_dim // 2) * 3  # 三个模态的特征维度之和
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, opt.hidden_dim),
            nn.ReLU(),
            nn.Dropout(opt.dropout),
            nn.Linear(opt.hidden_dim, opt.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(opt.dropout)
        )

        # 分类器
        self.classifier = nn.Linear(opt.hidden_dim // 2, opt.num_classes)

    def forward(self, text_x, audio_x, video_x):
        """前向传播

        Args:
            text_x: 文本特征 [batch_size, text_dim]
            audio_x: 音频特征 [batch_size, audio_dim]
            video_x: 视频特征 [batch_size, video_dim]

        Returns:
            logits: 分类输出
            modal_outputs: 各模态的分类输出
            fused_features: 融合后的特征
        """
        # 获取各模态的分类输出和特征
        text_logits, text_features = self.text_mlp(text_x)
        audio_logits, audio_features = self.audio_mlp(audio_x)
        video_logits, video_features = self.video_mlp(video_x)
        # 处理特征维度 - 移除多余的维度
        if text_features.dim() == 3:
            text_features = text_features.squeeze(1)
        if audio_features.dim() == 3:
            audio_features = audio_features.squeeze(1)
        if video_features.dim() == 3:
            video_features = video_features.squeeze(1)
        # 特征拼接
        concat_features = torch.cat([
            text_features, audio_features, video_features
        ], dim=1)
        # 特征融合
        fused_features = self.fusion_mlp(concat_features)

        # 分类
        logits = self.classifier(fused_features)

        # 返回总体分类结果、各模态分类结果和融合特征
        modal_outputs = {
            'text': text_logits,
            'audio': audio_logits,
            'video': video_logits
        }

        return logits, modal_outputs, fused_features
