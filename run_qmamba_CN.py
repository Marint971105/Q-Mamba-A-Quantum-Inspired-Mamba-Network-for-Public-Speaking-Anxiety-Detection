# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from types import SimpleNamespace
from dataset.features4quantum_reader import Features4QuantumReader
from models.QMamba import QMamba
from utils.model import train, test, save_model, save_performance
import random


def setup_params():
    """设置参数"""
    opt = SimpleNamespace(
        # 数据相关
        pickle_dir_path='/home/tjc/audio/QMamba/feature_extract',
        features='textual,acoustic,beats,visual',
        dataset_name='features4quantum',  # CN数据集
        embedding_enabled=False,
        dialogue_context=False,

        # 模型架构
        network_type='qmamba',
        sequence_model='cnn',
        # CNN特有参数
        kernel_size=3,      # 新增：CNN的卷积核大小
        cnn_dropout=0.1,    # 新增：CNN的dropout率
        # 模型参数
        embed_dim=50,
        num_layers=1,
        output_cell_dim=50,
        output_dim=4,
        speaker_num=1,

        # 训练参数
        epochs=50,
        patience=10,
        batch_size=64,
        lr=0.0005,
        unitary_lr=0.001,
        out_dropout_rate=0.2,
        clip=1.0,
        min_lr=1e-5,

        # 其他设置
        gpu=0,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        save_dir='results/qmamba',
        output_file=None,
        config_file='config.ini'
    )
    return opt


def main():
    # 获取参数
    opt = setup_params()

    print("\n=== CN数据集实验配置 ===")
    print(f"模型类型: {opt.network_type}")
    print(f"序列模型: {opt.sequence_model}")
    print(f"设备: {opt.device}")
    print(f"批大小: {opt.batch_size}")
    print(f"学习率: {opt.lr}")
    print(f"嵌入维度: {opt.embed_dim}")
    print("="*30 + "\n")

    # 初始化数据读取器
    reader = Features4QuantumReader(opt)
    opt.reader = reader

    # 设置输入维度
    opt.input_dims = reader.input_dims
    opt.total_input_dim = sum(opt.input_dims[:3])

    # 初始化模型
    model = QMamba(opt).to(opt.device)
    print(f"模型参数总量: {sum(p.numel() for p in model.parameters())}")

    # 训练模型
    print("\n开始训练...")
    train(opt, model)

    # 加载最佳模型
    best_model = QMamba(opt).to(opt.device)
    best_model.load_state_dict(torch.load(opt.best_model_file))

    # 测试模型
    print("\n开始测试...")
    test_performance = test(best_model, opt)

    # 保存模型和性能结果
    save_model(best_model, opt, test_performance)
    save_performance(opt, test_performance)

    # 删除临时文件
    if os.path.exists(opt.best_model_file):
        os.remove(opt.best_model_file)

    print("\n=== 实验完成 ===")
    print(f"结果已保存到: {opt.output_file}")
    print("="*30)


if __name__ == "__main__":
    main()
