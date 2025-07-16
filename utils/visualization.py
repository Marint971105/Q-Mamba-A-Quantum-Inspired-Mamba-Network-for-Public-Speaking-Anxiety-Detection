# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from sklearn.metrics import accuracy_score


def plot_confidence_scatter(model, data_loader, device, save_path):
    """绘制置信度散点图

    Args:
        model: 多模态模型
        data_loader: 数据加载器
        device: 设备
        save_path: 保存路径
    """
    model.eval()
    confidences = []
    accuracies = []
    modal_agreements = []

    with torch.no_grad():
        for batch in data_loader:
            # 获取数据
            text_x = batch['text'].to(device)
            audio_x = batch['audio'].to(device)
            video_x = batch['video'].to(device)
            y = batch['label'].to(device)

            # 前向传播
            logits, modal_outputs, _ = model(text_x, audio_x, video_x)

            # 计算总体预测的置信度
            probs = torch.softmax(logits, dim=1)
            confidence = probs.max(1)[0].cpu().numpy()

            # 计算预测准确性
            preds = logits.argmax(1).cpu().numpy()
            y = y.cpu().numpy()
            accuracy = (preds == y).astype(float)

            # 计算模态一致性
            modal_preds = []
            for modal_logits in modal_outputs.values():
                modal_preds.append(modal_logits.argmax(1).cpu().numpy())
            modal_preds = np.array(modal_preds)
            agreement = (modal_preds[0] == modal_preds[1]) & (
                modal_preds[1] == modal_preds[2])

            confidences.extend(confidence)
            accuracies.extend(accuracy)
            modal_agreements.extend(agreement)

    # 转换为numpy数组
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    modal_agreements = np.array(modal_agreements)

    # 创建散点图
    plt.figure(figsize=(10, 8))

    # 绘制四个象限
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

    # 绘制散点
    colors = ['red' if not acc else 'green' for acc in accuracies]
    markers = ['o' if agree else 'x' for agree in modal_agreements]

    for i in range(len(confidences)):
        plt.scatter(
            confidences[i],
            accuracies[i],
            c=colors[i],
            marker=markers[i],
            alpha=0.6
        )

    # 添加标签和标题
    plt.xlabel('预测置信度')
    plt.ylabel('预测准确性')
    plt.title('多模态预测置信度分析')

    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               label='正确预测', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               label='错误预测', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               label='模态一致', markersize=10),
        Line2D([0], [0], marker='x', color='gray',
               label='模态不一致', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper left')

    # 添加象限标注
    plt.text(0.25, 0.75, '低置信度\n高准确率', ha='center', va='center')
    plt.text(0.75, 0.75, '高置信度\n高准确率', ha='center', va='center')
    plt.text(0.25, 0.25, '低置信度\n低准确率', ha='center', va='center')
    plt.text(0.75, 0.25, '高置信度\n低准确率', ha='center', va='center')

    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
