# -*- coding: utf-8 -*-
import torch
import numpy as np
from types import SimpleNamespace
from dataset.multimodal_reader import MultiModalReader
from models.fusion_model import MultiModalMLP
from models.single_modal import ModalMLP
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import logging
# import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd


def setup_params():
    """设置参数"""
    opt = SimpleNamespace(
        pickle_dir_path='/home/tjc/audio/QMamba/feature_extract',
        features='textual,acoustic,beats,visual',
        dataset_name='features4quantum',
        embedding_enabled=False,
        dialogue_context=False,
        batch_size=32,
        num_workers=4,
        hidden_dim=256,
        num_classes=4,
        dropout=0.3,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        save_dir='problem_define',
        seed=42
    )
    return opt


def evaluate_single_modal(model, data_loader, criterion, device):
    """评估单模态模型"""
    model.eval()
    total_loss = 0
    predictions = []
    labels = []
    all_predictions = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for batch in data_loader:
            feature = batch['feature'].to(device)
            label = batch['label'].to(device)

            logits, _ = model(feature)
            logits = logits.squeeze(1)
            loss = criterion(logits, label.argmax(1))

            total_loss += loss.item()

            # 获取预测结果和置信度
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(1).cpu().numpy()
            true_labels = label.argmax(1).cpu().numpy()

            # 获取对真实标签的置信度
            true_confidences = []
            for i, true_label in enumerate(true_labels):
                true_confidences.append(probs[i, true_label].cpu().numpy())

            predictions.extend(preds)
            labels.extend(true_labels)
            all_predictions.extend(preds)
            all_labels.extend(true_labels)
            all_confidences.extend(true_confidences)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(labels, predictions)
    class_report = classification_report(
        labels, predictions, digits=4, zero_division=0)

    return {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'predictions': all_predictions,
        'labels': all_labels,
        'confidences': all_confidences
    }


def evaluate_fusion_model(model, data_loader, criterion, device):
    """评估多模态融合模型"""
    model.eval()
    total_loss = 0
    predictions = []
    labels = []
    modal_predictions = {
        'text': [],
        'audio': [],
        'video': []
    }
    fusion_features = []
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_modal_predictions = {
        'text': [],
        'audio': [],
        'video': []
    }

    with torch.no_grad():
        for batch in data_loader:
            text_x, audio_x, beats_x, video_x, label = batch
            text_x = text_x.to(device)
            audio_x = audio_x.to(device)
            video_x = video_x.to(device)
            label = label.to(device)

            logits, modal_outputs, fused_features = model(
                text_x, audio_x, video_x
            )
            logits = logits.squeeze(1)

            loss = criterion(logits, label.argmax(1))

            total_loss += loss.item()

            # 获取融合模型预测结果和置信度
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(1).cpu().numpy()
            true_labels = label.argmax(1).cpu().numpy()

            # 获取对真实标签的置信度
            true_confidences = []
            for i, true_label in enumerate(true_labels):
                true_confidences.append(probs[i, true_label].cpu().numpy())

            predictions.extend(preds)
            labels.extend(true_labels)
            all_predictions.extend(preds)
            all_labels.extend(true_labels)
            all_confidences.extend(true_confidences)

            # 保存各模态的预测结果
            for modal_name, modal_logits in modal_outputs.items():
                if modal_logits.dim() == 3:
                    modal_logits = modal_logits.squeeze(1)
                modal_preds = modal_logits.argmax(1).cpu().numpy()
                modal_predictions[modal_name].extend(modal_preds)
                all_modal_predictions[modal_name].extend(modal_preds)

            fusion_features.extend(fused_features.cpu().numpy())

    labels = np.array(labels)
    predictions = np.array(predictions)
    for modal_name in modal_predictions:
        modal_predictions[modal_name] = np.array(modal_predictions[modal_name])

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(labels, predictions)
    class_report = classification_report(
        labels, predictions, digits=4, zero_division=0
    )

    modal_metrics = {}
    for modal_name, modal_preds in modal_predictions.items():
        modal_metrics[modal_name] = {
            'accuracy': accuracy_score(labels, modal_preds),
            'f1': f1_score(labels, modal_preds, average='weighted',
                           zero_division=0),
            'conf_matrix': confusion_matrix(labels, modal_preds),
            'class_report': classification_report(
                labels, modal_preds, digits=4, zero_division=0
            )
        }

    modal_disagreement = 0
    for i in range(len(predictions)):
        modal_preds = [
            modal_predictions['text'][i],
            modal_predictions['audio'][i],
            modal_predictions['video'][i]
        ]
        if len(set(modal_preds)) > 1:
            modal_disagreement += 1

    modal_disagreement_rate = modal_disagreement / len(predictions)

    return {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'modal_metrics': modal_metrics,
        'modal_predictions': modal_predictions,
        'modal_disagreement_rate': modal_disagreement_rate,
        'fusion_features': np.array(fusion_features),
        'true_labels': np.array(labels),
        'predictions': all_predictions,
        'labels': all_labels,
        'confidences': all_confidences,
        'all_modal_predictions': all_modal_predictions
    }


def plot_confidence_scatter_from_results(results, model_type, save_dir):
    """根据预测结果绘制置信度散点图"""
    predictions = np.array(results['predictions'])
    labels = np.array(results['labels'])
    confidences = np.array(results['confidences'])

    # 计算预测准确性
    accuracies = (predictions == labels).astype(float)

    # 创建散点图
    plt.figure(figsize=(10, 8))

    # 绘制参考线
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

    # 绘制散点
    colors = ['green' if acc else 'red' for acc in accuracies]

    plt.scatter(confidences, accuracies, c=colors, alpha=0.6, s=50)

    # 添加标签和标题
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Prediction Accuracy')
    plt.title(f'{model_type.upper()} Model Confidence Analysis')

    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               label='Correct Prediction', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               label='Incorrect Prediction', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper left')

    # 添加象限标注
    plt.text(0.25, 0.75, 'Low Confidence\nHigh Accuracy', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    plt.text(0.75, 0.75, 'High Confidence\nHigh Accuracy', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    plt.text(0.25, 0.25, 'Low Confidence\nLow Accuracy', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    plt.text(0.75, 0.25, 'High Confidence\nLow Accuracy', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

    # 保存图片
    save_path = f"{save_dir}/{model_type}_confidence_scatter.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return save_path


def plot_fusion_confidence_scatter_from_results(results, save_dir):
    """根据融合模型预测结果绘制置信度散点图（包含模态一致性）"""
    predictions = np.array(results['predictions'])
    labels = np.array(results['labels'])
    confidences = np.array(results['confidences'])
    modal_predictions = results['all_modal_predictions']

    # 计算预测准确性
    accuracies = (predictions == labels).astype(float)

    # 计算模态一致性
    text_preds = np.array(modal_predictions['text'])
    audio_preds = np.array(modal_predictions['audio'])
    video_preds = np.array(modal_predictions['video'])

    modal_agreements = ((text_preds == audio_preds) & (
        audio_preds == video_preds)).astype(float)

    # 创建散点图
    plt.figure(figsize=(12, 10))

    # 绘制参考线
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

    # 绘制散点（根据准确性和模态一致性使用不同的标记）
    for i in range(len(confidences)):
        if accuracies[i] == 1:  # 正确预测
            if modal_agreements[i] == 1:  # 模态一致
                plt.scatter(confidences[i], accuracies[i], c='green', marker='o',
                            s=100, alpha=0.7, label='Correct+Modal Agree' if i == 0 else "")
            else:  # 模态不一致
                plt.scatter(confidences[i], accuracies[i], c='green', marker='^',
                            s=100, alpha=0.7, label='Correct+Modal Disagree' if i == 0 else "")
        else:  # 错误预测
            if modal_agreements[i] == 1:  # 模态一致
                plt.scatter(confidences[i], accuracies[i], c='red', marker='o',
                            s=100, alpha=0.7, label='Incorrect+Modal Agree' if i == 0 else "")
            else:  # 模态不一致
                plt.scatter(confidences[i], accuracies[i], c='red', marker='^',
                            s=100, alpha=0.7, label='Incorrect+Modal Disagree' if i == 0 else "")

        # 添加标签和标题
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Prediction Accuracy')
    plt.title('Multimodal Fusion Model Confidence Analysis')

    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               label='Correct+Modal Agree', markersize=10),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green',
               label='Correct+Modal Disagree', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               label='Incorrect+Modal Agree', markersize=10),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
               label='Incorrect+Modal Disagree', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper left')

    # 添加象限标注
    plt.text(0.25, 0.75, 'Low Confidence\nHigh Accuracy', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    plt.text(0.75, 0.75, 'High Confidence\nHigh Accuracy', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    plt.text(0.25, 0.25, 'Low Confidence\nLow Accuracy', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    plt.text(0.75, 0.25, 'High Confidence\nLow Accuracy', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

    # 保存图片
    save_path = f"{save_dir}/fusion_confidence_scatter.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return save_path


# def plot_confusion_matrix(conf_matrix, save_path):
#     """绘制混淆矩阵"""
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
#     plt.title('Confusion Matrix')
#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()


def save_predictions_to_csv(results, model_type, save_dir):
    """保存预测结果到CSV文件"""
    df = pd.DataFrame({
        'true_label': results['labels'],
        'predicted_label': results['predictions'],
        'confidence': results['confidences']
    })

    # 添加样本ID
    df['sample_id'] = range(len(df))

    # 重新排列列顺序
    df = df[['sample_id', 'true_label', 'predicted_label', 'confidence']]

    # 保存到CSV
    csv_path = f"{save_dir}/{model_type}_predictions.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


def save_fusion_predictions_to_csv(results, save_dir):
    """保存融合模型预测结果到CSV文件"""
    df = pd.DataFrame({
        'sample_id': range(len(results['labels'])),
        'true_label': results['labels'],
        'fusion_predicted_label': results['predictions'],
        'fusion_confidence': results['confidences'],
        'text_predicted_label': results['all_modal_predictions']['text'],
        'audio_predicted_label': results['all_modal_predictions']['audio'],
        'video_predicted_label': results['all_modal_predictions']['video']
    })

    # 保存到CSV
    csv_path = f"{save_dir}/fusion_all_predictions.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


def test_single_modal_model(model_path, model, test_loader, opt, model_type):
    """测试单模态模型"""
    logger = logging.getLogger(__name__)
    logger.info(f"\n=== Testing {model_type.upper()} Model ===")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=opt.device))
        logger.info(
            f"Successfully loaded {model_type} model weights: {model_path}")
    else:
        logger.warning(f"Model file does not exist: {model_path}")
        return None

    criterion = nn.CrossEntropyLoss()
    results = evaluate_single_modal(model, test_loader, criterion, opt.device)

    logger.info(f"Loss: {results['loss']:.4f}")
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    logger.info(f"F1 Score: {results['f1']:.4f}")
    logger.info(f"\n分类报告:\n{results['classification_report']}")

    # 保存混淆矩阵
    # plot_confusion_matrix(
    #     results['confusion_matrix'],
    #     f"{opt.save_dir}/{model_type}_test_confusion_matrix.png"
    # )

    # 保存预测结果
    csv_path = save_predictions_to_csv(results, model_type, opt.save_dir)
    logger.info("Prediction results saved to: %s", csv_path)

    # 绘制置信度散点图
    scatter_path = plot_confidence_scatter_from_results(
        results, model_type, opt.save_dir
    )
    logger.info("Confidence scatter plot saved to: %s", scatter_path)

    return results


def test_fusion_model(model_path, model, test_loader, opt):
    """测试多模态融合模型"""
    logger = logging.getLogger(__name__)
    logger.info("\n=== Testing Multimodal Fusion Model ===")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=opt.device))
        logger.info(f"Successfully loaded fusion model weights: {model_path}")
    else:
        logger.warning(f"Model file does not exist: {model_path}")
        return None

    criterion = nn.CrossEntropyLoss()
    results = evaluate_fusion_model(model, test_loader, criterion, opt.device)

    logger.info(f"Loss: {results['loss']:.4f}")
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    logger.info(f"F1 Score: {results['f1']:.4f}")
    logger.info(f"\n分类报告:\n{results['classification_report']}")

    logger.info("\nModal Performance:")
    for modal_name, metrics in results['modal_metrics'].items():
        logger.info(f"\n{modal_name.upper()}:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info(f"\n分类报告:\n{metrics['class_report']}")

    logger.info("\nModal Disagreement Analysis:")
    logger.info(
        f"Modal Disagreement Rate: {results['modal_disagreement_rate']:.4f}")

    # 保存混淆矩阵
    # plot_confusion_matrix(
    #     results['confusion_matrix'],
    #     f"{opt.save_dir}/fusion_test_confusion_matrix.png"
    # )

    # for modal_name, metrics in results['modal_metrics'].items():
    #     plot_confusion_matrix(
    #         metrics['conf_matrix'],
    #         f"{opt.save_dir}/fusion_{modal_name}_test_confusion_matrix.png"
    #     )

    # 保存预测结果
    csv_path = save_fusion_predictions_to_csv(results, opt.save_dir)
    logger.info("Fusion model prediction results saved to: %s", csv_path)

    # 绘制置信度散点图
    scatter_path = plot_fusion_confidence_scatter_from_results(
        results, opt.save_dir
    )
    logger.info(
        "Fusion model confidence scatter plot saved to: %s", scatter_path)

    return results


def plot_model_comparison_scatter(text_results, audio_results, video_results,
                                  fusion_results, save_dir):
    """绘制单模态模型与多模态融合模型的比较散点图"""

    # 创建三个子图
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Unimodal vs Multimodal Fusion Model Performance Comparison',
                 fontsize=16, y=0.95)

    # 定义颜色映射
    colors = {
        'both_correct': 'blue',      # 两个模型都正确
        'unimodal_correct': 'red',   # 单模态正确，多模态错误
        'fusion_correct': 'green',   # 单模态错误，多模态正确
        'both_wrong': 'yellow'       # 两个模型都错误
    }

    # 获取真实标签
    true_labels = np.array(fusion_results['labels'])

    # 获取融合模型对真实标签的置信度
    fusion_confidences = np.array(fusion_results['confidences'])

    # 需要重新计算融合模型对真实标签的置信度
    # 这里需要从模型输出中获取对真实标签的概率
    fusion_true_confidences = []
    for i, true_label in enumerate(true_labels):
        # 这里需要从fusion_results中获取对真实标签的置信度
        # 暂时使用现有的置信度，但实际应该是对真实标签的置信度
        fusion_true_confidences.append(fusion_confidences[i])

    # 处理每个单模态模型
    modal_results = [
        ('Text', text_results, axes[0, 0]),
        ('Audio', audio_results, axes[0, 1]),
        ('Video', video_results, axes[1, 0])
    ]

    for modal_name, modal_result, ax in modal_results:
        if modal_result is None:
            continue

        # 获取单模态模型对真实标签的置信度
        modal_confidences = np.array(modal_result['confidences'])

        # 计算象限和颜色
        quadrant_colors = []
        for i in range(len(true_labels)):
            unimodal_conf = modal_confidences[i]
            fusion_conf = fusion_true_confidences[i]

            if unimodal_conf > 0.5 and fusion_conf > 0.5:
                quadrant_colors.append(colors['both_correct'])
            elif unimodal_conf > 0.5 and fusion_conf <= 0.5:
                quadrant_colors.append(colors['unimodal_correct'])
            elif unimodal_conf <= 0.5 and fusion_conf > 0.5:
                quadrant_colors.append(colors['fusion_correct'])
            else:
                quadrant_colors.append(colors['both_wrong'])

        # 绘制散点图
        scatter = ax.scatter(modal_confidences, fusion_true_confidences,
                             c=quadrant_colors, alpha=0.6, s=50)

        # 添加参考线
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')

        # 设置标签和标题
        ax.set_xlabel(f'{modal_name} Model Confidence')
        ax.set_ylabel('Fusion Model Confidence')
        ax.set_title(f'{modal_name} vs Fusion Model')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                   label='Both Correct', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   label='Unimodal Correct', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                   label='Fusion Correct', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow',
                   label='Both Wrong', markersize=10)
        ]
        ax.legend(handles=legend_elements, loc='upper left')

        # 添加象限标注
        ax.text(0.75, 0.75, 'Both Correct', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax.text(0.75, 0.25, 'Unimodal Correct', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        ax.text(0.25, 0.75, 'Fusion Correct', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        ax.text(0.25, 0.25, 'Both Wrong', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

    # 隐藏第四个子图
    axes[1, 1].set_visible(False)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    save_path = f"{save_dir}/model_comparison_scatter.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return save_path


def plot_model_comparison_with_marginals(text_results, audio_results, video_results,
                                         fusion_results, save_dir):
    """绘制带边际分布的模型比较散点图"""

    # 创建三个独立的图，每个包含散点图和边际分布
    modal_results = [
        ('Text', text_results),
        ('Audio', audio_results),
        ('Video', video_results)
    ]

    for modal_name, modal_result in modal_results:
        if modal_result is None:
            continue

        # 创建带边际分布的图
        fig = plt.figure(figsize=(12, 10))

        # 定义网格
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 主散点图
        ax_main = fig.add_subplot(gs[1:, :-1])

        # 顶部直方图（单模态置信度分布）
        ax_hist_x = fig.add_subplot(gs[0, :-1], sharex=ax_main)

        # 右侧直方图（融合模型置信度分布）
        ax_hist_y = fig.add_subplot(gs[1:, -1], sharey=ax_main)

        # 获取数据
        true_labels = np.array(fusion_results['labels'])
        fusion_confidences = np.array(fusion_results['confidences'])
        modal_confidences = np.array(modal_result['confidences'])

        # 计算象限和颜色
        colors = {
            'both_correct': 'blue',
            'unimodal_correct': 'red',
            'fusion_correct': 'green',
            'both_wrong': 'yellow'
        }

        quadrant_colors = []
        for i in range(len(true_labels)):
            unimodal_conf = modal_confidences[i]
            fusion_conf = fusion_confidences[i]

            if unimodal_conf > 0.5 and fusion_conf > 0.5:
                quadrant_colors.append(colors['both_correct'])
            elif unimodal_conf > 0.5 and fusion_conf <= 0.5:
                quadrant_colors.append(colors['unimodal_correct'])
            elif unimodal_conf <= 0.5 and fusion_conf > 0.5:
                quadrant_colors.append(colors['fusion_correct'])
            else:
                quadrant_colors.append(colors['both_wrong'])

        # 绘制主散点图
        scatter = ax_main.scatter(modal_confidences, fusion_confidences,
                                  c=quadrant_colors, alpha=0.6, s=50)

        # 添加参考线
        ax_main.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax_main.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        ax_main.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')

        # 设置主图标签
        ax_main.set_xlabel(f'{modal_name} Model Confidence')
        ax_main.set_ylabel('Fusion Model Confidence')
        ax_main.set_title(f'{modal_name} vs Fusion Model Comparison')
        ax_main.set_xlim(0, 1)
        ax_main.set_ylim(0, 1)
        ax_main.grid(True, alpha=0.3)

        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                   label='Both Correct', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   label='Unimodal Correct', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                   label='Fusion Correct', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow',
                   label='Both Wrong', markersize=10)
        ]
        ax_main.legend(handles=legend_elements, loc='upper left')

        # 绘制顶部直方图（单模态置信度分布）
        ax_hist_x.hist(modal_confidences, bins=30, alpha=0.7,
                       color='skyblue', edgecolor='black')
        ax_hist_x.set_ylabel('Frequency')
        ax_hist_x.set_title(f'{modal_name} Model Confidence Distribution')
        ax_hist_x.axvline(x=0.5, color='red', linestyle='--', alpha=0.7)

        # 绘制右侧直方图（融合模型置信度分布）
        ax_hist_y.hist(fusion_confidences, bins=30, alpha=0.7,
                       color='lightgreen', edgecolor='black', orientation='horizontal')
        ax_hist_y.set_xlabel('Frequency')
        ax_hist_y.set_title('Fusion Model Confidence Distribution')
        ax_hist_y.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)

        # 隐藏顶部和右侧直方图的刻度标签
        ax_hist_x.set_xticklabels([])
        ax_hist_y.set_yticklabels([])

        # 保存图片
        save_path = f"{save_dir}/{modal_name.lower()}_comparison_with_marginals.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return f"{save_dir}/*_comparison_with_marginals.png"


def main():
    opt = setup_params()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting model testing...")
    logger.info(f"Model save directory: {opt.save_dir}")
    logger.info(f"Device: {opt.device}")

    logger.info("\nLoading data...")
    reader = MultiModalReader(opt)
    opt.input_dims = reader.input_dims

    text_test_loader = reader.text_test_loader
    audio_test_loader = reader.audio_test_loader
    video_test_loader = reader.video_test_loader
    fusion_test_loader = reader.test_loader

    logger.info("\nInitializing models...")
    text_model = ModalMLP(
        opt.input_dims[0], opt.hidden_dim, opt.num_classes, opt.dropout
    ).to(opt.device)
    audio_model = ModalMLP(
        opt.input_dims[1], opt.hidden_dim, opt.num_classes, opt.dropout
    ).to(opt.device)
    video_model = ModalMLP(
        opt.input_dims[3], opt.hidden_dim, opt.num_classes, opt.dropout
    ).to(opt.device)
    fusion_model = MultiModalMLP(opt).to(opt.device)

    logger.info("\nStarting single modal model testing...")

    text_model_path = f"{opt.save_dir}/text_best_model.pth"
    text_results = test_single_modal_model(
        text_model_path, text_model, text_test_loader, opt, 'text'
    )

    audio_model_path = f"{opt.save_dir}/audio_best_model.pth"
    audio_results = test_single_modal_model(
        audio_model_path, audio_model, audio_test_loader, opt, 'audio'
    )

    video_model_path = f"{opt.save_dir}/video_best_model.pth"
    video_results = test_single_modal_model(
        video_model_path, video_model, video_test_loader, opt, 'video'
    )

    logger.info("\nStarting multimodal fusion model testing...")
    fusion_model_path = f"{opt.save_dir}/fusion_best_model.pth"
    fusion_results = test_fusion_model(
        fusion_model_path, fusion_model, fusion_test_loader, opt
    )

    logger.info("\n" + "="*50)
    logger.info("Test Results Summary")
    logger.info("="*50)

    if text_results:
        logger.info(f"Text Model - Accuracy: {text_results['accuracy']:.4f}, "
                    f"F1: {text_results['f1']:.4f}")
    if audio_results:
        logger.info(f"Audio Model - Accuracy: {audio_results['accuracy']:.4f}, "
                    f"F1: {audio_results['f1']:.4f}")
    if video_results:
        logger.info(f"Video Model - Accuracy: {video_results['accuracy']:.4f}, "
                    f"F1: {video_results['f1']:.4f}")
    if fusion_results:
        logger.info(f"Fusion Model - Accuracy: {fusion_results['accuracy']:.4f}, "
                    f"F1: {fusion_results['f1']:.4f}")

    logger.info("\nTesting completed! All results saved to corresponding files.")

    # 绘制单模态模型与多模态融合模型的比较散点图
    if all([text_results, audio_results, video_results, fusion_results]):
        scatter_path = plot_model_comparison_scatter(
            text_results, audio_results, video_results, fusion_results, opt.save_dir
        )
        logger.info(
            "Unimodal vs multimodal fusion model comparison scatter plot saved to: %s", scatter_path)

        # 绘制带边际分布的模型比较散点图
        marginals_path = plot_model_comparison_with_marginals(
            text_results, audio_results, video_results, fusion_results, opt.save_dir
        )
        logger.info(
            "Model comparison scatter plot with marginal distributions saved to: %s", marginals_path)
    else:
        logger.warning(
            "Some model results are missing, skipping comparison plots")


if __name__ == "__main__":
    main()
