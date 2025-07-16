# -*- coding: utf-8 -*-
import torch
import numpy as np
from types import SimpleNamespace
from dataset.multimodal_reader import MultiModalReader, SingleModalDataset
from models.fusion_model import MultiModalFusion, MultiModalMLP
from models.single_modal import TextModel, AudioModel, VideoModel, ModalMLP
from utils.model import train, test, save_model, save_performance
from utils.visualization import plot_confidence_scatter
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import logging
import seaborn as sns
import matplotlib.pyplot as plt


def setup_params():
    """设置参数"""
    opt = SimpleNamespace(
        # 数据相关
        pickle_dir_path='/home/tjc/audio/QMamba/feature_extract',  # 使用绝对路径
        features='textual,acoustic,beats,visual',
        dataset_name='features4quantum',
        embedding_enabled=False,
        dialogue_context=False,
        batch_size=32,
        num_workers=4,
        # 模型参数
        hidden_dim=256,
        num_classes=4,
        dropout=0.3,

        # 训练参数
        epochs=50,
        lr=1e-4,
        weight_decay=1e-4,
        patience=5,

        # 其他设置
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        save_dir='problem_define',
        seed=42
    )
    return opt


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def train_single_modal(model, train_loader, val_loader, opt, model_type='text'):
    """训练单模态模型

    Args:
        model: 单模态模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        opt: 配置参数对象
        model_type: 模型类型('text', 'audio', 'video')
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始训练{model_type}模型...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr,
                           weight_decay=opt.weight_decay)

    best_val_f1 = 0
    patience_counter = 0

    for epoch in range(opt.epochs):
        # 训练
        model.train()
        total_loss = 0
        predictions = []
        labels = []

        for batch in train_loader:
            # 获取数据
            feature = batch['feature'].to(opt.device)
            label = batch['label'].to(opt.device)

            # 前向传播
            logits, _ = model(feature)
            logits = logits.squeeze(1)
            # print("logits", logits)
            # print("label", label.argmax(1))
            # 计算损失
            loss = criterion(logits, label.argmax(1))

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions.extend(logits.argmax(1).cpu().numpy())
            labels.extend(label.argmax(1).cpu().numpy())

        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(labels, predictions)
        train_f1 = f1_score(labels, predictions, average='weighted')

        # 验证
        val_results = evaluate_single_modal(
            model, val_loader, criterion, opt.device)
        val_loss = val_results['loss']
        val_acc = val_results['accuracy']
        val_f1 = val_results['f1']

        logger.info(
            f"Epoch {epoch+1}/{opt.epochs} - {model_type.upper()} - "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
            f"Train F1: {train_f1:.4f} - Val Loss: {val_loss:.4f} - "
            f"Val Acc: {val_acc:.4f} - Val F1: {val_f1:.4f}"
        )

        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_path = f"{opt.save_dir}/{model_type}_best_model.pth"
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= opt.patience:
            logger.info(f"{model_type.upper()}模型早停触发!")
            break

    logger.info(f"{model_type.upper()}模型训练完成!")
    return model


def train_fusion_model(model, train_loader, val_loader, opt):
    """训练多模态融合模型

    Args:
        model: 多模态融合模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        opt: 配置参数对象
    """
    logger = logging.getLogger(__name__)
    logger.info("开始训练多模态融合模型...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr,
                           weight_decay=opt.weight_decay)

    best_val_f1 = 0
    patience_counter = 0

    for epoch in range(opt.epochs):
        # 训练
        model.train()
        total_loss = 0
        predictions = []
        labels = []

        for batch in train_loader:
            # 获取数据
            text_x, audio_x, beats_x, video_x, label = batch
            text_x = text_x.to(opt.device)
            audio_x = audio_x.to(opt.device)
            beats_x = beats_x.to(opt.device)
            video_x = video_x.to(opt.device)
            label = label.to(opt.device)

            # 前向传播
            logits, modal_outputs, _ = model(text_x, audio_x, video_x)
            logits = logits.squeeze(1)
            # 计算损失(包括单模态损失)
            loss = criterion(logits, label.argmax(1))
            for modal_logits in modal_outputs.values():
                if modal_logits.dim() == 3:
                    modal_logits = modal_logits.squeeze(1)
                loss += 0.1 * criterion(modal_logits, label.argmax(1))

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions.extend(logits.argmax(1).cpu().numpy())
            labels.extend(label.argmax(1).cpu().numpy())

        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(labels, predictions)
        train_f1 = f1_score(labels, predictions, average='weighted')

        # 验证
        val_results = evaluate_fusion_model(
            model, val_loader, criterion, opt.device)
        val_loss = val_results['loss']
        val_acc = val_results['accuracy']
        val_f1 = val_results['f1']

        logger.info(
            f"Epoch {epoch+1}/{opt.epochs} - FUSION - "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
            f"Train F1: {train_f1:.4f} - Val Loss: {val_loss:.4f} - "
            f"Val Acc: {val_acc:.4f} - Val F1: {val_f1:.4f}"
        )

        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_path = f"{opt.save_dir}/fusion_best_model.pth"
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= opt.patience:
            logger.info("多模态融合模型早停触发!")
            break

    logger.info("多模态融合模型训练完成!")
    return model


def evaluate_single_modal(model, data_loader, criterion, device):
    """评估单模态模型

    Args:
        model: 单模态模型
        data_loader: 数据加载器
        criterion: 损失函数
        device: 设备

    Returns:
        results: 评估结果
    """
    model.eval()
    total_loss = 0
    predictions = []
    labels = []

    with torch.no_grad():
        for batch in data_loader:
            # 获取数据
            feature = batch['feature'].to(device)
            label = batch['label'].to(device)

            # 前向传播
            logits, _ = model(feature)
            logits = logits.squeeze(1)
            # 计算损失
            loss = criterion(logits, label.argmax(1))

            total_loss += loss.item()
            predictions.extend(logits.argmax(1).cpu().numpy())
            labels.extend(label.argmax(1).cpu().numpy())

    # 计算指标
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
        'classification_report': class_report
    }

    """评估多模态融合模型

    Args:
        model: 多模态融合模型
        data_loader: 数据加载器
        criterion: 损失函数
        device: 设备

    Returns:
        results: 评估结果
    """
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

    with torch.no_grad():
        for batch in data_loader:
            # 获取数据
            text_x, audio_x, beats_x, video_x, label = batch
            text_x = text_x.to(device)
            audio_x = audio_x.to(device)
            video_x = video_x.to(device)
            label = label.to(device)

            # 前向传播
            logits, modal_outputs, fused_features = model(
                text_x, audio_x, video_x
            )
            logits = logits.squeeze(1)
            # 计算损失
            loss = criterion(logits, label.argmax(1))

            total_loss += loss.item()
            predictions.extend(logits.argmax(1).cpu().numpy())
            labels.extend(label.argmax(1).cpu().numpy())

            # 保存各模态的预测结果
            for modal_name, modal_logits in modal_outputs.items():
                modal_predictions[modal_name].extend(
                    modal_logits.argmax(1).cpu().numpy()
                )

            # 保存融合特征
            fusion_features.extend(fused_features.cpu().numpy())

    # 确保所有数据都是numpy数组且格式一致
    print(type(labels))
    print(type(predictions))
    print("真实标签示例:", labels[:5])
    print("预测标签示例:", predictions[:5])
    # 计算总体指标
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(labels, predictions)
    class_report = classification_report(
        labels, predictions, digits=4, zero_division=0
    )

    # 计算各模态的指标
    modal_metrics = {}
    for modal_name, modal_preds in modal_predictions.items():
        # 将numpy数组转换为列表，然后extend
        modal_predictions[modal_name].extend(
            modal_preds.argmax(1).cpu().numpy().tolist()
        )
        modal_metrics[modal_name] = {
            'accuracy': accuracy_score(labels, modal_preds),
            'f1': f1_score(labels, modal_preds, average='weighted', zero_division=0),
            'conf_matrix': confusion_matrix(labels, modal_preds),
            'class_report': classification_report(
                labels, modal_preds, digits=4, zero_division=0
            )
        }

    # 计算模态分歧
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
        'true_labels': np.array(labels)
    }


def evaluate_fusion_model(model, data_loader, criterion, device):
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

    with torch.no_grad():
        for batch in data_loader:
            # 获取数据
            text_x, audio_x, beats_x, video_x, label = batch
            text_x = text_x.to(device)
            audio_x = audio_x.to(device)
            video_x = video_x.to(device)
            label = label.to(device)

            # 前向传播
            logits, modal_outputs, fused_features = model(
                text_x, audio_x, video_x
            )
            logits = logits.squeeze(1)

            # 计算损失
            loss = criterion(logits, label.argmax(1))

            total_loss += loss.item()
            predictions.extend(logits.argmax(1).cpu().numpy())
            labels.extend(label.argmax(1).cpu().numpy())

            # 保存各模态的预测结果
            for modal_name, modal_logits in modal_outputs.items():
                # 确保modal_logits的维度正确
                if modal_logits.dim() == 3:
                    modal_logits = modal_logits.squeeze(1)
                modal_predictions[modal_name].extend(
                    modal_logits.argmax(1).cpu().numpy()
                )

            # 保存融合特征
            fusion_features.extend(fused_features.cpu().numpy())

    # 确保所有数据都是numpy数组且格式一致
    labels = np.array(labels)
    predictions = np.array(predictions)
    for modal_name in modal_predictions:
        modal_predictions[modal_name] = np.array(modal_predictions[modal_name])

    # 计算总体指标
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(labels, predictions)
    class_report = classification_report(
        labels, predictions, digits=4, zero_division=0
    )

    # 计算各模态的指标
    modal_metrics = {}
    for modal_name, modal_preds in modal_predictions.items():
        modal_metrics[modal_name] = {
            'accuracy': accuracy_score(labels, modal_preds),
            'f1': f1_score(labels, modal_preds, average='weighted', zero_division=0),
            'conf_matrix': confusion_matrix(labels, modal_preds),
            'class_report': classification_report(
                labels, modal_preds, digits=4, zero_division=0
            )
        }

    # 计算模态分歧
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
        'true_labels': np.array(labels)
    }


def plot_confusion_matrix(conf_matrix, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def test_model(model_path, opt):
    """测试模型性能

    Args:
        model_path: 模型权重路径
        opt: 配置参数对象
    """
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # 加载数据
    logger.info("\n加载测试数据...")
    reader = MultiModalReader(opt)
    test_loader = DataLoader(
        reader.test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers
    )

    # 加载模型
    logger.info("\n加载模型...")
    model = MultiModalFusion(opt).to(opt.device)
    model.load_state_dict(torch.load(model_path))
    criterion = nn.CrossEntropyLoss()

    # 测试模型
    logger.info("\n开始测试...")
    results = evaluate(model, test_loader, criterion, opt.device)

    # 输出测试报告
    logger.info("\n=== 测试报告 ===")
    logger.info(f"\n1. 总体性能:")
    logger.info(f"Loss: {results['loss']:.4f}")
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    logger.info(f"F1 Score: {results['f1']:.4f}")
    logger.info(f"\n分类报告:\n{results['classification_report']}")

    logger.info(f"\n2. 各模态性能:")
    for modal_name, metrics in results['modal_metrics'].items():
        logger.info(f"\n{modal_name.upper()}:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info(f"\n分类报告:\n{metrics['class_report']}")

    logger.info(f"\n3. 模态分歧分析:")
    logger.info(f"模态分歧率: {results['modal_disagreement_rate']:.4f}")

    # 保存可视化结果
    logger.info("\n保存可视化结果...")

    # 绘制总体混淆矩阵
    plot_confusion_matrix(
        results['confusion_matrix'],
        f"{opt.save_dir}/confusion_matrix.png"
    )

    # 绘制各模态混淆矩阵
    for modal_name, metrics in results['modal_metrics'].items():
        plot_confusion_matrix(
            metrics['conf_matrix'],
            f"{opt.save_dir}/{modal_name}_confusion_matrix.png"
        )

    # 绘制置信度散点图
    plot_confidence_scatter(
        model,
        test_loader,
        opt.device,
        f"{opt.save_dir}/confidence_scatter.png"
    )

    logger.info("\n=== 测试完成 ===")
    logger.info(f"结果已保存到: {opt.save_dir}")

    return results


def main():
    # 获取参数
    opt = setup_params()

    # 设置随机种子
    set_seed(opt.seed)

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("实验配置:")
    logger.info("批大小: {}".format(opt.batch_size))
    logger.info("学习率: {}".format(opt.lr))
    logger.info("隐藏层维度: {}".format(opt.hidden_dim))
    logger.info("设备: {}".format(opt.device))
    # 加载数据
    logger.info("加载数据...")
    reader = MultiModalReader(opt)
    opt.input_dims = reader.input_dims
    # 获取多模态数据加载器
    train_loader = reader.train_loader
    val_loader = reader.dev_loader
    test_loader = reader.test_loader

    # 获取单模态数据加载器
    text_train_loader = reader.text_train_loader
    text_val_loader = reader.text_dev_loader
    text_test_loader = reader.text_test_loader

    audio_train_loader = reader.audio_train_loader
    audio_val_loader = reader.audio_dev_loader
    audio_test_loader = reader.audio_test_loader

    video_train_loader = reader.video_train_loader
    video_val_loader = reader.video_dev_loader
    video_test_loader = reader.video_test_loader
    # 初始化模型
    logger.info("初始化模型...")
    text_model = ModalMLP(
        opt.input_dims[0], opt.hidden_dim, opt.num_classes, opt.dropout).to(opt.device)
    audio_model = ModalMLP(
        opt.input_dims[1], opt.hidden_dim, opt.num_classes, opt.dropout).to(opt.device)
    video_model = ModalMLP(
        opt.input_dims[3], opt.hidden_dim, opt.num_classes, opt.dropout).to(opt.device)  # 使用索引3

    # 初始化多模态融合模型
    fusion_model = MultiModalMLP(opt).to(opt.device)

    print(text_model)
    print(audio_model)
    print(video_model)
    print(fusion_model)
    # 训练单模态模型
    logger.info("开始训练单模态模型...")
    text_model = train_single_modal(
        text_model, text_train_loader, text_val_loader, opt, 'text')
    audio_model = train_single_modal(
        audio_model, audio_train_loader, audio_val_loader, opt, 'audio')
    video_model = train_single_modal(
        video_model, video_train_loader, video_val_loader, opt, 'video')

    # 将训练好的单模态模型权重复制到融合模型中
    fusion_model.text_mlp.load_state_dict(text_model.state_dict())
    fusion_model.audio_mlp.load_state_dict(audio_model.state_dict())
    fusion_model.video_mlp.load_state_dict(video_model.state_dict())

    # 训练融合模型
    logger.info("开始训练多模态融合模型...")
    fusion_model = train_fusion_model(
        fusion_model, train_loader, val_loader, opt)

    # 测试模型
    logger.info("开始测试...")
    criterion = nn.CrossEntropyLoss()

    # 测试单模态模型
    text_results = evaluate_single_modal(
        text_model, text_test_loader, criterion, opt.device)
    audio_results = evaluate_single_modal(
        audio_model, audio_test_loader, criterion, opt.device)
    video_results = evaluate_single_modal(
        video_model, video_test_loader, criterion, opt.device)

    # 测试融合模型
    fusion_results = evaluate_fusion_model(
        fusion_model, test_loader, criterion, opt.device)

    # 输出测试结果
    logger.info("\n=== 测试结果 ===")
    logger.info("\n1. 单模态模型性能:")
    logger.info(
        f"文本模型 - 准确率: {text_results['accuracy']:.4f}, F1: {text_results['f1']:.4f}")
    logger.info(
        f"音频模型 - 准确率: {audio_results['accuracy']:.4f}, F1: {audio_results['f1']:.4f}")
    logger.info(
        f"视频模型 - 准确率: {video_results['accuracy']:.4f}, F1: {video_results['f1']:.4f}")

    logger.info("\n2. 多模态融合模型性能:")
    logger.info(
        f"融合模型 - 准确率: {fusion_results['accuracy']:.4f}, F1: {fusion_results['f1']:.4f}")

    logger.info("实验完成")


if __name__ == "__main__":
    main()
