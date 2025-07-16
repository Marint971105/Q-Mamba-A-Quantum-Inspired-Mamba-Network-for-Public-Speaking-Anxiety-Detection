# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import copy
from utils.evaluation import evaluate
import time
import pickle
from optimizer import RMSprop_Unitary
from models.DialogueRNN import MaskedNLLLoss
import torch.nn.functional as F


def train(params, model):
    criterion = get_criterion(params)
    if hasattr(model, 'get_params'):
        unitary_params, remaining_params = model.get_params()
    else:
        remaining_params = model.parameters()
        unitary_params = []

    if len(unitary_params) > 0:
        unitary_optimizer = RMSprop_Unitary(
            unitary_params, lr=params.unitary_lr)
        unitary_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            unitary_optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

    # 修改优化器为Adam并添加权重衰减
    optimizer = torch.optim.Adam(
        remaining_params, lr=params.lr, weight_decay=1e-5)

    # 降低学习率
    params.lr = 0.0005

    # 增加梯度裁剪阈值
    params.clip = 1.0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    early_stopping = EarlyStopping(patience=params.patience, verbose=True)

    temp_file_name = str(int(np.random.rand()*int(time.time())))
    params.best_model_file = os.path.join('tmp', temp_file_name)

    best_val_loss = float('inf')

    for epoch in range(params.epochs):
        print(f'Epoch: {epoch}')
        model.train()
        train_losses = []

        with tqdm(total=params.reader.train_sample_num) as pbar:
            time.sleep(0.05)
            for _i, data in enumerate(params.reader.get_data(iterable=True, shuffle=True, split='train'), 0):
                b_inputs = [inp.to(params.device) for inp in data[:-1]]
                b_targets = data[-1].to(params.device)

                if b_inputs[0].shape[0] == 1:
                    continue

                optimizer.zero_grad()
                if len(unitary_params) > 0:
                    unitary_optimizer.zero_grad()

                outputs = model(b_inputs)

                # 添加梯度检查
                torch.nn.utils.clip_grad_norm_(model.parameters(), params.clip)

                # 检查输出是否包含NaN
                if torch.isnan(outputs).any():
                    print("Warning: NaN in model output")
                    continue

                b_targets, outputs, loss = get_loss(
                    params, criterion, outputs, b_targets, b_inputs[-1])

                # 检查损失值
                if torch.isnan(loss).any():
                    print("Warning: NaN in loss value")
                    continue

                if loss.item() > 100:
                    print(f"Warning: Large loss value: {loss.item()}")
                    continue

                loss.backward()

                # 检查梯度
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f"Warning: NaN gradient in {name}")
                            continue

                optimizer.step()
                if len(unitary_params) > 0:
                    unitary_optimizer.step()

                train_losses.append(loss.item())

                n_total = len(outputs)
                n_correct = (outputs.argmax(dim=-1) == b_targets).sum().item()
                train_acc = n_correct/n_total

                pbar.update(params.batch_size)
                ordered_dict = {'acc': train_acc, 'loss': loss.item()}
                pbar.set_postfix(ordered_dict=ordered_dict)

        # 计算平均训练损失
        if train_losses:  # 确保有有效的损失值
            avg_train_loss = np.mean(train_losses)
            print(f'Average Train Loss: {avg_train_loss:.4f}')

            # 验证
            model.eval()
            val_output, val_target, val_mask = get_predictions(
                model, params, split='dev')
            val_target, val_output, val_loss = get_loss(
                params, criterion, val_output, val_target, val_mask)

            print('Validation Performance:')
            performances = evaluate(params, val_output, val_target)
            print(
                f'Val Acc = {performances["acc"]:.4f}, Val Loss = {val_loss:.4f}')

            # 更新学习率
            scheduler.step(val_loss)
            if len(unitary_params) > 0:
                unitary_scheduler.step(val_loss)

            # 早停检查
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

            if val_loss < best_val_loss:
                torch.save(model.state_dict(), params.best_model_file)
                print('Best model so far. Saved to file.')
                best_val_loss = val_loss

# 添加早停类


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.verbose:
                print(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def get_criterion(params):
    """获取损失函数"""
    if params.dataset_name.lower() in ['features4quantum', 'features4quantum_fudan']:
        # 使用普通的NLLLoss
        criterion = nn.NLLLoss()
        if hasattr(params, 'loss_weights') and params.loss_weights is not None:
            criterion = nn.NLLLoss(weight=params.loss_weights)
    else:
        # 使用带掩码的NLLLoss
        criterion = MaskedNLLLoss()
        if hasattr(params, 'loss_weights') and params.loss_weights is not None:
            criterion = MaskedNLLLoss(params.loss_weights)

    return criterion


def get_loss(params, criterion, outputs, b_targets, mask):
    #    """计算损失

    #     Args:
    #         params: 配置参数
    #         criterion: 损失函数
    #         outputs: 模型输出 [batch_size, seq_len, output_dim]
    #         b_targets: 目标标签 [batch_size, output_dim]
    #         mask: 掩码

    #     Returns:
    #         b_targets: 处理后的目标标签
    #         outputs: 处理后的输出
    #         loss: 损失值
    #     """
    # Features4Quantum数据集的特殊处理
    if params.dataset_name.lower() in ['features4quantum', 'features4quantum_fudan']:
        # 处理维度
        outputs = outputs.squeeze(1)  # 去掉seq_len维度
        b_targets = b_targets.argmax(dim=-1)  # 转换为类别索引

        # 计算损失
        loss = criterion(outputs, b_targets)
        return b_targets, outputs, loss
    # 其他数据集的原有处理逻辑
    b_targets = b_targets.reshape(-1, params.output_dim).argmax(dim=-1)
    outputs = outputs.reshape(-1, params.output_dim)

    if params.dialogue_context:
        loss = criterion(outputs, b_targets)
    else:
        loss = criterion(outputs, b_targets, mask)
        nonzero_idx = mask.view(-1).nonzero()[:, 0]
        outputs = outputs[nonzero_idx]
        b_targets = b_targets[nonzero_idx]

    return b_targets, outputs, loss


def test(model, params):
    model.eval()
    test_output, test_target, test_mask = get_predictions(
        model, params, split='test')

    print("\n=== 初始数据维度 ===")
    print(f"Initial test_output shape: {test_output.shape}")
    print(f"Initial test_target shape: {test_target.shape}")
    print("="*30)

    if params.dataset_name.lower() in ['features4quantum', 'features4quantum_fudan']:
        # Features4Quantum的特殊处理
        test_output = test_output.squeeze(1)  # 去掉seq_len维度
        test_target = test_target.argmax(dim=-1)  # 转换为类别索引
    elif params.dataset_name.upper() == 'MELD':
        # MELD数据集的特殊处理
        # [batch*seq_len, n_classes]
        test_output = test_output.reshape(-1, params.output_dim)
        # [batch*seq_len]
        test_target = test_target.reshape(-1, params.output_dim).argmax(dim=-1)
        if not params.dialogue_context:
            nonzero_idx = test_mask.view(-1).nonzero()[:, 0]
            test_output = test_output[nonzero_idx]
            test_target = test_target[nonzero_idx]
    else:
        # 其他数据集的原有处理逻辑
        test_target = torch.argmax(
            test_target.reshape(-1, params.output_dim), -1)
        test_output = test_output.reshape(-1, params.output_dim)
        if not params.dialogue_context:
            nonzero_idx = test_mask.view(-1).nonzero()[:, 0]
            test_output = test_output[nonzero_idx]
            test_target = test_target[nonzero_idx]

    # 添加维度检查和打印
    print("\n=== 测试阶段维度检查 ===")
    print(f"test_output shape: {test_output.shape}")
    print(f"test_target shape: {test_target.shape}")
    print("="*30)

    performances = evaluate(params, test_output, test_target)
    # 删除test_target，不需要保存
    return performances


def print_performance(performance_dict, params):
    performance_str = ''
    for key, value in performance_dict.items():
        performance_str = performance_str + '{} = {} '.format(key, value)
    print(performance_str)
    return performance_str


def get_predictions(model, params, split='dev'):
    outputs = []
    targets = []
    masks = []
    iterator = params.reader.get_data(
        iterable=True, shuffle=False, split=split)

    for _ii, data in enumerate(iterator, 0):
        data_x = [inp.to(params.device) for inp in data[:-1]]
        data_t = data[-1].to(params.device)
        data_o = model(data_x)
        if not params.dialogue_context:
            masks.append(data_x[-1])

        outputs.append(data_o.detach())
        targets.append(data_t.detach())

    outputs = torch.cat(outputs)
    targets = torch.cat(targets)
    if not params.dialogue_context:
        masks = torch.cat(masks)

    return outputs, targets, masks

# def save_model(model,params,s):
#     if not os.path.exists('tmp'):
#         os.mkdir('tmp')
#     params.dir_name = str(round(time.time()))
#     dir_path = os.path.join('tmp',params.dir_name)
#     if not os.path.exists(dir_path):
#         os.mkdir(dir_path)
#     torch.save(model.state_dict(),os.path.join(dir_path,'model'))
# #    copyfile(params.config_file, os.path.join(dir_path,'config.ini'))
#     params.export_to_config(os.path.join(dir_path,'config.ini'))
#     params_2 = copy.deepcopy(params)
#     if 'lookup_table' in params_2.__dict__:
#         del params_2.lookup_table
#     if 'sentiment_dic' in params_2.__dict__:
#         del params_2.sentiment_dic
#     del params_2.reader
#     pickle.dump(params_2, open(os.path.join(dir_path,'config.pkl'),'wb'))

#     del params_2
#     if 'save_phases' in params.__dict__ and params.save_phases:
#         print('Saving Phases.')
#         phase_dict = model.get_phases()
#         for key in phase_dict:
#             file_path = os.path.join(dir_path,'{}_phases.pkl'.format(key))
#             pickle.dump(phase_dict[key],open(file_path,'wb'))
#     eval_path = os.path.join(dir_path,'eval')
#     with open(eval_path,'w') as f:
#         f.write(s)


def save_model(model, params, performance_dict):
    """保存模型和配置"""
    # 根据数据集名称创建子目录
    dataset_name = params.dataset_name.lower()
    for base_dir in ['tmp', 'results', 'eval']:
        dir_path = os.path.join(base_dir, dataset_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 创建更有辨识度的目录名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    acc = performance_dict['acc'] if isinstance(
        performance_dict, dict) else eval(performance_dict)['acc']

    # 添加更多性能指标到文件名
    if isinstance(performance_dict, dict):
        macro_f1 = performance_dict['macro avg']['f1-score']
        weighted_f1 = performance_dict['weighted avg']['f1-score']
        params.dir_name = f"{params.network_type}_{timestamp}_ACC={acc:.4f}_MacroF1={macro_f1:.4f}_WeightedF1={weighted_f1:.4f}"
    else:
        perf_dict = eval(performance_dict)
        macro_f1 = perf_dict['macro avg']['f1-score']
        weighted_f1 = perf_dict['weighted avg']['f1-score']
        params.dir_name = f"{params.network_type}_{timestamp}_ACC={acc:.4f}_MacroF1={macro_f1:.4f}_WeightedF1={weighted_f1:.4f}"

    # 创建模型保存目录
    dir_path = os.path.join('tmp', dataset_name, params.dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 保存模型
    torch.save(model.state_dict(), os.path.join(dir_path, 'model'))

    # 同时保存到 results 目录
    results_dir = os.path.join('results', dataset_name, params.network_type)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pth'))

    # 保存配置
    with open(os.path.join(dir_path, 'config.ini'), 'w') as f:
        for key, value in vars(params).items():
            if key not in ['reader', 'lookup_table', 'sentiment_dic']:
                f.write(f'{key} = {value}\n')

    # 保存评估结果
    with open(os.path.join(dir_path, 'eval'), 'w') as f:
        f.write(str(performance_dict))


def save_performance(params, performance_dict):
    """保存性能结果"""
    # 计算总体的指标
    total_precision = np.mean(
        [performance_dict[str(i)]['precision'] for i in range(4)])
    total_recall = np.mean([performance_dict[str(i)]['recall']
                           for i in range(4)])
    total_f1 = np.mean([performance_dict[str(i)]['f1-score']
                       for i in range(4)])

    output_dic = {
        'dataset': params.dataset_name,
        'modality': params.features,
        'network': params.network_type,
        'model_dir_name': params.dir_name,
        'accuracy': performance_dict['acc'],
        'precision': total_precision,
        'recall': total_recall,
        'f1_score': total_f1,
        'macro_f1': performance_dict['macro avg']['f1-score'],
        'weighted_f1': performance_dict['weighted avg']['f1-score']
    }

    # 添加每个类别的指标
    for i in range(4):
        output_dic.update({
            f'class_{i}_acc': performance_dict[str(i)]['acc'],
            f'class_{i}_precision': performance_dict[str(i)]['precision'],
            f'class_{i}_recall': performance_dict[str(i)]['recall'],
            f'class_{i}_f1': performance_dict[str(i)]['f1-score']
        })

    # 打印详细结果
    print("\n=== 详细实验结果 ===")
    print(f"数据集: {params.dataset_name}")
    print(f"模型: {params.network_type}")
    print("\n整体性能:")
    print(f"Accuracy: {performance_dict['acc']:.4f}")
    print(f"Precision: {performance_dict['weighted avg']['precision']:.4f}")
    print(f"Recall: {performance_dict['weighted avg']['recall']:.4f}")
    print(f"Weighted F1: {performance_dict['weighted avg']['f1-score']:.4f}")
    print(f"Macro F1: {performance_dict['macro avg']['f1-score']:.4f}")

    print("\n各类别性能:")
    for i in range(4):
        print(f"\n类别 {i}:")
        print(f"  Accuracy: {performance_dict[str(i)]['acc']:.4f}")
        print(f"  Precision: {performance_dict[str(i)]['precision']:.4f}")
        print(f"  Recall: {performance_dict[str(i)]['recall']:.4f}")
        print(f"  F1-score: {performance_dict[str(i)]['f1-score']:.4f}")
    print("="*30)

    # 使用pd.concat替代append
    df = pd.concat([pd.DataFrame([output_dic])], ignore_index=True)

    # 设置更有辨识度的输出文件名，包含数据集信息
    dataset_name = params.dataset_name.lower()
    if not hasattr(params, 'output_file') or params.output_file is None:
        params.output_file = os.path.join(
            'eval', dataset_name, f'{params.network_type}_results.csv')

    # 如果文件已存在，则读取并追加
    if os.path.exists(params.output_file):
        existing_df = pd.read_csv(params.output_file, index_col=0)
        df = pd.concat([existing_df, df], ignore_index=True)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(params.output_file), exist_ok=True)

    # 保存结果
    df.to_csv(params.output_file, encoding='utf-8', index=True)
