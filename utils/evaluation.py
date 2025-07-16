# -*- coding: utf-8 -*-

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def evaluate(params, outputs, targets):
    """评估模型性能"""
    outputs_max_ids = outputs.argmax(dim=-1).cpu().numpy()
    targets_max_ids = targets.cpu().numpy()
    
    # 设置target_names
    if params.dataset_name.lower() == 'features4quantum':
        params.emotion_dic = ['0', '1', '2', '3']
    elif params.dataset_name.upper() == 'MELD':
        # MELD数据集的标签已经是0-6的数字
        params.emotion_dic = [str(i) for i in range(7)]  # ['0', '1', '2', '3', '4', '5', '6']
    elif params.dataset_name.upper() == 'IEMOCAP':
        # IEMOCAP数据集使用数字标签
        params.emotion_dic = [str(i) for i in range(6)]  # ['0', '1', '2', '3', '4', '5']
    
    # 使用原有的评估逻辑
    report = classification_report(
        targets_max_ids, 
        outputs_max_ids, 
        target_names=params.emotion_dic, 
        output_dict=True,
        zero_division=0
    )
    
    # 计算性能指标
    performances = {}
    performances['acc'] = report['accuracy']
    
    # 计算每个类别的性能
    for emotion in params.emotion_dic:
        performances[emotion] = {}
        performances[emotion]['precision'] = report[emotion]['precision']
        performances[emotion]['recall'] = report[emotion]['recall']
        performances[emotion]['f1-score'] = report[emotion]['f1-score']
        # 添加每个类别的准确率
        class_mask = targets_max_ids == int(emotion)
        if class_mask.sum() > 0:  # 避免除零
            class_acc = (outputs_max_ids[class_mask] == targets_max_ids[class_mask]).mean()
            performances[emotion]['acc'] = class_acc
        else:
            performances[emotion]['acc'] = 0.0
    
    # 添加宏平均和加权平均
    performances['macro avg'] = report['macro avg']
    performances['weighted avg'] = report['weighted avg']
    
    # 添加宏平均和加权平均的acc
    macro_acc = np.mean([performances[e]['acc'] for e in params.emotion_dic])
    weighted_acc = np.average([performances[e]['acc'] for e in params.emotion_dic],
                            weights=[np.sum(targets_max_ids == int(e)) for e in params.emotion_dic])
    
    performances['macro avg']['acc'] = macro_acc
    performances['weighted avg']['acc'] = weighted_acc
    
    # 为MELD设置少数类关注
    if params.dataset_name.upper() == 'MELD':
        minority_classes = ['0','1', '2', '3', '5', '6']  # anger, disgust, fear, sadness, surprise
    elif params.dataset_name.upper() == 'IEMOCAP':
        minority_classes = ['0', '1', '3', '4']  # happy, sad, angry, excited
    else:
        minority_classes = ['2', '3']
    
    minority_performance = {
        'precision': np.mean([performances[c]['precision'] for c in minority_classes]),
        'recall': np.mean([performances[c]['recall'] for c in minority_classes]),
        'f1-score': np.mean([performances[c]['f1-score'] for c in minority_classes]),
        'acc': np.mean([performances[c]['acc'] for c in minority_classes])
    }
    performances['minority_classes_avg'] = minority_performance
    
    return performances