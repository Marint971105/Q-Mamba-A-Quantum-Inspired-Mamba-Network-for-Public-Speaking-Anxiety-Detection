import torch
from types import SimpleNamespace
from models.QMamba_meld import QMamba_meld
from dataset.meld_reader import MELDReader

def test_model(opt, model_name):
    """测试指定的模型"""
    print(f"\n=== 测试 {model_name} 模型 ===\n")
    
    # 设置模型类型
    opt.sequence_model = model_name
    
    print("\n初始化模型...")
    model = QMamba_meld(opt).to(opt.device)
    print(model)
    # 获取一个batch的数据
    batch = next(iter(train_loader))
    
    print("\n=== 输入数据维度 ===")
    for i, feature in enumerate(batch[:-1]):
        if i == 0:
            print(f"文本特征: {feature.shape}")
        elif i == 1:
            print(f"视觉特征: {feature.shape}")
        elif i == 2:
            print(f"音频特征: {feature.shape}")
    print(f"标签: {batch[-1].shape}")
    
    # 测试前向传播
    print("\n执行前向传播...")
    with torch.no_grad():
        inputs = [x.to(opt.device) for x in batch[:-1]]
        outputs = model(inputs)
        print("\n=== 输出维度 ===")
        print(f"模型输出: {outputs.shape}")
        
        # 打印模型中间状态的维度
        print("\n=== 中间状态维度 ===")
        print(f"量子态数量: {len(model.quantum_states)}")
        print(f"纠缠态数量: {len(model.entangled_states)}")
    
    print("\n测试完成!")

def print_dataset_stats(reader, opt, split='train'):
    """打印指定数据集的统计信息"""
    print(f"\n=== {split}集统计 ===")
    data_loader = reader.get_data(shuffle=False, split=split)
    
    # 统计每个类别的样本数
    label_counts = torch.zeros(opt.output_dim)
    total_utterances = 0
    total_dialogues = 0
    seq_lengths = []
    
    for batch in data_loader:
        labels = batch[-1]  # [batch_size, seq_len, num_classes]
        batch_size, seq_len, _ = labels.shape
        total_dialogues += batch_size
        total_utterances += batch_size * seq_len
        seq_lengths.extend([seq_len] * batch_size)
        
        # 统计每个类别的样本数
        for i in range(opt.output_dim):
            label_counts[i] += (labels[:,:,i] == 1).sum().item()
    
    # 打印统计信息
    print(f"\n对话数: {total_dialogues}")
    print(f"话语数: {total_utterances}")
    print(f"平均对话长度: {sum(seq_lengths)/len(seq_lengths):.2f}")
    print(f"最短对话长度: {min(seq_lengths)}")
    print(f"最长对话长度: {max(seq_lengths)}")
    
    print("\n情感类别分布:")
    emotions = ['neutral', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    for i in range(opt.output_dim):
        count = int(label_counts[i])
        percentage = count/total_utterances * 100
        print(f"{emotions[i]}: {count} ({percentage:.2f}%)")

def test_with_meld_data():
    print("\n=== 使用MELD数据集测试模型 ===\n")
    
    # 配置参数
    opt = SimpleNamespace(
        # 数据相关
        pickle_dir_path = '/home/tjc/audio/QMamba/Data',
        features = 'textual,visual,acoustic',  # MELD的三个模态
        dataset_name = 'meld',
        embedding_enabled = False,
        dialogue_context = False,
        label = 'emotion',
        
        # 输入维度 (根据实际MELD特征维度修改)
        input_dims = [600, 300, 300],  # text:600, visual:300, acoustic:300
        total_input_dim = 600 + 300 + 300,
        
        # 模型相关
        network_type = 'qmamba',
        embed_dim = 50,
        output_dim = 7,   # MELD是7分类问题
        output_cell_dim = 50,
        out_dropout_rate = 0.2,
        num_layers = 1,
        
        # 训练相关
        gpu = 0,
        batch_size = 32,
        epochs = 2,
        lr = 0.001,
        unitary_lr = 0.001,
        clip = 0.8,
        patience = 10,
        min_lr = 1e-5,
        
        # 设备
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        
        # 保存相关
        save_dir = 'results/qmamba/meld',
        dir_name = None,
        output_file = None,
        config_file = 'config.ini',
        
        # Transformer特有参数
        num_heads = 5,
        dim_feedforward = 50,
        attn_dropout = 0.1,
        relu_dropout = 0.1,
        res_dropout = 0.1,
        embed_dropout = 0.1,
        
        # RNN特有参数
        rnn_type = 'rnn',
        bidirectional = False,
        rnn_dropout = 0.1,
        
        # CNN特有参数
        kernel_size = 3,
        cnn_dropout = 0.1,
    )
    
    print("\n加载数据...")
    reader = MELDReader(opt)
    reader.read(opt)
    
    # 打印各个数据集的统计信息
    print_dataset_stats(reader, opt, 'train')
    print_dataset_stats(reader, opt, 'dev')
    print_dataset_stats(reader, opt, 'test')
    
    # 获取训练集数据加载器
    global train_loader
    train_loader = reader.get_data(shuffle=False, split='train')
    
    # 测试所有模型
    models = ['mamba', 'transformer', 'rwkv', 'rnn', 'cnn']
    for model_name in models:
        test_model(opt, model_name)

if __name__ == "__main__":
    test_with_meld_data() 