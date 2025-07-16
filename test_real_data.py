import torch
from types import SimpleNamespace
from models.QMamba import QMamba
from dataset.features4quantum_reader import Features4QuantumReader

def test_model(opt, model_name):
    """测试指定的模型"""
    print(f"\n=== 测试 {model_name} 模型 ===\n")
    
    # 设置模型类型
    opt.sequence_model = model_name
    
    print("\n初始化模型...")
    model = QMamba(opt).to(opt.device)
    
    # 获取一个batch的数据
    batch = next(iter(train_loader))
    
    print("\n=== 输入数据维度 ===")
    for i, feature in enumerate(batch[:-1]):
        print(f"模态{i}: {feature.shape}")
    print(f"标签: {batch[-1].shape}")
    
    # 测试前向传播
    print("\n执行前向传播...")
    with torch.no_grad():
        inputs = [x.to(opt.device) for x in batch[:-1]]
        outputs = model(inputs)
        print("\n=== 输出维度 ===")
        print(f"模型输出: {outputs.shape}")
    
    print("\n测试完成!")

def test_with_real_data():
    print("\n=== 使用真实数据测试模型 ===\n")
    
    # 配置参数，与run_qmamba_CN.py保持一致
    opt = SimpleNamespace(
        # 数据相关
        pickle_dir_path = '/home/tjc/audio/QMamba/feature_extract',
        features = 'textual,acoustic,beats,visual',
        dataset_name = 'features4quantum',
        embedding_enabled = False,
        dialogue_context = False,
        
        # 模型相关
        network_type = 'qmamba',
        embed_dim = 50,
        speaker_num = 1,
        output_dim = 4,
        output_cell_dim = 50,
        out_dropout_rate = 0.2,
        num_layers = 1,
        
        # 训练相关
        gpu = 0,
        batch_size = 64,
        epochs = 50,
        lr = 0.001,
        unitary_lr = 0.001,
        clip = 0.8,
        patience = 10,
        min_lr = 1e-5,
        
        # 设备
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        
        # 保存相关
        save_dir = 'results/qmamba/Transformer',
        dir_name = None,
        output_file = None,
        config_file = 'config.ini',
        
        # 输入维度
        input_dims = [768, 768, 768, 768],
        total_input_dim = 768 * 4,
        
        # 类别平衡相关
        class_balance_strategy = 'weighted_loss',
       
        # Transformer特有参数
        num_heads = 5,
        attn_dropout = 0.1,
        relu_dropout = 0.1,
        res_dropout = 0.1,
        embed_dropout = 0.1,
        
        # RNN特有参数
        rnn_type = 'rnn',  # 'lstm', 'gru', 或 'rnn'
        bidirectional = False,
        rnn_dropout = 0.1,
        
        # CNN特有参数
        kernel_size = 3,
        cnn_dropout = 0.1,
    )
    
    print("\n加载数据...")
    reader = Features4QuantumReader(opt)
    reader.read(opt)
    
    print("\n=== 类别统计 ===")
    print("类别样本数:")
    # 获取训练集数据加载器
    global train_loader
    train_loader = reader.get_data(shuffle=False, split='train')
    
    # 统计每个类别的样本数
    label_counts = torch.zeros(opt.output_dim)
    for batch in train_loader:
        labels = batch[-1]
        for i in range(opt.output_dim):
            label_counts[i] += (labels == i).sum().item()
    
    # 打印统计信息
    total_samples = label_counts.sum().item()
    for i in range(opt.output_dim):
        count = int(label_counts[i])
        print(f"类别 {i}: {count} 样本 ({count/total_samples:.4f})")
    print(f"总样本数: {int(total_samples)}")
    
    # 测试四种模型
    models = ['mamba', 'transformer', 'rwkv', 'rnn','cnn']
    for model_name in models:
        test_model(opt, model_name)

if __name__ == "__main__":
    test_with_real_data() 