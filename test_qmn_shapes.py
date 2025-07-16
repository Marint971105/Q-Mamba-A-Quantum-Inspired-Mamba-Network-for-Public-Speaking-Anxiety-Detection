import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from models.QMN import QMN

def print_shape(name, tensor, extra_info=""):
    """打印张量的shape和额外信息"""
    def get_shape(t):
        if isinstance(t, torch.Tensor):
            return list(t.shape)
        elif isinstance(t, list):
            return [get_shape(item) for item in t]
        else:
            return None
    
    shape = get_shape(tensor)
    print(f"{name}: {shape} {extra_info}")

# 配置参数 - 根据MELD数据集调整
opt = SimpleNamespace(
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    input_dims = [600, 300, 300],  # [文本维度, 声学维度, 视觉维度] - MELD格式
    embed_dim = 50,                # 量子态维度
    speaker_num = 9,               # MELD数据集的说话者数量
    dataset_name = 'meld',         # 数据集名称
    output_dim = 7,                # MELD的情感类别数
    output_cell_dim = 24,          # 输出层隐藏维度
    out_dropout_rate = 0.1,        # dropout率
    num_layers = 1                 # RNN层数
)

# 创建模型
model = QMN(opt)
model = model.to(opt.device)

# 准备测试数据 - 使用MELD格式
batch_size = 1      # MELD示例批次大小
seq_len = 33        # MELD最大序列长度

# 创建输入数据
textual = torch.randn(batch_size, seq_len, opt.input_dims[0])    
print_shape("输入文本特征", textual, 
           "[batch_size, max_seq_len, text_dim] - BERT嵌入")

acoustic = torch.randn(batch_size, seq_len, opt.input_dims[1])   
print_shape("输入声学特征", acoustic,
           "[batch_size, max_seq_len, acoustic_dim] - 声学特征")

visual = torch.randn(batch_size, seq_len, opt.input_dims[2])     
print_shape("输入视觉特征", visual,
           "[batch_size, max_seq_len, visual_dim] - 视觉特征")

# 说话者掩码 - MELD格式的one-hot编码
speaker_mask = torch.zeros(batch_size, seq_len, opt.speaker_num)
# 模拟对话中交替的说话者
for i in range(seq_len):
    speaker_mask[0, i, i % opt.speaker_num] = 1
print_shape("说话者掩码", speaker_mask,
           "[batch_size, max_seq_len, speaker_num] - one-hot编码的说话者ID")

# 对话掩码 - 标识有效句子
dialogue_mask = torch.ones(batch_size, seq_len)
print_shape("对话掩码", dialogue_mask,
           "[batch_size, max_seq_len] - 1表示有效句子,0表示填充")

# 移到设备
textual = textual.to(opt.device)
acoustic = acoustic.to(opt.device)
visual = visual.to(opt.device)
speaker_mask = speaker_mask.to(opt.device)
dialogue_mask = dialogue_mask.to(opt.device)

def forward_with_shapes(model, in_modalities):
    print("\n=== 模型内部维度变换追踪 ===")
    
    # 1. 输入处理
    smask = in_modalities[-2]  # 提取说话者掩码
    in_modalities = in_modalities[:-2]  # 分离模态特征
    print_shape("处理后的输入模态", in_modalities, 
               "列表包含三个模态张量:[文本,声学,视觉]")
    
    # 2. 特征投影 - 将各模态投影到相同维度
    utterance_reps = [nn.ReLU()(projection(x)) 
                     for x, projection in zip(in_modalities, model.projections)]
    print_shape("投影后的特征", utterance_reps,
               f"每个模态: [batch_size, max_seq_len, {opt.embed_dim}] - 统一特征维度")
    
    # 3. 模态权重计算
    weights = [model.norm(rep) for rep in utterance_reps]
    weights = F.softmax(torch.cat(weights, dim=-1), dim=-1)
    print_shape("模态融合权重", weights,
               f"[batch_size, max_seq_len, {3*opt.embed_dim}] - 三个模态的权重分布")
    
    # 4. 量子态准备
    # 振幅
    amplitudes = [F.normalize(rep, dim=-1) for rep in utterance_reps]
    print_shape("量子态振幅", amplitudes,
               f"每个模态: [batch_size, max_seq_len, {opt.embed_dim}] - 归一化特征")
    
    # 相位
    phases = [phase_embed(smask.argmax(dim=-1)) 
              for phase_embed in model.phase_embeddings]
    print_shape("量子态相位", phases,
               f"每个模态: [batch_size, max_seq_len, {opt.embed_dim}] - 说话者编码")
    
    # 5. 量子态构建
    # 纯态
    unimodal_pure = [model.multiply([phase, amplitude]) 
                    for phase, amplitude in zip(phases, amplitudes)]
    print_shape("模态纯态", unimodal_pure,
               f"每个模态: [[batch,seq_len,{opt.embed_dim}], [batch,seq_len,{opt.embed_dim}]] - [实部,虚部]")
    
    # 密度矩阵
    unimodal_matrices = [model.outer(s) for s in unimodal_pure]
    print_shape("模态密度矩阵", unimodal_matrices,
               f"每个模态: [[batch,seq_len,{opt.embed_dim},{opt.embed_dim}], [...]] - [实部,虚部]")
    
    # 6. 混合状态
    in_states = model.mixture([unimodal_matrices, weights])
    print_shape("混合量子态", in_states,
               f"[seq_len, [batch,{opt.embed_dim},{opt.embed_dim}], [...]] - 时序混合状态")
    
    # 7. RNN处理
    for l in range(model.num_layers):
        # 初始化隐状态
        h_r = torch.stack(batch_size*[torch.eye(model.embed_dim)/model.embed_dim], dim=0)
        h_i = torch.zeros_like(h_r)
        h = [h_r.to(model.device), h_i.to(model.device)]
        print_shape(f"RNN层{l}初始隐状态", h,
                   f"[[batch,{opt.embed_dim},{opt.embed_dim}], [...]] - [实部,虚部]")
        
        # 序列处理
        all_h = []
        for t in range(seq_len):
            h = model.recurrent_cells[l](in_states[t], h)
            all_h.append(h)
        in_states = all_h
        print_shape(f"RNN层{l}输出", in_states,
                   f"[seq_len个[[batch,{opt.embed_dim},{opt.embed_dim}], [...]]] - 序列输出")
    
    # 8. 测量和输出
    output = []
    for _h in in_states:
        # 量子测量
        measurement_probs = model.measurement(_h)
        print_shape("量子测量结果", measurement_probs,
                   f"[batch_size, {opt.embed_dim}] - 投影到经典空间")
        
        # 分类层
        _output = model.fc_out(measurement_probs)
        output.append(_output)
    
    # 堆叠序列输出
    output = torch.stack(output, dim=-2)
    print_shape("模型输出", output,
               f"[batch_size, max_seq_len, {opt.output_dim}] - 序列情感预测")
    
    # 对数概率
    log_prob = F.log_softmax(output, 2)
    print_shape("最终概率分布", log_prob,
               f"[batch_size, max_seq_len, {opt.output_dim}] - 情感类别概率")
    
    return log_prob

# 运行测试
print("\n=== 开始前向传播测试 ===")
inputs = [textual, acoustic, visual, speaker_mask, dialogue_mask]
with torch.no_grad():
    output = forward_with_shapes(model, inputs) 