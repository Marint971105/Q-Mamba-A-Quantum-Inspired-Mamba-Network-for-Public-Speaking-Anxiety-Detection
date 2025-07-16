import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.QMN import QMN
from models.QMamba import QMamba

def create_test_data(batch_size=2, seq_len=3):
    """创建测试数据"""
    # 创建模拟的多模态数据
    text = torch.randn(batch_size, seq_len, 768)    
    audio = torch.randn(batch_size, seq_len, 768)   
    beats = torch.randn(batch_size, seq_len, 768)   
    visual = torch.randn(batch_size, seq_len, 768)  
    
    # QMN额外需要的数据
    speaker = torch.zeros(batch_size, seq_len, 1)   
    mask = torch.ones(batch_size, seq_len)          
    
    class Options:
        def __init__(self):
            self.device = torch.device('cpu')
            self.embed_dim = 50
            self.output_dim = 4  
            self.output_cell_dim = 32
            self.out_dropout_rate = 0.1
            
            # QMN参数
            self.speaker_num = 1
            self.dataset_name = 'test'
            self.num_layers = 1
            self.input_dims = [768, 768, 768, 768]  # text, audio, beats, visual
            
            # QMamba参数
            self.qmamba_input_dims = [768, 768, 768]  # text, audio, visual
    
    opt = Options()
    qmn_data = [text, audio, beats, visual, speaker, mask]
    qmamba_data = [text, audio, beats, visual]
    return qmn_data, qmamba_data, opt

def test_quantum_state_construction():
    """测试量子态构造过程"""
    print("\n=== 量子态构造测试 ===")
    
    # 创建测试数据
    batch_size = 2
    seq_len = 1  # 简化测试,使用单个时间步
    qmn_data, qmamba_data, opt = create_test_data(batch_size, seq_len)
    
    # 初始化模型
    qmn = QMN(opt)
    qmamba = QMamba(opt)
    print(qmn)
    print(qmamba)
    try:
        # 1. 相位生成测试
        print("\n1. 相位生成对比:")
        
        # QMN相位生成
        smask = qmn_data[4]  # speaker mask
        qmn_phases = [phase_embed(smask.argmax(dim=-1)) 
                     for phase_embed in qmn.phase_embeddings]
        print("QMN相位:")
        print(f"- 输入shape: {smask.shape}")
        print(f"- 输出shape: {qmn_phases[0].shape}")
        
        # QMamba相位生成 - 保持序列长度维度
        beats = qmamba_data[2]
        beats_proj = qmamba.beats_projection(beats)  # [batch_size, seq_len, embed_dim]
        phase = nn.ReLU()(beats_proj)  # 保持[batch_size, seq_len, embed_dim]
        print("\nQMamba相位:")
        print(f"- 输入shape: {beats.shape}")
        print(f"- 输出shape: {phase.shape}")
        
        # 2. 振幅生成测试
        print("\n2. 振幅生成对比:")
        
        # QMN振幅
        qmn_reps = [nn.ReLU()(proj(x)) for x, proj in zip(qmn_data[:-2], qmn.projections)]
        qmn_amplitudes = [F.normalize(rep, dim=-1) for rep in qmn_reps]
        print("QMN振幅:")
        print(f"- 模态数: {len(qmn_amplitudes)}")
        print(f"- 振幅shape: {qmn_amplitudes[0].shape}")
        
        # QMamba振幅 - 保持序列长度维度
        qmamba_reps = [nn.ReLU()(proj(x)) for x, proj in zip(qmamba_data[:3], qmamba.projections)]
        qmamba_amplitudes = [F.normalize(rep, dim=-1) for rep in qmamba_reps]  # 不再squeeze
        print("\nQMamba振幅:")
        print(f"- 模态数: {len(qmamba_amplitudes)}")
        print(f"- 振幅shape: {qmamba_amplitudes[0].shape}")
        
        # 3. 量子态构造测试
        print("\n3. 量子态构造对比:")
        
        # QMN量子态
        qmn_states = [qmn.multiply([p, a]) for p, a in zip(qmn_phases, qmn_amplitudes)]
        print("QMN量子态:")
        print(f"- 状态数: {len(qmn_states)}")
        print(f"- 实部shape: {qmn_states[0][0].shape}")
        print(f"- 虚部shape: {qmn_states[0][1].shape}")
        
        # QMamba量子态 - 保持序列长度维度
        qmamba_states = [qmamba.multiply([phase, amp]) for amp in qmamba_amplitudes]
        print("\nQMamba量子态:")
        print(f"- 状态数: {len(qmamba_states)}")
        print(f"- 实部shape: {qmamba_states[0][0].shape}")
        print(f"- 虚部shape: {qmamba_states[0][1].shape}")
        
        # 4. 验证量子态性质
        print("\n4. 量子态性质验证:")
        
        # 验证范数
        def verify_norm(state):
            norm = torch.sqrt(state[0].pow(2).sum(-1) + state[1].pow(2).sum(-1))
            return norm.mean().item()
        
        qmn_norm = verify_norm(qmn_states[0])
        qmamba_norm = verify_norm(qmamba_states[0])
        
        print(f"QMN量子态范数: {qmn_norm:.4f}")
        print(f"QMamba量子态范数: {qmamba_norm:.4f}")
        
        # 验证通过条件
        norm_ok = (abs(qmn_norm - 1.0) < 0.01 and abs(qmamba_norm - 1.0) < 0.01)
        return norm_ok
        
    except Exception as e:
        print(f"\nError in quantum state construction test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始量子态构造测试...")
    success = test_quantum_state_construction()
    print(f"\n测试结果: {'PASSED' if success else 'FAILED'}")
