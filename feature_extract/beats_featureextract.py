import torch
import torchaudio
from BEATs import BEATs, BEATsConfig

def load_audio(audio_path, target_sr=16000):
    """
    加载音频文件并进行重采样
    """
    # 加载音频
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # 如果是双声道,转换为单声道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # 重采样到16kHz(如果需要)
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    
    return waveform

# 加载预训练模型
checkpoint = torch.load('/home/tjc/audio/QMamba/feature_extract/BEATs_iter3_plus_AS2M.pt')
cfg = BEATsConfig(checkpoint['cfg'])
BEATs_model = BEATs(cfg)
BEATs_model.load_state_dict(checkpoint['model'])
BEATs_model.eval()

# 加载WAV文件
audio_path = "/home/tjc/audio/Data/chinese/A/1_lixihuan_017.wav"
waveform = load_audio(audio_path)

# 创建padding mask
padding_mask = torch.zeros(1, waveform.shape[1]).bool()

# 提取特征
with torch.no_grad():
    representation = BEATs_model.extract_features(waveform, padding_mask=padding_mask)[0]
    
    # 对时间维度进行平均池化 [1, T, 768] -> [1, 768]
    averaged_features = torch.mean(representation, dim=1)

print("Input audio shape:", waveform.shape)
print("Original feature shape:", representation.shape)
print("Averaged feature shape:", averaged_features.shape)

# 可选：保存提取的特征
# torch.save(averaged_features, 'beats_averaged_features.pt')