from transformers import (
    BertTokenizer, BertModel, 
    Wav2Vec2Processor, Wav2Vec2Model,
    VivitImageProcessor, VivitModel
)
import torch
import torchaudio
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import librosa
import cv2
from PIL import Image
import pickle
from BEATs import BEATs, BEATsConfig

class FeatureExtractor:
    def __init__(self, device):
        self.device = device
        
        # 初始化文本模型
        self.text_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.text_model = BertModel.from_pretrained('bert-base-chinese').to(device)
        
        # 初始化音频模型
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
        
        # 初始化视频模型
        self.video_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.video_model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400").to(device)
        
        # 初始化BEATs模型
        checkpoint = torch.load('/home/tjc/audio/QMamba/feature_extract/BEATs_iter3_plus_AS2M.pt')
        cfg = BEATsConfig(checkpoint['cfg'])
        self.beats_model = BEATs(cfg).to(device)
        self.beats_model.load_state_dict(checkpoint['model'])
        self.beats_model.eval()

    def extract_text_features(self, text_path):
        # 读取文本
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # 处理文本
        inputs = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 提取特征
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)  # [1, 768]
        
        return features.cpu().numpy()

    def extract_audio_features(self, audio_path):
        # 读取音频
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # 处理音频
        inputs = self.audio_processor(audio, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 提取特征
        with torch.no_grad():
            outputs = self.audio_model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)  # [1, 768]
        
        return features.cpu().numpy()

    def extract_video_features(self, video_path):
        # 读取视频帧
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < 32:  # 提取32帧
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        # 如果帧数不足，循环填充
        while len(frames) < 32:
            frames.extend(frames[:32-len(frames)])
        
        # 处理视频帧
        inputs = self.video_processor(frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 提取特征
        with torch.no_grad():
            outputs = self.video_model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)  # [1, 768]
        
        return features.cpu().numpy()

    def extract_beats_features(self, audio_path):
        """提取BEATs特征"""
        # 读取音频
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # 处理音频
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
        padding_mask = torch.zeros(1, waveform.shape[1]).bool().to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features, _ = self.beats_model.extract_features(waveform, padding_mask=padding_mask)
            features = features.mean(dim=1)  # 取平均得到固定维度的特征
        
        return features.cpu().numpy()

def process_data(csv_path, feature_folder_name):
    """
    Args:
        csv_path: CSV文件的路径
        feature_folder_name: 特征保存的子文件夹名称
    """
    # 创建输出目录结构
    base_dir = "/home/tjc/audio/QMamba/feature_extract/Features4Quantum"
    feature_dir = os.path.join(base_dir, feature_folder_name)
    os.makedirs(feature_dir, exist_ok=True)
    
    print(f"特征将被保存到: {feature_dir}")
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 过滤掉没有标签的样本
    df = df.dropna(subset=['label'])
    print(f"Total samples after filtering: {len(df)}")
    
    # 修改路径格式：替换旧前缀为新前缀
    old_prefix = "/home/tanjiachen-23/data/"
    new_prefix = "/home/tjc/audio/Data/"
    
    df['video_path'] = df['video_path'].apply(lambda x: x.replace(old_prefix, new_prefix))
    df['audio_path'] = df['audio_path'].apply(lambda x: x.replace(old_prefix, new_prefix))
    df['text_path'] = df['text_path'].apply(lambda x: x.replace(old_prefix, new_prefix))
    
    # 保存新的CSV文件
    processed_csv_path = os.path.join(feature_dir, "processed_paths.csv")
    df.to_csv(processed_csv_path, index=False)
    print(f"处理后的CSV文件保存到: {processed_csv_path}")
    
    # 初始化特征提取器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = FeatureExtractor(device)
    
    # 批量处理
    features_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing files"):
        try:
            # 检查文件是否存在
            if not all(os.path.exists(path) for path in [row['video_path'], row['audio_path'], row['text_path']]):
                print(f"Warning: Files not found for sample {idx}")
                print(f"Video path: {row['video_path']}")
                print(f"Audio path: {row['audio_path']}")
                print(f"Text path: {row['text_path']}")
                continue
                
            # 提取特征
            text_features = extractor.extract_text_features(row['text_path'])
            audio_features = extractor.extract_audio_features(row['audio_path'])
            beats_features = extractor.extract_beats_features(row['audio_path'])
            video_features = extractor.extract_video_features(row['video_path'])
            
            # 创建特征字典
            feature_dict = {
                'id': idx,
                'path': row['video_path'],
                'label': row['label'],
                'text_features': text_features,
                'audio_features': audio_features,
                'beats_features': beats_features,
                'video_features': video_features
            }
            features_list.append(feature_dict)
                
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            continue
    
    # 保存为pkl文件
    with open(os.path.join(feature_dir, "features.pkl"), 'wb') as f:
        pickle.dump(features_list, f)
    
    print(f"Feature extraction completed! Total processed samples: {len(features_list)}")

if __name__ == "__main__":
    try:
        # 指定特征保存的子文件夹名称
        feature_folder_name = "fudan_val"
        process_data("/home/tjc/audio/Data/Fudan_val_processed.csv", feature_folder_name)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
