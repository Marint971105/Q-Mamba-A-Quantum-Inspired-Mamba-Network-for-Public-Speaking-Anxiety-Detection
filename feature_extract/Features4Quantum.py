from transformers import (
    BertTokenizer, BertModel, 
    Wav2Vec2Processor, Wav2Vec2Model,
    VivitImageProcessor, VivitModel
)
import torch
import torchaudio
import numpy as np
import librosa
import cv2
from PIL import Image
import pandas as pd
from tqdm import tqdm
import os
import pickle
from BEATs import BEATs, BEATsConfig

class FeatureExtractor:
    def __init__(self, device):
        self.device = device
        print("Initializing models...")
        
        # BERT
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = BertModel.from_pretrained('bert-base-multilingual-cased').to(device).eval()
        
        # Wav2Vec2
        self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()
        
        # ViViT
        self.vivit_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.vivit_model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400").to(device).eval()
        
        # BEATs
        checkpoint = torch.load('/home/tjc/audio/QMamba/feature_extract/BEATs_iter3_plus_AS2M.pt')
        cfg = BEATsConfig(checkpoint['cfg'])
        self.beats_model = BEATs(cfg).to(device)
        self.beats_model.load_state_dict(checkpoint['model'])
        self.beats_model.eval()

    def load_audio_for_beats(self, audio_path, target_sr=16000):
        """加载音频文件并进行预处理(用于BEATs)"""
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 重采样到16kHz
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        return waveform.to(self.device)

    def extract_beats_features(self, audio_path):
        """提取BEATs特征"""
        waveform = self.load_audio_for_beats(audio_path)
        padding_mask = torch.zeros(1, waveform.shape[1]).bool().to(self.device)
        
        with torch.no_grad():
            representation = self.beats_model.extract_features(waveform, padding_mask=padding_mask)[0]
            averaged_features = torch.mean(representation, dim=1).cpu().numpy()
        
        return averaged_features

    def extract_text_features(self, text_path):
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.bert_model(**inputs, output_hidden_states=True)
            return outputs.hidden_states[-2].mean(dim=1).cpu().numpy()

    def extract_audio_features(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=16000)
        
        with torch.no_grad():
            inputs = self.wav2vec2_processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.wav2vec2_model(**inputs, output_hidden_states=True)
            return outputs.hidden_states[-2].mean(dim=1).cpu().numpy()

    def extract_video_features(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames = 32
        indices = np.linspace(0, total_frames-1, min(total_frames, num_frames), dtype=int)
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
            frame_count += 1
        cap.release()

        if len(frames) < num_frames:
            zero_frame = np.zeros((224, 224, 3), dtype=np.uint8)
            frames.extend([zero_frame] * (num_frames - len(frames)))

        frames = np.array(frames)
        
        with torch.no_grad():
            inputs = self.vivit_processor(list(frames), return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.vivit_model(**inputs, output_hidden_states=True)
            return outputs.hidden_states[-2].mean(dim=1).cpu().numpy()

def process_data(csv_path, feature_folder_name):
    """
    Args:
        csv_path: CSV文件的路径
        feature_folder_name: 特征保存的子文件夹名称
    """
    # 创建输出目录结构
    base_dir = "Features4Quantum"
    feature_dir = os.path.join(base_dir, feature_folder_name)
    os.makedirs(feature_dir, exist_ok=True)
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 修改路径 - 将旧路径映射到新路径
    def convert_path(old_path):
        # 提取关键部分：chinese/A/xxx.mp4
        parts = old_path.split('/')
        for i, part in enumerate(parts):
            if part == 'chinese':
                key_path = '/'.join(parts[i:])
                return os.path.join("/home/tjc/audio/Data", key_path)
        return old_path

    # 应用路径转换
    df['video_path'] = df['video_path'].apply(convert_path)
    df['audio_path'] = df['audio_path'].apply(convert_path)
    df['text_path'] = df['text_path'].apply(convert_path)
    
    # 保存新的CSV文件，包含转换后的路径
    df.to_csv(os.path.join(feature_dir, "processed_paths.csv"), index=False)
    
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
    
    print("Feature extraction completed!")

if __name__ == "__main__":
    try:
        # 指定特征保存的子文件夹名称
        feature_folder_name = "chinese_val"  # 你可以修改这个名称
        process_data("/home/tjc/audio/Data/CN_val_processed_processed.csv", feature_folder_name)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()