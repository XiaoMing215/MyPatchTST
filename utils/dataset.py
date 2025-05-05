import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class SineDataset(Dataset):
    def __init__(self, csv_path, input_len=96, pred_len=24, split='train', train_ratio=0.8):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len

        df = pd.read_csv(csv_path)
        self.features = [col for col in df.columns if col.startswith("feature_")]
        self.n_features = len(self.features)

        # 按 sample_id 分组
        grouped = df.groupby("sample_id")
        self.series = [group[self.features].values for _, group in grouped]

        # 滑窗切片
        self.samples = []
        for seq in self.series:
            total_len = input_len + pred_len
            if seq.shape[0] < total_len:
                continue
            for i in range(seq.shape[0] - total_len + 1):
                input_seq = seq[i: i + input_len]
                target_seq = seq[i + input_len: i + input_len + pred_len]
                self.samples.append((input_seq, target_seq))

        # 划分 train/val
        split_idx = int(len(self.samples) * train_ratio)
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class ElectricityDataset(Dataset):
    def __init__(self, csv_path, input_len=96, pred_len=24, split='train', train_ratio=0.8, normalize=True):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.normalize = normalize

        # 读取数据
        df = pd.read_csv(csv_path)

        # 假设第0列是时间戳，最后一列是 OT（目标）
        self.timestamps = df.iloc[:, 0]
        data = df.iloc[:, 1:].astype(float)  # 去掉时间戳，仅保留数值列

        self.n_features = data.shape[1]

        # 初始化归一化器
        if self.normalize:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            data.iloc[:, :-1] = self.scaler.fit_transform(data.iloc[:, :-1])  # 对输入特征归一化

        # 滑动窗口切片
        total_len = input_len + pred_len
        self.samples = []
        for i in range(len(data) - total_len + 1):
            input_seq = data.iloc[i : i + input_len].values  # [input_len, C]
            target_seq = data.iloc[i + input_len : i + input_len + pred_len, -1].values  # [pred_len], 只取 OT 列
            self.samples.append((input_seq, target_seq))

        # 划分 train/val
        split_idx = int(len(self.samples) * train_ratio)
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        #不要在这里反归一
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
