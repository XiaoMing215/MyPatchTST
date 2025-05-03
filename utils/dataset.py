import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

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
