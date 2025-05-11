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



class ETThDataset(Dataset):
    def __init__(self, csv_path, input_len=96, pred_len=24, split='train', train_ratio=0.8, normalize=True):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.normalize = normalize

        print(f"正在加载数据集：{csv_path}")
        df = pd.read_csv(csv_path, sep=',')

        self.timestamps = df.iloc[:, 0]
        data = df.iloc[:, 1:].astype(float)
        print("data 前几行：\n", data.head())
        print("data 类型：", type(data))
        self.n_features = data.shape[1]
        print(f"数据包含 {self.n_features} 个数值特征（不含时间列）")

        # 归一化（对最后一列 OT 也需要归一化）
        if self.normalize:
            self.scaler = MinMaxScaler()
            before_norm = data.head(2).copy()
            data.iloc[:, :] = self.scaler.fit_transform(data.iloc[:, :])
            print("归一化前前两行：\n", before_norm)
            print("归一化后前两行：\n", data.head(2))

        # 滑动窗口采样
        total_len = input_len + pred_len
        self.samples = []
        for i in range(len(data) - total_len + 1):
            input_seq = data.iloc[i : i + input_len].values
            target_seq = data.iloc[i + input_len : i + input_len + pred_len].values
            self.samples.append((input_seq, target_seq))

        # 数据集划分
        split_idx = int(len(self.samples) * train_ratio)
        if split == 'train':
            self.samples = self.samples[:split_idx]
            print(f"加载训练集，共 {len(self.samples)} 个样本")
        else:
            self.samples = self.samples[split_idx:]
            print(f"加载验证集，共 {len(self.samples)} 个样本")

        # 输出一个样本的形状确认
        x0, y0 = self.samples[0]
        # print(f"示例输入 x 的 shape: {x0.shape}，目标 y 的 shape: {y0.shape}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
