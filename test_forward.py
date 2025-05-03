import torch
from models.patchtst import PatchTST

def main():
    # 模拟输入数据：32 条序列，每条长度为 96，每个时间点 7 个特征
    batch_size = 32
    seq_len = 96
    n_features = 7

    x = torch.randn(batch_size, seq_len, n_features)  # [B, L, C]

    # 实例化模型
    model = PatchTST(
        input_len=96,
        pred_len=24,
        patch_len=16,
        stride=8,
        d_model=64,
        n_heads=4,
        d_ff=128,
        num_layers=3,
        dropout=0.1,
        n_features=n_features
    )

    # 前向传播测试
    out = model(x)
    print("输出 shape:", out.shape)  # 应该是 [32, 24, 7]

if __name__ == "__main__":
    main()
