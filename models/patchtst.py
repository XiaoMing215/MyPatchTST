import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)  # 学习位置编码的层
    def forward(self, x):
        B, N, _ = x.shape
        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        return self.position_embeddings(positions)

class PatchEmbedding(nn.Module):
    def __init__(self, patch_num, patch_len, stride, input_dim, d_model,dropout):
        super().__init__()
        self.patch_len = patch_len
        self.patch_num = patch_num
        self.stride = stride
        self.input_dim = input_dim
        self.d_model = d_model
        self.proj = nn.Linear(patch_len * input_dim, d_model) #线性变换更改为向量便于下一步处理
        self.position_embedding = PositionalEmbedding(d_model, max_len=patch_num)
        self.padding = stride
        self.padding = nn.ReplicationPad1d((0, self.padding))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, C] batch 时间步 通道数
        B, L, C = x.shape
        patch_num = (L - self.patch_len) // self.stride + 1
        x = self.padding(x)
        x = x.permute(0, 2, 1)  # [B, L, C] 转换为 [B, C, L]
        x = x.unfold(dimension=2, size=self.patch_len, step=self.stride)  # [B, C, patch_num, patch_len]
        x = x.permute(0, 2, 3, 1)  # [B, patch_num, patch_len, C]
        x = x.reshape(B, patch_num, -1)  # [B, patch_num, patch_len * C]
        x = self.proj(x)  # [B, patch_num, d_model]
        position_emb = self.position_embedding(x)  # [B, patch_num, d_model]
        x = x + position_emb

        return self.dropout(x)

class PatchTransformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, num_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [B, N_patch, d_model]
        return self.encoder(x)

class PatchTSThead(nn.Module):
    def __init__(self, d_model, patch_num, pred_len, num_vars):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(d_model * patch_num, pred_len) for _ in range(num_vars)
        ])

    def forward(self, x):
        # x: [B, N_patch, d_model]
        B, N, D = x.shape
        x = x.reshape(B, -1)  # [B, N_patch * d_model]
        out = torch.stack([head(x) for head in self.heads], dim=1)  # [B, C, pred_len]
        return out

class PatchTST(nn.Module):
    def __init__(self, input_len, pred_len, patch_len, stride,
                 d_model, n_heads, d_ff, num_layers, dropout, n_features):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.n_features = n_features

        patch_num = (input_len - patch_len) // stride + 1
        self.patch_embed = PatchEmbedding(patch_num, patch_len, stride, n_features, d_model,dropout)
        self.transformer = PatchTransformer(d_model, n_heads, d_ff, dropout, num_layers)

        self.head = PatchTSThead(d_model, patch_num, pred_len, n_features)

    def forward(self, x):
        # x: [B, L, C]
        x = self.patch_embed(x)    # [B, N_patch, d_model]
        x = self.transformer(x)    # [B, N_patch, d_model]
        out = self.head(x)         # [B, C, pred_len]
        return out.permute(0, 2, 1)  # [B, pred_len, C]，与输入风格保持一致
