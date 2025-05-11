import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)#注册固定的张量，不会参与梯度计算。

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class PatchEmbedding(nn.Module):#输入应该是x:[B,C(input_len),L(features)] 一个样例：[32,96,7]
    def __init__(self,patch_len, stride, input_dim, d_model,dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.input_dim = input_dim
        self.d_model = d_model
        self.proj = nn.Linear(patch_len, d_model) #线性变换更改为64维 (两个参数：输入特征维度及输出维度，只作用于最后一维)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def get_required_padding(self,input_len, patch_len, stride):
        remainder = (input_len - patch_len) % stride
        if remainder == 0:
            return 0
        else:
            return stride - remainder
    def forward(self, x):
        #x:[B,L(input_len),C(channels)]
        #不需要额外传入patchnum
        # print(x.shape) #32,96,7
        B, L, C = x.shape
        #动态生成padding
        padding_len = self.get_required_padding(L, self.patch_len, self.stride)
        print(padding_len)
        padding_model = nn.ReplicationPad1d((0, padding_len))
        x = x.permute(0, 2, 1)  # [B, C, L] -> [B, L, C]
        x = padding_model(x) #填充最后一维 时间维度
        x = x.unfold(dimension=2, size=self.patch_len, step=self.stride)  #这一步是把input len分成patch和stride [B, L, patch_num, patch_len]
        # print(x.shape) #32,7，12,16
        x = x.reshape(B*C,x.shape[2],x.shape[3])  # [B, patch_num, patch_len * C] 224 11 16
        x = self.proj(x)  # [B, patch_num, d_model] 224 11 64
        position_emb = self.position_embedding(x)  # [B, patch_num, d_model]
        x = x + position_emb
        print(f"输出维度：{x.shape}") #224，11, 64 [B*C patch_num patch_len]

        return self.dropout(x)

#输入到encoder编码器之前的特征维度[(batch_size*channel)，patch_num，d_model]
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

class FlattenHead(nn.Module):
    def __init__(self, nf, pred_len, head_dropout=0):
        #nf:d_model × patch_num 可以不用传入 但是为了少耦合就选择了传入
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2) #d_model 和 patch_num合并
        self.linear = nn.Linear(nf, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [B, C, d_model, patch_num]
        x = self.flatten(x) #[B, C, nf]
        x = self.linear(x) #[B, C, nf->pred_len]
        #这个 linear 就是输出层，负责把最后一层表示映射成预测值。
        x = self.dropout(x)
        return x


class PatchTST(nn.Module):
    def __init__(self, input_len, pred_len, patch_len, stride,
                 d_model, n_heads, d_ff, num_layers, dropout, n_features):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.n_features = n_features

        patch_num = (input_len - patch_len) // stride + 1
        self.patch_embed = PatchEmbedding(patch_len, stride, n_features, d_model,dropout)
        self.transformer = PatchTransformer(d_model, n_heads, d_ff, dropout, num_layers)

        self.head = PatchTSThead(d_model, patch_num, pred_len, n_features)

    def forward(self, x):
        # x: [B, L, C]
        x = self.patch_embed(x)    # [B, N_patch, d_model]
        x = self.transformer(x)    # [B, N_patch, d_model]
        out = self.head(x)         # [B, C, pred_len]
        return out.permute(0, 2, 1)  # [B, pred_len, C]，与输入风格保持一致
