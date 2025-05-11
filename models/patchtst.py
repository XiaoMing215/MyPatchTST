import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

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
        x = x.permute(0, 2, 1)  # [B, C, L] -> [B, L, C]
        padding_len = self.get_required_padding(L, self.patch_len, self.stride)
        # print(padding_len)
        padding_model = nn.ReplicationPad1d((0, padding_len))
        x = padding_model(x) #填充最后一维 时间维度
        x = x.unfold(dimension=2, size=self.patch_len, step=self.stride)  #这一步是把input len分成patch和stride [B, L, patch_num, patch_len]
        # print(x.shape) #32,7，12,16
        x = x.reshape(B*C,x.shape[2],x.shape[3])  # [B, patch_num, patch_len * C] 224 11 16
        x = self.proj(x)  # [B, patch_num, d_model] 224 11 64
        position_emb = self.position_embedding(x)  # [B, patch_num, d_model]
        x = x + position_emb
        # print(f"输出维度：{x.shape}") #224，11, 64 [B*C patch_num patch_len]

        return self.dropout(x)


# class PatchTransformer(nn.Module):
#     def __init__(self, d_model, n_heads, d_ff, dropout, num_layers):
#         super().__init__()
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=n_heads,
#             dim_feedforward=d_ff,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#
#     def forward(self, x):
#         # x: [B, N_patch, d_model]
#         return self.encoder(x)

class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        # print('queries.shape:',queries.shape)
        # print('values.shape:',values.shape)
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

#输入到encoder的特征维度x：[(batch_size*channel)，patch_num，d_model]
class PatchTST(nn.Module):
    # def __init__(self, config, patch_len=16, stride=8):
    def __init__(self, config):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super(PatchTST, self).__init__()
        self.seq_len = config['input_len']
        self.pred_len = config['pred_len']
        self.patch_len = config['patch_len']
        self.stride = config['stride']
        self.d_model = config['d_model']
        padding = self.stride

        # patching and embedding
        #(self,patch_len, stride, input_dim, d_model,dropout)
        self.patch_embedding = PatchEmbedding(
            self.patch_len,self.stride,self.seq_len,config['d_model'],config['dropout']
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            config['factor'],
                            attention_dropout=config['dropout'],
                            output_attention=config['output_attention'],
                        ),
                        config['d_model'],
                        config['n_heads'],
                    ),
                    config['d_model'],
                    config['d_ff'],
                    dropout=config['dropout'],
                    activation=config['activation'],
                )
                for l in range(config['num_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(config['d_model']),
        )

        # Prediction Head
        self.head_nf = config['d_model'] * int((config['input_len'] - self.patch_len) / self.stride + 1)
        # print(self.head_nf)
        self.head = FlattenHead(
            self.head_nf,
            config['pred_len'],
            head_dropout=config['dropout'],
        )

    def forward(self, x):
        patch_num = int((x.shape[1] - self.patch_len) / self.stride + 2)
        # print(x.shape)
        n_var = x.shape[-1]
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
        # print(x.shape)

        x = self.patch_embedding(x)
        # print(x.shape)
        x, attns = self.encoder(x)
        # print('x,attns:',x.shape)
        nf,patch_num,d_model = x.shape
        x = x.reshape(-1, n_var, patch_num, d_model)
        x = x.permute(0, 1, 3, 2)
        x = self.head(x)
        x = x.permute(0, 2, 1)
        x *= stdev
        x += means
        return x

class FlattenHead(nn.Module):
    def __init__(self, nf, pred_len, head_dropout=0):
        #nf:d_model × patch_num 可以不用传入 但是为了少耦合就选择了传入
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2) #d_model 和 patch_num合并
        self.linear = nn.Linear(nf, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [B, C, d_model, patch_num]
        # print("before flatten x shape:", x.shape)
        x = self.flatten(x) #[B, C, nf]
        x = self.linear(x) #[B, C, nf->pred_len]
        #这个 linear 就是输出层，负责把最后一层表示映射成预测值。
        x = self.dropout(x)
        return x

