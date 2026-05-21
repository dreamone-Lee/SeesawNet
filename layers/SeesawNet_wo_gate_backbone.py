import copy
import math

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from layers.SelfAttention_Family import TSMixer, ResAttention, _MultiheadAttention

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class TSEncoder(nn.Module):
    def __init__(self, attn_layers):
        super(TSEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)

    def forward(self, x, xx):
        """
        input
            x: [B, L, D], xx: [B, L, D]
        output
            x: [B, L, D], xx: [B, L, D]
        """
        for attn_layer in self.attn_layers:
            x, xx = attn_layer(x, xx)
        return x, xx

class MixAttention(nn.Module):
    def __init__(self, token_num, dropout=0.1, d_model=256, n_heads=8):
        super().__init__()
        self.attn1 = TSMixer(ResAttention(attention_dropout=0., d_model=d_model, n_heads=n_heads), 
                             d_model=d_model, n_heads=n_heads, proj_dropout=dropout)
        self.attn2 = TSMixer(ResAttention(attention_dropout=0., d_model=d_model, n_heads=n_heads), 
                             d_model=d_model, n_heads=n_heads, proj_dropout=dropout)
        # self.gate = nn.Sequential(nn.Conv1d(token_num * 2, token_num, 1), nn.Sigmoid())
        # Todo: consider using a Norm
    
    def forward(self, x, xx):
        """
        input
            x: [B, L, D], xx: [B, L, D]
        output
            out: [B, L, D]
        """
        # gate_weight = self.gate(torch.cat([x, xx], dim=-2))
        out1, A1 = self.attn1(x, x, x)
        out2, A2 = self.attn2(xx, xx, x)
        # out = (1 - gate_weight) * out1 + gate_weight * out2
        out = (out1 + out2) / 2
        return out

class MixGroupAttention(nn.Module):
    def __init__(self, token_num, dropout=0.1, d_model=256, n_heads=8):
        super().__init__()
        self.token_group = math.ceil(token_num ** 0.5)
        self.pad_channel = nn.ConstantPad1d((0, self.token_group ** 2 - token_num), 0)
        self.atten1 = MixAttention(self.token_group, dropout=dropout, d_model=d_model, n_heads=n_heads)
        self.atten2 = MixAttention(self.token_group, dropout=dropout, d_model=d_model, n_heads=n_heads)

    def forward(self, x, xx):
        """
        input
            x: [B, L, D], xx: [B, L, D]
        output
            out: [B, L, D]
        """
        b, l, d = x.shape
        g = r = self.token_group

        x = self.pad_channel(x.transpose(-1, -2)).transpose(-1, -2)
        x = rearrange(x, 'b (r g) d -> (b r) g d', g=g, r=r)
        xx = self.pad_channel(xx.transpose(-1, -2)).transpose(-1, -2)
        xx = rearrange(xx, 'b (r g) d -> (b r) g d', g=g, r=r)
        
        out1 = self.atten1(x, xx)
        out1 = rearrange(out1, '(b r) g d -> b (r g) d', g=g, r=r)[:, :l, :]
        
        x = rearrange(x, '(b r) g d -> (b g) r d', g=g, r=r)
        xx = rearrange(xx, '(b r) g d -> (b g) r d', g=g, r=r)

        out2 = self.atten2(x, xx)
        out2 = rearrange(out2, '(b g) r d -> b (r g) d', g=g, r=r)[:, :l, :]

        out = (out1 + out2) / 2
        return out

class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation="gelu", norm="LayerNorm"):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        if norm == "BatchNorm":
            self.norm = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        """
        input
            x: [B, L, D]
        output
            out: [B, L, D]
        """
        return self.norm(x1 + self.ffn(x2))

class MixEncoder(nn.Module):
    def __init__(self,token_num, d_model, d_ff, n_heads, dropout=0., activation="gelu", norm="LayerNorm", group=False):
        super().__init__()    
        if group:
            self.attention = MixGroupAttention(token_num, dropout=dropout, d_model=d_model, n_heads=n_heads)
        else:
            self.attention = MixAttention(token_num, dropout=dropout, d_model=d_model, n_heads=n_heads)
        self.ffn1 = FFN(d_model, d_ff, dropout, activation, norm)
        self.ffn2 = FFN(d_model, d_ff, dropout, activation, norm)

        if norm == "BatchNorm":
            self.norm1 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm2 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, xx):
        """
        input
            x: [B, L, D], xx: [B, L, D]
        output
            x: [B, L, D], xx: [B, L, D]
        """
        out = self.attention(x, xx)
        x = self.norm1(x + self.dropout(out)) 
        x = self.ffn1(x, x)
        xx = self.ffn2(xx, out)
        return x, xx


class PD_Attention_layer(nn.Module):
    def __init__(self, patch_num, d_model, d_ff, n_heads, dropout=0., activation="gelu", norm="LayerNorm", group=False):
        super().__init__()
        self.encoder = MixEncoder(patch_num, d_model, d_ff, n_heads, dropout, activation, norm, group=group)
    
    def forward(self, x, xx):
        """
        input
            x: [B, N, P, D], xx: [B, N, P, D]
        output
            x: [B, N, P, D], xx: [B, N, P, D]
        """
        b, n, p, d = x.shape
        x = rearrange(x, 'b n p d -> (b n) p d')
        xx = rearrange(xx, 'b n p d -> (b n) p d')
        x, xx = self.encoder(x, xx)
        x = rearrange(x, '(b n) p d -> b n p d', b=b, n=n)
        xx = rearrange(xx, '(b n) p d -> b n p d', b=b, n=n)
        return x, xx

class aggregation_layer(nn.Module):
    def __init__(self, patch_num, aggr_num, d_model, n_heads, dropout=0., activation="gelu", norm="LayerNorm"):
        super().__init__()
        bias = True
        self.proj1 = nn.Conv1d(patch_num, aggr_num, 1, bias=bias)
        self.proj2 = nn.Conv1d(patch_num, aggr_num, 1, bias=bias)
        self.aggr_num = aggr_num
    
    def forward(self, x, xx):
        """
        input
            x: [B, N, P, D], xx: [B, N, P, D]
        output
            x: [B, N, A, D], xx: [B, N, A, D]
        """
        b, n, p, d = x.shape
        x = rearrange(x, 'b n p d -> (b n) p d')
        xx = rearrange(xx, 'b n p d -> (b n) p d')
        x = self.proj1(x)
        xx = self.proj2(xx)
        x = rearrange(x, '(b n) a d -> b n a d', b=b, n=n, a=self.aggr_num)
        xx = rearrange(xx, '(b n) a d -> b n a d', b=b, n=n, a=self.aggr_num)
        return x, xx

class CR_Attention_layer(nn.Module):
    def __init__(self, channel, d_model, d_ff, n_heads, dropout=0., activation="gelu", norm="LayerNorm", group=False):
        super().__init__()
        self.encoder = MixEncoder(channel, d_model, d_ff, n_heads, dropout, activation, norm, group=group)

    def forward(self, x, xx):
        """
        input
            x: [B, N, A, D], xx: [B, N, A, D]
        output
            x: [B, N, A, D], xx: [B, N, A, D]
        """
        b, n, a, d = x.shape
        x = rearrange(x, 'b n a d -> (b a) n d')
        xx = rearrange(xx, 'b n a d -> (b a) n d')
        x, xx = self.encoder(x, xx)
        x = rearrange(x, '(b a) n d -> b n a d', b=b, a=a)
        xx = rearrange(xx, '(b a) n d -> b n a d', b=b, a=a)
        return x, xx