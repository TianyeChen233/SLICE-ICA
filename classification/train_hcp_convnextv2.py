#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================
# LayerNorm2d
# ======================
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x


# ======================
# GRN
# ======================
class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


# ======================
# ConvNeXt Block
# ======================
class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pw2 = nn.Conv2d(4 * dim, dim, 1)

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pw2(x)
        return x + identity


# ======================
# Spatial Encoder
# ======================
class ConvNeXtV2Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        dims = [96, 192, 384, 768]
        depths = [3, 3, 9, 3]

        self.down = nn.ModuleList()
        self.down.append(nn.Sequential(
            nn.Conv2d(1, dims[0], 4, stride=4),
            LayerNorm2d(dims[0])
        ))

        for i in range(3):
            self.down.append(nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], 2, stride=2)
            ))

        self.stages = nn.ModuleList()
        for i in range(4):
            blocks = [ConvNeXtV2Block(dims[i]) for _ in range(depths[i])]
            self.stages.append(nn.Sequential(*blocks))

        self.norm = LayerNorm2d(dims[-1])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.feat_dim = dims[-1]

    def forward(self, x):
        for i in range(4):
            x = self.down[i](x)
            x = self.stages[i](x)
        x = self.norm(x)
        x = self.pool(x).flatten(1)
        return x


# ======================
# Temporal Encoder（修复版）
# ======================
class TimeCourseEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = 128

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.attn = nn.Sequential(
            nn.Linear(self.hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.feat_dim = self.hidden * 2

    def forward(self, x, seq_len):
        lstm_out, _ = self.lstm(x)

        attn_score = self.attn(lstm_out).squeeze(-1)  # [B,T]

        # ===== mask（关键修复）=====
        device = x.device
        max_len = x.size(1)

        mask = torch.arange(max_len, device=device).unsqueeze(0) < seq_len.unsqueeze(1)
        attn_score = attn_score.masked_fill(~mask, -1e9)

        attn_weight = torch.softmax(attn_score, dim=1).unsqueeze(-1)
        feat = torch.sum(attn_weight * lstm_out, dim=1)

        return feat


# ======================
# FINAL MODEL
# ======================
class HCPConvNeXtClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.spatial = ConvNeXtV2Encoder()
        self.temporal = TimeCourseEncoder()

        dim_s = self.spatial.feat_dim
        dim_t = self.temporal.feat_dim

        total_dim = dim_s + dim_t

        # ===== gating =====
        self.gate = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, total_dim),
            nn.Sigmoid()
        )

        self.head = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, batch):
        x_img = batch['image']
        x_tc = batch['time_course']
        seq_len = batch['seq_len']

        f_s = self.spatial(x_img)
        f_t = self.temporal(x_tc, seq_len)

        # normalize（稳定训练）
        f_s = F.normalize(f_s, dim=1)
        f_t = F.normalize(f_t, dim=1)

        feat = torch.cat([f_s, f_t], dim=1)

        # # gating
        # g = self.gate(feat)
        # feat = feat * g

        logits = self.head(feat)

        return {
            "predict": logits
        }


# ======================
# 测试
# ======================
if __name__ == "__main__":
    model = HCPConvNeXtClassifier()

    batch = {
        "image": torch.randn(2, 1, 64, 64),
        "time_course": torch.randn(2, 300, 1),
        "seq_len": torch.tensor([300, 250])
    }

    out = model(batch)
    print("OK, output shape:", out["predict"].shape)
