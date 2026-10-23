#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configurable image + temporal encoders for HCP hybrid deep model (same I/O as HCPConvNeXtClassifier)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, resnet18, swin_t

from train_hcp_convnextv2 import ConvNeXtV2Encoder, TimeCourseEncoder
from hcp_mamba_minimal import MambaStack


def _norm_aliases(name: str) -> str:
    n = name.strip().lower().replace("-", "_")
    aliases = {
        "convnext": "convnextv2",
        "resnet": "resnet18",
        "swin": "swin_t",
        "efficientnet": "efficientnet_b0",
        "efficient_net": "efficientnet_b0",
        "simple_cnn": "cnn",
        "plain_cnn": "cnn",
    }
    return aliases.get(n, n)


class ResNet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        w = resnet18(weights=None)
        w.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.body = nn.Sequential(*list(w.children())[:-1])
        self.feat_dim = 512

    def forward(self, x):
        return self.body(x).flatten(1)


class SwinTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        m = swin_t(weights=None)
        m.features[0][0] = nn.Conv2d(1, 96, kernel_size=4, stride=4, bias=False)
        self.features = m.features
        self.norm = m.norm
        self.permute = m.permute
        self.avgpool = m.avgpool
        self.flatten = m.flatten
        self.feat_dim = m.head.in_features

    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        return self.flatten(x)


class EfficientNetB0Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        m = efficientnet_b0(weights=None)
        m.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.features = m.features
        self.avgpool = m.avgpool
        self.feat_dim = m.classifier[1].in_features

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class SimpleCNNEncoder(nn.Module):
    """Light baseline CNN on 64x64 single-channel."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.feat_dim = 256

    def forward(self, x):
        return self.net(x).flatten(1)


class GRUTimeCourseEncoder(nn.Module):
    def __init__(self, hidden: int = 128, num_layers: int = 2):
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = nn.Sequential(nn.Linear(hidden * 2, 64), nn.Tanh(), nn.Linear(64, 1))
        self.feat_dim = hidden * 2

    def forward(self, x, seq_len):
        out, _ = self.gru(x)
        attn_score = self.attn(out).squeeze(-1)
        device = x.device
        max_len = x.size(1)
        mask = torch.arange(max_len, device=device).unsqueeze(0) < seq_len.unsqueeze(1)
        attn_score = attn_score.masked_fill(~mask, -1e9)
        w = torch.softmax(attn_score, dim=1).unsqueeze(-1)
        return torch.sum(w * out, dim=1)


class MambaTimeCourseEncoder(nn.Module):
    def __init__(self, d_model: int = 128, n_layer: int = 2, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.core = MambaStack(d_model=d_model, n_layer=n_layer, d_state=d_state, expand=expand)
        self.attn = nn.Sequential(nn.Linear(d_model, 64), nn.Tanh(), nn.Linear(64, 1))
        self.feat_dim = d_model

    def forward(self, x, seq_len):
        h = self.core(self.proj(x))
        attn_score = self.attn(h).squeeze(-1)
        device = x.device
        max_len = x.size(1)
        mask = torch.arange(max_len, device=device).unsqueeze(0) < seq_len.unsqueeze(1)
        attn_score = attn_score.masked_fill(~mask, -1e9)
        w = torch.softmax(attn_score, dim=1).unsqueeze(-1)
        return torch.sum(w * h, dim=1)


class TransformerTimeCourseEncoder(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 4, nlayers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.attn = nn.Sequential(nn.Linear(d_model, 64), nn.Tanh(), nn.Linear(64, 1))
        self.feat_dim = d_model

    def forward(self, x, seq_len):
        h = self.input_proj(x)
        b, t, _ = h.shape
        pad = torch.arange(t, device=x.device).unsqueeze(0) >= seq_len.unsqueeze(1)
        h = self.enc(h, src_key_padding_mask=pad)
        attn_score = self.attn(h).squeeze(-1)
        attn_score = attn_score.masked_fill(pad, -1e9)
        w = torch.softmax(attn_score, dim=1).unsqueeze(-1)
        return torch.sum(w * h, dim=1)


def build_spatial_encoder(name: str) -> nn.Module:
    name = _norm_aliases(name)
    if name == "convnextv2":
        return ConvNeXtV2Encoder()
    if name == "resnet18":
        return ResNet18Encoder()
    if name == "swin_t":
        return SwinTEncoder()
    if name == "efficientnet_b0":
        return EfficientNetB0Encoder()
    if name == "cnn":
        return SimpleCNNEncoder()
    raise ValueError(f"unknown image backbone {name!r}; try convnextv2|resnet18|swin_t|efficientnet_b0|cnn")


def build_temporal_encoder(name: str) -> nn.Module:
    name = _norm_aliases(name)
    if name == "lstm":
        return TimeCourseEncoder()
    if name == "gru":
        return GRUTimeCourseEncoder()
    if name == "mamba":
        return MambaTimeCourseEncoder()
    if name == "transformer":
        return TransformerTimeCourseEncoder()
    raise ValueError(f"unknown temporal backbone {name!r}; try lstm|gru|mamba|transformer")


class HCPHybridClassifier(nn.Module):
    """Drop-in replacement for HCPConvNeXtClassifier (forward returns dict with key predict)."""

    def __init__(self, image_backbone: str = "convnextv2", temporal_backbone: str = "lstm"):
        super().__init__()
        self.image_backbone_name = _norm_aliases(image_backbone)
        self.temporal_backbone_name = _norm_aliases(temporal_backbone)
        self.spatial = build_spatial_encoder(self.image_backbone_name)
        self.temporal = build_temporal_encoder(self.temporal_backbone_name)
        dim_s = self.spatial.feat_dim
        dim_t = self.temporal.feat_dim
        total = dim_s + dim_t
        self.head = nn.Sequential(
            nn.Linear(total, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def encode_embeddings(self, batch):
        """L2-normalized spatial/temporal vectors (same as used inside forward before head)."""
        f_s = F.normalize(self.spatial(batch["image"]), dim=1)
        f_t = F.normalize(self.temporal(batch["time_course"], batch["seq_len"]), dim=1)
        return f_s, f_t

    def forward(self, batch):
        f_s, f_t = self.encode_embeddings(batch)
        feat = torch.cat([f_s, f_t], dim=1)
        return {"predict": self.head(feat)}


def build_hcp_classifier(image_backbone: str, temporal_backbone: str) -> nn.Module:
    return HCPHybridClassifier(image_backbone=image_backbone, temporal_backbone=temporal_backbone)


def resolve_deep_arch_tags(image_backbone: str, temporal_backbone: str) -> dict:
    """Return canonical backbone names (after alias normalization)."""
    m = HCPHybridClassifier(image_backbone, temporal_backbone)
    return {"image_backbone": m.image_backbone_name, "temporal_backbone": m.temporal_backbone_name}
