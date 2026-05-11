#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fusion_net.py

功能说明
--------
ERT–GPR 双分支融合网络（FusionNet）：

- ERT 分支：MLP，输入 1D 特征（ERT 2D-DCT / block-DCT / pos 编码等）
- GPR 分支：轻量 CNN，输入 2D DCT 频谱（B, 1, H, W）
- 融合方式：拼接后线性投影
- 训练目标：InfoNCE 跨模态对齐 + 可选分类监督

工程定位
--------
- 本网络仅用于 CLD-Fusion（跨模态对齐）
- GPR 分支导出的权重将作为 RT-DETR 蒸馏的 teacher backbone
- 不在此处处理目标检测、不处理 RT-DETR
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ERT 分支：MLP（1D 特征）
# ============================================================

class ERTMLP(nn.Module):
    """
    ERT 特征编码器（MLP）

    输入：
        x: (B, D_ert)

    输出：
        (B, emb_dim)
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 512,
        out_dim: int = 256,
        p_drop: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(p_drop),
            nn.Linear(hidden, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# GPR 分支：CNN（2D DCT 频谱）
# ============================================================

class GPRCNN(nn.Module):
    """
    GPR 特征编码器（轻量 CNN）

    输入：
        x: (B, 1, H, W)
        - H, W 通常为 32×32 / 64×64 的 2D-DCT 频谱块

    输出：
        (B, emb_dim)
    """

    def __init__(
        self,
        in_ch: int = 1,
        base: int = 32,
        out_dim: int = 256,
    ):
        super().__init__()

        C = base

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.SiLU(),
            nn.MaxPool2d(2),  # H/2
        )

        self.b1 = nn.Sequential(
            nn.Conv2d(C, C * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(C * 2),
            nn.SiLU(),
            nn.MaxPool2d(2),  # H/4
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(C * 2, C * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(C * 4),
            nn.SiLU(),
            nn.MaxPool2d(2),  # H/8
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(C * 4, C * 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(C * 8),
            nn.SiLU(),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(C * 8, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.head(x)
        return x


# ============================================================
# 融合网络
# ============================================================

class FusionNet(nn.Module):
    """
    ERT–GPR 融合网络

    输出：
        e1     : ERT 归一化 embedding (B, D)
        e2     : GPR 归一化 embedding (B, D)
        z      : 融合特征 (B, D)
        logits : 分类输出（若 num_classes > 1）
    """

    def __init__(
        self,
        ert_dim: int,
        num_classes: int = 1,
        emb_dim: int = 256,
    ):
        super().__init__()

        # ERT 分支（1D）
        self.ert_branch = ERTMLP(
            in_dim=ert_dim,
            hidden=512,
            out_dim=emb_dim,
        )

        # GPR 分支（2D CNN）
        self.gpr_branch = GPRCNN(
            in_ch=1,
            base=32,
            out_dim=emb_dim,
        )

        # 融合投影
        self.proj = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.GELU(),
            nn.LayerNorm(emb_dim),
        )

        # 分类头（可选）
        self.classifier = nn.Linear(emb_dim, num_classes)

    # --------------------
    # 编码接口
    # --------------------

    def encode_ert(self, x1: torch.Tensor) -> torch.Tensor:
        """
        ERT 编码并 L2 归一化
        """
        return F.normalize(self.ert_branch(x1), dim=-1)

    def encode_gpr(self, x2: torch.Tensor) -> torch.Tensor:
        """
        GPR 编码并 L2 归一化
        """
        return F.normalize(self.gpr_branch(x2), dim=-1)

    # --------------------
    # 前向
    # --------------------

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        e1 = self.encode_ert(x1)
        e2 = self.encode_gpr(x2)

        z = self.proj(torch.cat([e1, e2], dim=-1))
        logits = self.classifier(z)

        return z, logits, e1, e2


# ============================================================
# 对称 InfoNCE（备用）
# ============================================================

def info_nce(
    e1: torch.Tensor,
    e2: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    对称 InfoNCE 损失（未加权版本）

    e1, e2: (B, D)，需已 L2 归一化
    """
    assert e1.shape == e2.shape, "InfoNCE 输入 shape 不一致"

    B = e1.size(0)
    sim = (e1 @ e2.t()) / temperature
    target = torch.arange(B, device=e1.device)

    loss_a = F.cross_entropy(sim, target)
    loss_b = F.cross_entropy(sim.t(), target)

    return 0.5 * (loss_a + loss_b)
