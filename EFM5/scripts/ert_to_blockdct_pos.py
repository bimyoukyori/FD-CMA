#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ert_to_blockdct_pos.py
==================================================
ERT LWF 规则网格 → 分块 2D-DCT → 低频块 → 位置编码 → 特征

【消融分支】
- 不影响现有 ert_to_2Ddct.py
- 输出到 features/ert_blockdct_pos/

输入:
  data/ert/ert_lwf/{ID}_lwf.dat

输出:
  features/ert_blockdct_pos/{ID}.npy
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from scipy.fft import dctn
from math import pi, sin, cos

from dct_config import DCT_CFG


# ==================================================
# 参数读取
# ==================================================
CFG = DCT_CFG["ert_block"]
BLOCK = CFG["block_size"]
STRIDE = CFG["stride"]
KEEP = CFG["keep"]
POS_DIM = CFG["pos_dim"]
AGG = CFG["agg"]

PRINT_EVERY = 100


# ==================================================
# 读取 LWF 数据（完全复用你原有语义）
# ==================================================
def load_lwf_dat(path: Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    idx = 0
    while idx < len(lines):
        parts = lines[idx].strip().split()
        if len(parts) == 2:
            try:
                nx = int(parts[0])
                nz = int(parts[1])
                break
            except ValueError:
                pass
        idx += 1

    if idx >= len(lines):
        raise RuntimeError(f"[ERT BLOCK DCT] 未找到 nx nz: {path}")

    data_start = idx + 3
    data = np.zeros((nz, nx), dtype=np.float32)

    for line in lines[data_start:]:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            i = int(parts[0])
            j = int(parts[1])
            val = float(parts[4])
            if 0 <= i < nx and 0 <= j < nz:
                data[j, i] = val
        except ValueError:
            continue

    return data


# ==================================================
# 工具：2D-DCT
# ==================================================
def dct2(x: np.ndarray) -> np.ndarray:
    return dctn(x, type=2, norm="ortho")


# ==================================================
# 工具：2D sin-cos 位置编码
# ==================================================
def pos_encoding_2d(cx: float, cy: float, dim: int) -> np.ndarray:
    """
    cx, cy ∈ [0,1]
    dim 必须为偶数
    """
    assert dim % 2 == 0
    half = dim // 2
    enc = np.zeros(dim, dtype=np.float32)

    for i in range(half // 2):
        freq = 1.0 / (10000 ** (2 * i / half))
        enc[2 * i] = sin(2 * pi * cx * freq)
        enc[2 * i + 1] = cos(2 * pi * cx * freq)
        enc[half + 2 * i] = sin(2 * pi * cy * freq)
        enc[half + 2 * i + 1] = cos(2 * pi * cy * freq)

    return enc


# ==================================================
# 单样本处理
# ==================================================
def process_one(dat_path: Path, out_path: Path):
    grid = load_lwf_dat(dat_path)

# ==================================================
# [MOD] padding，确保至少能切出一个 block
# ==================================================
    H, W = grid.shape
    pad_h = max(0, BLOCK - H)
    pad_w = max(0, BLOCK - W)

    if pad_h > 0 or pad_w > 0:
        grid = np.pad(
            grid,
            ((0, pad_h), (0, pad_w)),
            mode="constant"
        )

    H, W = grid.shape


    feats = []

    for y in range(0, H - BLOCK + 1, STRIDE):
        for x in range(0, W - BLOCK + 1, STRIDE):
            patch = grid[y:y + BLOCK, x:x + BLOCK]
            coeff = dct2(patch)
            low = coeff[:KEEP, :KEEP].reshape(-1)

            # 块中心位置（归一化）
            cx = (x + BLOCK / 2) / W
            cy = (y + BLOCK / 2) / H
            pos = pos_encoding_2d(cx, cy, POS_DIM)

            feats.append(np.concatenate([low, pos], axis=0))

    #feats = np.stack(feats, axis=0)
    # ==================================================
    # [MOD] 分块特征聚合（论文级：不引入学习参数）
    # ==================================================
    feats = np.stack(feats, axis=0)     # (N_blocks, D_block)

    # 平均聚合：维度回落到固定长度
    feat = feats.mean(axis=0)           # (D_block,)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, feat.astype(np.float32))


# ==================================================
# 主程序
# ==================================================
def main():
    ROOT = Path(__file__).resolve().parents[1]
    ert_root = ROOT / "data" / "ert" / "ert_lwf"
    out_root = ROOT / "features" / "ert_blockdct_pos"

    files = sorted(ert_root.glob("*_lwf.dat"))
    total = len(files)
    processed = 0

    print(f"[ERT BLOCK DCT] Input  : {ert_root}")
    print(f"[ERT BLOCK DCT] Output : {out_root}")

    for dat_path in files:
        sid = dat_path.stem.replace("_lwf", "")
        out_path = out_root / f"{sid}.npy"
        process_one(dat_path, out_path)
        processed += 1

        if processed % PRINT_EVERY == 0 or processed == total:
            print(f"[ERT BLOCK DCT] processed {processed}/{total}")

    print("[ERT BLOCK DCT] DONE")


if __name__ == "__main__":
    main()
