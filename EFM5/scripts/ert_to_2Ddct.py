#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ert_to_2Ddct.py
==================================================
ERT 二维数值场 (.npy) → 2D DCT → 低频块 → 一维特征

输入:
  data/ert/ert_npy/{ID}.npy

输出:
  features/ert/{ID}.npy
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from scipy.fft import dctn


# ==========================
# 全局参数（与 GPR 完全一致）
# ==========================
KX = 32
KY = 32
PRINT_EVERY = 100


def pad_to_min_size(x: np.ndarray, min_x: int, min_y: int) -> np.ndarray:
    """若二维矩阵小于 KX×KY，进行 zero-padding"""
    h, w = x.shape
    pad_h = max(0, min_x - h)
    pad_w = max(0, min_y - w)
    if pad_h > 0 or pad_w > 0:
        x = np.pad(x, ((0, pad_h), (0, pad_w)), mode="constant")
    return x


def dct2(x: np.ndarray) -> np.ndarray:
    return dctn(x, type=2, norm="ortho")


def process_one(npy_path: Path, out_path: Path):
    # ---------- 1. 读取二维 ERT 数值场 ----------
    data = np.load(npy_path).astype(np.float32)

    # === [MOD] 数值闭合（必须）===
    if not np.isfinite(data).all():
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
    if data.ndim != 2:
        raise ValueError(f"[ERT DCT] 非二维数据: {npy_path}")

    # ---------- 2. padding ----------
    data = pad_to_min_size(data, KX, KY)

    # ---------- 3. 2D DCT ----------
    coeff = dct2(data)

    # ---------- 4. 低频块 ----------
    feat = coeff[:KX, :KY].reshape(-1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, feat)


def main():
    ROOT = Path(__file__).resolve().parents[1]
    ert_root = ROOT / "data" / "ert" / "ert_npy"
    out_root = ROOT / "features" / "ert"

    files = sorted(ert_root.glob("*.npy"))
    total = len(files)
    processed = 0
    failed = 0

    print(f"[ERT DCT] Input root  : {ert_root}")
    print(f"[ERT DCT] Output root : {out_root}")

    for npy_path in files:
        raw = npy_path.stem
        sid = raw.split("_")[1]  # 00001
        out_path = out_root / f"{sid}.npy"

        try:
            process_one(npy_path, out_path)
            processed += 1
        except Exception as e:
            print(f"[ERT DCT][FAIL] {sid}: {e}")
            failed += 1
            continue

        if processed % PRINT_EVERY == 0 or processed == total:
            print(f"[ERT DCT] processed {processed} / {total}")

    print(f"[ERT DCT] DONE | total={processed} | failed={failed}")
    print(f"[ERT DCT] Features saved to : {out_root}")


if __name__ == "__main__":
    main()
