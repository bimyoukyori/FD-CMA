#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ert_to_2Ddct.py
==================================================
ERT LWF 规则网格 → 2D DCT → 低频块 → 一维特征

【输入策略说明｜2026-01 更新】
1. 若存在 data/pairs.csv：
   - 优先从 pairs.csv 中读取 ert_dat 字段
   - 仅对已对齐样本进行 DCT
2. 若不存在 pairs.csv：
   - 回退为扫描 data/ert/ert_lwf/*.dat 的旧行为

输出:
  features/ert/{ID}.npy
"""

from __future__ import annotations

import csv
import numpy as np
from pathlib import Path
from scipy.fft import dctn


# ==========================
# 全局参数（主线固定）
# ==========================
KX = 32
KY = 32
PRINT_EVERY = 100


# ==========================
# 工具函数
# ==========================
def load_lwf_dat(path: Path) -> np.ndarray:
    """
    读取 LWF_EFM *_lwf.dat
    返回:
      data: (nz, nx) float32
    """
    with path.open("r", encoding="utf-8", errors="ignore") as f:
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
        raise RuntimeError(f"[ERT DCT] 未找到 nx nz: {path}")

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


def pad_to_min_size(x: np.ndarray, min_x: int, min_y: int) -> np.ndarray:
    h, w = x.shape
    pad_h = max(0, min_x - h)
    pad_w = max(0, min_y - w)
    if pad_h > 0 or pad_w > 0:
        x = np.pad(x, ((0, pad_h), (0, pad_w)), mode="constant")
    return x


def dct2(x: np.ndarray) -> np.ndarray:
    return dctn(x, type=2, norm="ortho")


def process_one(dat_path: Path, out_path: Path):
    data = load_lwf_dat(dat_path)
    data = pad_to_min_size(data, KX, KY)
    coeff = dct2(data)
    feat = coeff[:KX, :KY].reshape(-1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, feat)


# ==========================
# [MOD] 从 pairs.csv 读取 ERT 输入
# ==========================
def load_ert_list_from_pairs(pairs_csv: Path, root: Path) -> list[Path]:
    files = []
    with pairs_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = root / row["ert_dat"]
            if p.exists():
                files.append(p)
    return files


def main():
    ROOT = Path(__file__).resolve().parents[1]

    ert_root = ROOT / "data" / "ert" / "ert_lwf"
    out_root = ROOT / "features" / "ert"
    pairs_csv = ROOT / "data" / "pairs.csv"

    # --------------------------
    # [MOD] 输入文件选择逻辑
    # --------------------------
    if pairs_csv.exists():
        files = load_ert_list_from_pairs(pairs_csv, ROOT)
        print(f"[ERT DCT] Using pairs.csv ({len(files)} samples)")
    else:
        files = sorted(ert_root.glob("*_lwf.dat"))
        print(f"[ERT DCT] Using directory scan ({len(files)} samples)")

    total = len(files)
    processed = 0
    failed = 0

    for dat_path in files:
        sid = dat_path.stem.replace("_lwf", "")
        out_path = out_root / f"{sid}.npy"

        try:
            process_one(dat_path, out_path)
            processed += 1
        except Exception:
            failed += 1
            continue

        if processed % PRINT_EVERY == 0 or processed == total:
            print(f"[ERT DCT] processed {processed} / {total}")

    print(f"[ERT DCT] DONE | total={processed} | failed={failed}")
    print(f"[ERT DCT] Features saved to : {out_root}")


if __name__ == "__main__":
    main()
