#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpr_to_2Ddct.py
==================================================
GPR B-scan → 2D DCT → 低频块 → 一维特征

【输入策略说明｜2026-01 更新】
1. 若存在 data/pairs.csv：
   - 优先从 pairs.csv 中读取 gpr_csv 字段
2. 否则：
   - 扫描 data/gpr/train/{class}/*.csv

输出:
  features/gpr/{ID}.npy
"""

from __future__ import annotations

import csv
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.fft import dctn


# ==========================
# 全局参数
# ==========================
KX = 32
KY = 32
PRINT_EVERY = 100


def dct2(x: np.ndarray) -> np.ndarray:
    return dctn(x, type=2, norm="ortho")


def process_one(csv_path: Path, out_path: Path):
    data = pd.read_csv(csv_path, header=None).values.astype(np.float32)
    coeff = dct2(data)
    feat = coeff[:KX, :KY].reshape(-1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, feat)


# ==========================
# [MOD] 从 pairs.csv 读取 GPR 输入
# ==========================
def load_gpr_list_from_pairs(pairs_csv: Path, root: Path) -> list[Path]:
    files = []
    with pairs_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = Path(row["gpr_csv"])
            if not p.is_absolute():
                p = root / p
            if p.exists():
                files.append(p)
    return files


def main():
    ROOT = Path(__file__).resolve().parents[1]

    gpr_root = ROOT / "data" / "gpr" / "train"
    out_root = ROOT / "features" / "gpr"
    pairs_csv = ROOT / "data" / "pairs.csv"

    # --------------------------
    # [MOD] 输入文件选择逻辑
    # --------------------------
    if pairs_csv.exists():
        files = load_gpr_list_from_pairs(pairs_csv, ROOT)
        print(f"[GPR DCT] Using pairs.csv ({len(files)} samples)")
    else:
        files = []
        for class_dir in sorted(gpr_root.iterdir()):
            if class_dir.is_dir():
                files.extend(sorted(class_dir.glob("*.csv")))
        print(f"[GPR DCT] Using directory scan ({len(files)} samples)")

    total = len(files)
    processed = 0
    failed = 0

    for csv_path in files:
        sid = csv_path.stem
        out_path = out_root / f"{sid}.npy"

        try:
            process_one(csv_path, out_path)
            processed += 1
        except Exception:
            failed += 1
            continue

        if processed % PRINT_EVERY == 0 or processed == total:
            print(f"[GPR DCT] processed {processed} / {total}")

    print(f"[GPR DCT] DONE | total={processed} | failed={failed}")
    print(f"[GPR DCT] Features saved to : {out_root}")


if __name__ == "__main__":
    main()
