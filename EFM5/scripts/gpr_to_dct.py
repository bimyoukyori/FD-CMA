#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpr_to_dct.py
============================================================
GPR B-scan 图像 → 2D DCT 低频特征块（hw × hw）

【功能说明】
- 读取 GPR B-scan 图像（灰度）
- 可选灰度反转（invert）
- 图像归一化（none / minmax / zscore）
- 执行 2D DCT（Type-II, ortho）
- 截取左上角低频 hw × hw 区域作为特征
- 输出为 .npy（可选输出预览 PNG）

【运行模式】
1) 批量模式（推荐，工程/论文默认）
   python scripts/gpr_to_dct.py

2) 单文件模式（调试）
   python scripts/gpr_to_dct.py --img xxx.png --out xxx.npy

【重要说明（2026 版规范）】
- 默认参数写死在脚本内（DEFAULT_CONFIG），用于实验可追溯
- 命令行参数仅用于“覆盖默认值”
- 每次运行自动记录参数到 logs/gpr_to_dct.log
- 参数风格、日志机制与 ert_to_dct.py 完全一致

============================================================
"""

from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

# ============================================================
# [MOD] 默认配置区（论文 / 工程唯一可信参数源）
# ============================================================
DEFAULT_CONFIG = {
    "pairs_csv": "data/pairs.csv",
    "out_dir": "features/gpr",
    "hw": 64,
    "normalize": "minmax",     # none | minmax | zscore
    "invert": False,
    "save_png": False,
    "log_file": "logs/gpr_to_dct.log",
}

# ============================================================
# Try SciPy DCT first
# ============================================================
try:
    from scipy.fft import dctn
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _dct1(a: np.ndarray) -> np.ndarray:
    """1D DCT-II (orthonormal) fallback."""
    n = a.shape[-1]
    k = np.arange(n)
    j = np.arange(n)[:, None]
    cos_basis = np.cos(np.pi * (j + 0.5) * k / n)
    y = a @ cos_basis
    y[:, 0] *= np.sqrt(1 / n)
    y[:, 1:] *= np.sqrt(2 / n)
    return y


def _dct2(a: np.ndarray) -> np.ndarray:
    """2D DCT-II with ortho norm."""
    if _HAS_SCIPY:
        return dctn(a, type=2, norm="ortho")
    # separable fallback
    return _dct1(_dct1(a.T).T)


def _load_gray(path: Path, invert: bool) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if invert:
        arr = 1.0 - arr
    return arr


def _normalize(arr: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower()
    if mode == "none":
        return arr
    if mode == "zscore":
        mu = float(arr.mean())
        sd = float(arr.std()) or 1.0
        return (arr - mu) / sd
    if mode == "minmax":
        mn, mx = float(arr.min()), float(arr.max())
        if mx <= mn:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    raise ValueError("normalize must be: none | zscore | minmax")


def make_feature(img_path: Path, cfg: dict) -> np.ndarray:
    a = _load_gray(img_path, invert=cfg["invert"])
    a = _normalize(a, cfg["normalize"])
    D = _dct2(a)

    hw = cfg["hw"]
    h = min(hw, D.shape[0])
    w = min(hw, D.shape[1])
    patch = D[:h, :w].astype(np.float32)
    return patch


def process_one(img: Path, out: Path, cfg: dict):
    feat = make_feature(img, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, feat)

    print(f"[GPR] {img.name} -> {out}  shape={feat.shape}")

    if cfg["save_png"]:
        vis = feat.copy()
        mn, mx = float(vis.min()), float(vis.max())
        if mx > mn:
            vis = (vis - mn) / (mx - mn)
        else:
            vis = np.zeros_like(vis)
        img8 = (vis * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(img8).save(out.with_suffix(".preview.png"))


# ============================================================
# [MOD] CLI 参数（仅用于覆盖默认配置）
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser(description="GPR → DCT feature extraction")

    ap.add_argument("--pairs", default=None, help="Override pairs.csv")
    ap.add_argument("--out-dir", default=None, help="Override output dir")
    ap.add_argument("--hw", type=int, default=None)
    ap.add_argument("--normalize", default=None, choices=["none", "minmax", "zscore"])
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--save-png", action="store_true")

    ap.add_argument("--img", default=None, help="Single image (debug)")
    ap.add_argument("--out", default=None, help="Single output .npy (debug)")

    return ap.parse_args()


def merge_config(args) -> dict:
    cfg = DEFAULT_CONFIG.copy()

    if args.pairs:
        cfg["pairs_csv"] = args.pairs
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.hw is not None:
        cfg["hw"] = args.hw
    if args.normalize:
        cfg["normalize"] = args.normalize
    if args.invert:
        cfg["invert"] = True
    if args.save_png:
        cfg["save_png"] = True

    return cfg


def write_log(cfg: dict):
    log_path = Path(cfg["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Run time: {datetime.now()}\n")
        f.write(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    cfg = merge_config(args)

    write_log(cfg)

    print("[INFO] GPR → DCT using config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    # -------------------------------
    # 批量模式（默认）
    # -------------------------------
    if args.img is None:
        import csv
        pairs = Path(cfg["pairs_csv"])

        with pairs.open("r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                img = Path(row["gpr_img"])
                out = Path(row["gpr_feat"])
                process_one(img, out, cfg)
    else:
        # -------------------------------
        # 单文件模式（调试）
        # -------------------------------
        if not args.out:
            raise SystemExit("Single-file mode requires --out")
        process_one(Path(args.img), Path(args.out), cfg)


if __name__ == "__main__":
    main()
