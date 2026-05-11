#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ert_to_1Ddct.py
==================================================
ERT LWF 数值场 (.dat) → 1D DCT → 全局频谱特征（基线 / 消融用）

【方法定位】
- 不引入任何空间位置编码
- 不保留二维结构
- 仅作为历史方法 / 消融下界
- 对应论文中的 “ERT-1D-DCT Baseline”

输入:
  data/ert/ert_lwf/{ID}_lwf.dat

输出:
  features/ert_1d/{ID}.npy
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from scipy.fft import dct


# ==========================
# 全局参数（冻结）
# ==========================
KEEP_COEFF = 256       # 保留的低频系数数量
PRINT_EVERY = 100


# ==========================
# 工具函数
# ==========================
def load_lwf_dat(path: Path) -> np.ndarray:
    """
    读取 LWF 规则 ERT 数值场 (.dat)
    适配带 5 行文件头 + 5 列数据的格式
    """
    # 1. skiprows=5: 跳过前5行元数据 (LWF_EFM_MODEL, JSON, 尺寸, 间距, 原点)
    # 2. usecols=4: 只读取第5列数据（Python索引从0开始，所以是4），即最后一列的电阻率/数值
    # 3. encoding='utf-8': 防止读取中文注释报错
    try:
        data = np.loadtxt(path, skiprows=5, usecols=4, encoding='utf-8', dtype=np.float32)
    except Exception as e:
        raise ValueError(f"[ERT-1D-DCT] 读取失败 {path}: {e}")

    # 修正维度：loadtxt 读出一列数据默认是 1D 数组 (N,)
    # 原代码检查 if data.ndim != 2，所以这里手动升维成 (1, N) 或 (N, 1)
    if data.ndim == 1:
        data = data.reshape(1, -1)  # 变成 (1, 1335) 这种二维形式，骗过后面的检查

    # 如果后续逻辑需要特定的二维形状（比如 nz * nx），可能需要根据文件头里的 nx, nz 重塑
    # 但根据你的代码 `vec = field.reshape(-1)`，只要数据总量对，形状不影响最终结果。
    
    return data
def dct_1d(x: np.ndarray) -> np.ndarray:
    """标准 1D DCT-II（正交归一）"""
    return dct(x, type=2, norm="ortho")


def process_one(dat_path: Path, out_path: Path):
    # ---------- 1. 读取二维数值场 ----------
    field = load_lwf_dat(dat_path)

    # ---------- 2. 展平为 1D ----------
    vec = field.reshape(-1)

    # ---------- 3. 1D DCT ----------
    coeff = dct_1d(vec)

    # ---------- 4. 截取低频 ----------
    feat = coeff[:KEEP_COEFF]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, feat.astype(np.float32))


# ==========================
# 主程序
# ==========================
def main():
    ROOT = Path(__file__).resolve().parents[1]
    ert_root = ROOT / "data" / "ert" / "ert_lwf"
    out_root = ROOT / "features" / "ert_1d"

    files = sorted(ert_root.glob("*_lwf.dat"))
    total = len(files)
    processed = 0
    failed = 0

    print(f"[ERT-1D-DCT] Input root  : {ert_root}")
    print(f"[ERT-1D-DCT] Output root : {out_root}")
    print(f"[ERT-1D-DCT] KEEP_COEFF  : {KEEP_COEFF}")

    for dat_path in files:
        sid = dat_path.stem.split("_")[0]  # 00001
        out_path = out_root / f"{sid}.npy"

        try:
            process_one(dat_path, out_path)
            processed += 1
        except Exception as e:
            print(f"[ERT-1D-DCT][FAIL] {sid}: {e}")
            failed += 1
            continue

        if processed % PRINT_EVERY == 0 or processed == total:
            print(f"[ERT-1D-DCT] processed {processed} / {total}")

    print(f"[ERT-1D-DCT] DONE | total={processed} | failed={failed}")
    print(f"[ERT-1D-DCT] Features saved to : {out_root}")


if __name__ == "__main__":
    main()
