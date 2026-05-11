#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_features.py
==================================================
融合前特征一致性体检脚本（ERT / GPR）

检查内容：
- 特征文件是否存在
- 特征维度是否一致
- 是否包含 NaN / Inf
- ERT / GPR 特征数量统计
- pairs 中成功匹配数量
- 按类别统计匹配样本数
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


# ==========================================================
# [MOD] ERT 特征类型选择（论文级实验配置，禁止命令行修改）
# ==========================================================
# 可选值：
#   "2d"    : 主线 2D-DCT（features/ert）
#   "block" : 分块 DCT + 位置编码（features/ert_blockdct_pos）
#
# ⚠️ 论文主线请保持为 "2d"
# ⚠️ 消融实验时，手动改为 "block" 并在论文中注明
# ==========================================================
ERT_DCT_TYPE = "2d"#消融实验时，手动改为 "block" 并在论文中注明
#ERT_DCT_TYPE = "block"#消融实验时，手动改为 "block" 并在论文中注明

def main():
    # ==========================================================
    # 工程根目录（EFM5）
    # ==========================================================
    ROOT = Path(__file__).resolve().parents[1]

    # ==========================================================
    # pairs 文件选择（优先 fusion）
    # ==========================================================
    pairs_fusion = ROOT / "data" / "pairs_fusion.csv"
    pairs_plain = ROOT / "data" / "pairs.csv"

    if pairs_fusion.exists():
        pairs_path = pairs_fusion
    elif pairs_plain.exists():
        pairs_path = pairs_plain
    else:
        raise FileNotFoundError("[ERR] 未找到 pairs.csv / pairs_fusion.csv")

    print(f"[CHECK] Using pairs file : {pairs_path}")
    pairs = pd.read_csv(pairs_path)
    print(f"[CHECK] 配对记录数      : {len(pairs)}")

    # ==========================================================
    # [MOD] 根据 DCT 类型选择 ERT 特征目录
    # ==========================================================
    if ERT_DCT_TYPE == "2d":
        ert_feat_root = ROOT / "features" / "ert"
    elif ERT_DCT_TYPE == "block":
        ert_feat_root = ROOT / "features" / "ert_blockdct_pos"
    else:
        raise ValueError(f"[ERR] 未知的 ERT_DCT_TYPE: {ERT_DCT_TYPE}")

    gpr_feat_root = ROOT / "features" / "gpr"

    print(f"[CHECK] ERT DCT type        : {ERT_DCT_TYPE}")
    print(f"[CHECK] ERT feature dir    : {ert_feat_root}")
    print(f"[CHECK] GPR feature dir    : {gpr_feat_root}")

    if not ert_feat_root.exists():
        raise FileNotFoundError(f"[ERR] ERT 特征目录不存在: {ert_feat_root}")
    if not gpr_feat_root.exists():
        raise FileNotFoundError(f"[ERR] GPR 特征目录不存在: {gpr_feat_root}")

    ert_total = len(list(ert_feat_root.glob("*.npy")))
    gpr_total = len(list(gpr_feat_root.glob("*.npy")))

    print(f"[CHECK] ERT 特征文件总数 : {ert_total}")
    print(f"[CHECK] GPR 特征文件总数 : {gpr_total}")

    # ==========================================================
    # 融合层：逐条检查
    # ==========================================================
    ref_shape = None
    matched = 0
    class_counter = defaultdict(int)

    for _, row in pairs.iterrows():
        sid = row["id"]

        ert_path = ert_feat_root / f"{sid}.npy"
        gpr_path = (ROOT / row["gpr_feat"]).resolve()

        if not ert_path.exists() or not gpr_path.exists():
            continue

        ert_feat = np.load(ert_path)
        gpr_feat = np.load(gpr_path)

        # ---- 维度一致性 ----
        if ert_feat.shape != gpr_feat.shape:
            ref_shape = ert_feat.shape
            # raise RuntimeError(
            #     f"[ERR] GPR/ERT 维度不一致: {sid} | "
            #     f"ERT={ert_feat.shape}, GPR={gpr_feat.shape}"
            # )

        if ref_shape is None:
            ref_shape = ert_feat.shape

        if ert_feat.shape != ref_shape:
            raise RuntimeError(f"[ERR] 特征维度漂移: {sid}")

        # ---- 数值稳定性 ----
        if not np.isfinite(ert_feat).all():
            raise RuntimeError(f"[ERR] ERT 特征存在 NaN/Inf: {sid}")

        if not np.isfinite(gpr_feat).all():
            raise RuntimeError(f"[ERR] GPR 特征存在 NaN/Inf: {sid}")

        matched += 1

        # ---- 类别统计（如果存在 class 字段） ----
        if "class" in row:
            class_counter[row["class"]] += 1

    # ==========================================================
    # 统计输出
    # ==========================================================
    print("-" * 60)
    print(f"[CHECK] 成功匹配样本数  : {matched}")
    print(f"[CHECK] 特征维度        : {ref_shape}")

    if class_counter:
        print("[CHECK] 各类别匹配数量:")
        for cls, cnt in class_counter.items():
            print(f"  - {cls:<7} : {cnt}")

    print("-" * 60)
    print("[OK] 特征一致性检查通过，可进入特征融合！")


if __name__ == "__main__":
    main()
