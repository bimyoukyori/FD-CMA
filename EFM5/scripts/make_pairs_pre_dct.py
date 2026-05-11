#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_pairs_pre_dct.py
==================================================
构建 DCT / CLD-Fusion 使用的 pairs.csv（直接写入真实 confidence）

核心说明：
- GPR 主数据来源为 CSV（而非 PNG）
- CSV 位于 data/gpr/train/{class}/{ID}.csv
- 采用 ID + 类别一致性进行对齐
- 在对齐阶段直接写入 GPR Generation V2 产生的真实 confidence
- 作为 DCT 特征提取与 CLD-Fusion 的统一样本索引文件

适用阶段：
GPR Generation V2 完成之后 → DCT / 融合 / 蒸馏之前
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, Optional, List


# ==================================================
# 工具函数：从文件名中提取数字 ID
# 例如 "00001.csv" / "00001_lwf.dat" → "00001"
# ==================================================
def extract_id(name: str) -> Optional[str]:
    m = re.search(r"(\d+)", name)
    return m.group(1) if m else None


# ==================================================
# 收集 ERT LWF 数据
# 结构：
#   data/ert/ert_lwf/{ID}_lwf.dat
# ==================================================
def collect_ert_lwf(ert_dir: Path) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    for p in ert_dir.glob("*_lwf.dat"):
        sid = extract_id(p.stem)
        if sid and sid not in found:
            found[sid] = p
    return found


# ==================================================
# 收集 GPR CSV 数据（训练阶段 · DCT 前置）
#
# 数据结构：
#   data/gpr/train/{class}/{ID}.csv
#
# [MOD] 输出 pairs.csv 时，gpr_csv 统一写“相对 EFM5_ROOT 的相对路径”，
#       避免 Windows 盘符/个人目录导致跨机器不可复现。
# ==================================================
def collect_gpr_csv(gpr_train_dir: Path, efm5_root: Path) -> Dict[str, Dict[str, str]]:
    found: Dict[str, Dict[str, str]] = {}

    for class_dir in gpr_train_dir.iterdir():
        if not class_dir.is_dir():
            continue

        label = class_dir.name

        for csv_path in class_dir.iterdir():
            if csv_path.suffix.lower() != ".csv":
                continue

            sid = extract_id(csv_path.stem)
            if not sid:
                continue

            if sid in found:
                continue

            # ------------------------------
            # [MOD] 写入相对路径（相对 EFM5_ROOT）
            # e.g. data/gpr/train/cavity/00001.csv
            # ------------------------------
            try:
                rel_csv = csv_path.relative_to(efm5_root).as_posix()
            except Exception:
                # 兜底：如果不在 efm5_root 下，则回退为原路径（仍可读，但不可复现）
                rel_csv = str(csv_path)

            found[sid] = {
                "csv": rel_csv,
                "label": label
            }

    return found


# ==================================================
# 从 GPR Generation V2 输出中读取真实 confidence
# 路径：
#   gpr_generationV2/outputs/{class}/{ID}/{ID}_gpr.json
# ==================================================
def load_confidence(
    project_root: Path,
    label: str,
    sid: str
) -> float:
    gpr_json = (
        project_root
        / "gpr_generationV2"
        / "outputs"
        / label
        / sid
        / f"{sid}_gpr.json"
    )

    if not gpr_json.exists():
        return 1.0

    try:
        with gpr_json.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return float(meta.get("confidence", 1.0))
    except Exception:
        return 1.0


# ==================================================
# 主程序：生成 pairs.csv（直接写真实 confidence）
# ==================================================
def main():
    ROOT = Path(__file__).resolve().parents[1]   # EFM5_ROOT
    PROJECT_ROOT = ROOT.parent                  # MUL05_ROOT（gpr_generationV2 所在工程根目录）

    # ------------------------------
    # 路径配置
    # ------------------------------
    ert_lwf_dir = ROOT / "data" / "ert" / "ert_lwf"
    ert_npy_dir = ROOT / "data" / "ert" / "ert_npy"
    gpr_train_dir = ROOT / "data" / "gpr" / "train"

    feat_ert_dir = ROOT / "features" / "ert"
    feat_gpr_dir = ROOT / "features" / "gpr"

    out_csv = ROOT / "data" / "pairs.csv"

    # ------------------------------
    # 基本存在性校验
    # ------------------------------
    if not ert_lwf_dir.exists():
        raise SystemExit(f"[ERR] ERT LWF 目录不存在: {ert_lwf_dir}")

    if not ert_npy_dir.exists():
        raise SystemExit(f"[ERR] ERT NPY 目录不存在: {ert_npy_dir}")

    if not gpr_train_dir.exists():
        raise SystemExit(f"[ERR] GPR train 目录不存在: {gpr_train_dir}")

    # ------------------------------
    # 收集数据
    # ------------------------------
    ert_map = collect_ert_lwf(ert_lwf_dir)

    # [MOD] 传入 EFM5_ROOT 以便输出 gpr_csv 相对路径
    gpr_map = collect_gpr_csv(gpr_train_dir, ROOT)

    common_ids = sorted(
        set(ert_map.keys()) & set(gpr_map.keys()),
        key=lambda x: int(x)
    )

    # ------------------------------
    # 创建特征输出目录
    # ------------------------------
    feat_ert_dir.mkdir(parents=True, exist_ok=True)
    feat_gpr_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # 生成 CSV 行
    # ------------------------------
    rows: List[Dict[str, str]] = []

    for sid in common_ids:
        sid5 = sid.zfill(5)
        gpr_info = gpr_map[sid]

        # 读取真实 confidence
        confidence = load_confidence(
            PROJECT_ROOT,
            gpr_info["label"],
            sid5
        )

        rows.append({
            "id": sid5,

            # ===== ERT 原始输入（消融用）=====
            "ert_dat": f"data/ert/ert_lwf/{sid5}_lwf.dat",

            # ===== ERT 主线二维输入 =====
            "ert_npy": f"data/ert/ert_npy/model_{sid5}_pseudo.npy",

            # ===== GPR 输入 =====
            # [MOD] 此处 gpr_csv 已是相对 EFM5_ROOT 的相对路径
            "gpr_csv": gpr_info["csv"],

            # ===== 特征输出（运行时生成）=====
            "ert_feat": f"features/ert/{sid5}.npy",
            "gpr_feat": f"features/gpr/{sid5}.npy",

            "label": gpr_info["label"],
            "split": "train",
            "confidence": f"{confidence:.3f}",
        })

    # ------------------------------
    # 写出 pairs.csv
    # ------------------------------
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "ert_dat",
                "ert_npy",
                "gpr_csv",
                "ert_feat",
                "gpr_feat",
                "label",
                "split",
                "confidence"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------
    # 日志输出
    # ------------------------------
    print("[OK] pairs.csv 已生成（直接写入真实 confidence）")
    print(f"     输出路径 : {out_csv}")
    print(f"     ERT 样本 : {len(ert_map)}")
    print(f"     GPR 样本 : {len(gpr_map)}")
    print(f"     成功配对 : {len(common_ids)}")
    print("     gpr_csv  : 已统一写为相对 EFM5_ROOT 的相对路径（跨机器可复现）")


if __name__ == "__main__":
    main()
