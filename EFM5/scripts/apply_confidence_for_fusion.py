#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_confidence_for_fusion.py
==================================================
在融合阶段前，为 pairs.csv 注入样本级物理置信度（confidence）

核心说明：
- pairs.csv 来源于 DCT 前置配对（confidence 为占位值）
- 真实 confidence 来自 GPR Generation V2 输出的 *_gpr.json
- 本脚本不修改原 pairs.csv，而是生成 pairs_fusion.csv

适用阶段：
DCT 特征提取完成之后 → CLD-Fusion / InfoNCE 训练之前
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def load_confidence_from_gpr_json(gpr_json_path: Path) -> float:
    if not gpr_json_path.exists():
        return 1.0

    try:
        with gpr_json_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return float(meta.get("confidence", 1.0))
    except Exception:
        return 1.0


def main():
    ROOT = Path(__file__).resolve().parents[1]
    PROJECT_ROOT = ROOT.parent
    
    pairs_in = ROOT / "data" / "pairs.csv"
    pairs_out = ROOT / "data" / "pairs_fusion.csv"

    # EFM5 的父目录 = MUL05
    
    gpr_outputs = PROJECT_ROOT / "gpr_generationV2" / "outputs"

    if not pairs_in.exists():
        raise SystemExit(f"[ERR] 未找到 pairs.csv: {pairs_in}")

    rows = []

    with pairs_in.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["id"]
            label = row["label"]

            # gpr_generationV2/outputs/{class}/{ID}/{ID}_gpr.json
            gpr_json = (
                gpr_outputs
                / label
                / sid
                / f"{sid}_gpr.json"
            )

            #print(f"[DEBUG] {gpr_json} exists={gpr_json.exists()}")
            
            confidence = load_confidence_from_gpr_json(gpr_json)
            row["confidence"] = f"{confidence:.3f}"

            rows.append(row)

    with pairs_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("[OK] pairs_fusion.csv 已生成")
    print(f"     输入: {pairs_in}")
    print(f"     输出: {pairs_out}")
    print(f"     样本数: {len(rows)}")


if __name__ == "__main__":
    main()
