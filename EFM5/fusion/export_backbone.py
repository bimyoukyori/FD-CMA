#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_backbone.py

从 CLD-Fusion 训练结果中导出 GPR 分支（CNN backbone）

工程语义：
- 主线中通常【不需要】运行本脚本
- 仅在“只有 fusion_best.pth、但缺少 fused_backbone.pth”时使用
"""

from __future__ import annotations
from pathlib import Path
import torch


# ============================================================
# 工程路径（与 train_fusion / distill 完全一致）
# ============================================================

EFM5_ROOT = Path(__file__).resolve().parents[1]

FUSION_RUN_DIR = EFM5_ROOT / "fusion" / "runs" / "exp_fusion_main"

FUSED_MODEL = FUSION_RUN_DIR / "fusion_best.pth"
OUT_MODEL   = FUSION_RUN_DIR / "fused_backbone.pth"


# ============================================================
# 主逻辑
# ============================================================

def main():
    if not FUSED_MODEL.exists():
        raise FileNotFoundError(f"[ERR] fusion_best.pth 不存在: {FUSED_MODEL}")

    print("[INFO] Loading:", FUSED_MODEL)
    state = torch.load(FUSED_MODEL, map_location="cpu")

    # 判断是否是完整 FusionNet
    has_gpr_branch = any(k.startswith("gpr_branch.") for k in state.keys())

    if has_gpr_branch:
        gpr_state = {
            k.replace("gpr_branch.", "", 1): v
            for k, v in state.items()
            if k.startswith("gpr_branch.")
        }
        print("[INFO] Detected full FusionNet → extracting gpr_branch")
    else:
        gpr_state = state
        print("[INFO] Input already looks like pure GPR backbone")

    if not gpr_state:
        raise RuntimeError("[ERR] 未找到任何 GPR 权重")

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    torch.save(gpr_state, OUT_MODEL)

    print("[OK] GPR backbone saved to:")
    print("     ", OUT_MODEL)
    print(f"[OK] tensor count = {len(gpr_state)}")


if __name__ == "__main__":
    main()
