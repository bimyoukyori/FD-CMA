# pipelines/stage4_distill_rtdetr.py
# ============================================================
# Stage 4 : RT-DETR Backbone 特征级蒸馏（Distillation）
#
# 入口脚本位置（工程根目录）：
#   - ultralytics4/distill/train_rtdetr_distill.py
#
# 输入来源（主线）：
#   - EFM5/data/pairs.csv
#   - EFM5/features/gpr/*.npy
#   - EFM5/fusion/runs/exp_fusion_main/fused_backbone.pth
#
# 输出：
#   - ultralytics4/distill/runs/exp_rtdetr_distill/rtdetr_distilled.pth
#
# 设计原则：
#   - pipelines 不改动 distill 内部逻辑，仅做 subprocess 调度
#   - cwd 固定为 ROOT，保证 distill 脚本内部相对路径解析稳定
# ============================================================

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pipelines.config import ROOT, ULTRA_DIR, PYTHON, LOG_DIR


def run():
    print("[Stage4] RT-DETR distillation started...")

    log_path = LOG_DIR / "stage4_distill.log"

    distill_script = ULTRA_DIR / "distill" / "train_rtdetr_distill.py"
    if not distill_script.exists():
        raise FileNotFoundError(f"[Stage4] Distill script not found: {distill_script}")

    cmd = [PYTHON, str(distill_script)]

    print("[Stage4] Command:")
    print(" ", " ".join(cmd))
    print("[Stage4] Log file:")
    print(" ", log_path)

    env = os.environ.copy()
    env["DISTILL_LOG_PATH"] = str(log_path)

    with log_path.open("w", encoding="utf-8") as f:
        subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )

    print("[Stage4] RT-DETR distillation finished.")
    print(f"[Stage4] Log saved to: {log_path}")


if __name__ == "__main__":
    run()
