# pipelines/stage3c_export_feature_heatmaps.py
# ============================================================
# Stage 3c : Export feature heatmaps from CLD-Fusion
#
# 语义定位：
#   - 位于 Stage3（fusion 训练）之后、Stage3b（骨干导出）之前
#   - 对 fusion_best.pth 做“相似度驱动”的 GPR 分支热力图分析
#   - 生成 DCT 域热图、B-scan 回投热图与叠加图
# ============================================================

from __future__ import annotations

import os
import subprocess
from pipelines.config import EFM5_DIR, PYTHON, LOG_DIR


def run():
    print("[Stage3c] Export paired feature heatmaps from CLD-Fusion")

    script = EFM5_DIR / "fusion" / "export_feature_heatmaps.py"
    if not script.exists():
        raise FileNotFoundError(f"[Stage3c] Script not found: {script}")

    log = LOG_DIR / "stage3c_export_feature_heatmaps.log"
    env = os.environ.copy()
    env.setdefault("FUSION_BASELINE_RUN", "exp_fusion_baseline")
    env.setdefault("FUSION_MAIN_RUN", "exp_fusion_main")

    with open(log, "w", encoding="utf-8") as f:
        subprocess.run(
            [PYTHON, str(script)],
            cwd=str(EFM5_DIR),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )

    print(f"[Stage3c] Done. Log: {log}")


if __name__ == "__main__":
    run()