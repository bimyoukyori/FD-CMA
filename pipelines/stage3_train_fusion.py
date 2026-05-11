# pipelines/stage3_train_fusion.py
# ============================================================
# Stage 3 : CLD-Fusion 多模态对比学习训练
#
# 功能：
#   - 调用 EFM5/fusion/train_fusion.py
#   - 使用脚本内部 CONFIG 作为唯一训练参数来源
#
# 输入：
#   - EFM5/data/pairs.csv
#   - EFM5/features/ert/*.npy
#   - EFM5/features/gpr/*.npy
#
# 输出（由 train_fusion.py 决定）：
#   - EFM5/fusion/runs/exp_debug/
#       ├── fusion_best.pth
#       ├── backbone_ert.pth
#       ├── backbone_gpr.pth
#       ├── fused_backbone.pth
#       └── meta.json
#
# 设计原则：
#   - Pipeline 不干预训练参数
#   - Pipeline 仅负责流程顺序与日志
# ============================================================

import subprocess
from pipelines.config import EFM5_DIR, PYTHON, LOG_DIR

def run():
    print("[Stage3] Train CLD-Fusion (use internal CONFIG)")

    script = EFM5_DIR / "fusion" / "train_fusion.py"
    log = LOG_DIR / "stage3_train_fusion.log"

    with open(log, "w", encoding="utf-8") as f:
        subprocess.run(
            [PYTHON, str(script)],
            cwd=str(EFM5_DIR),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )

    print(f"[Stage3] Done. Log: {log}")

if __name__ == "__main__":
    run()
