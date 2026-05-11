# pipelines/stage3b_export_gpr_backbone.py
# ============================================================
# Stage 3b : Export GPR backbone from CLD-Fusion
#
# 语义定位：
#   - 显式完成 “多模态训练 → 单模态推理” 的工程断点
#   - 从 fusion_best.pth 导出 fused_backbone.pth
#
# 输入：
#   - EFM5/fusion/runs/exp_fusion_main/fusion_best.pth
#
# 输出：
#   - EFM5/fusion/runs/exp_fusion_main/fused_backbone.pth
# ============================================================

import subprocess
from pathlib import Path
from pipelines.config import EFM5_DIR, PYTHON, LOG_DIR


def run():
    print("[Stage3b] Export GPR backbone from CLD-Fusion")

    fusion_dir = EFM5_DIR / "fusion" / "runs" / "exp_fusion_main"
    fusion_ckpt = fusion_dir / "fusion_best.pth"

    if not fusion_ckpt.exists():
        raise RuntimeError(
            "[Stage3b][ERR] fusion_best.pth not found. "
            "Please run Stage3 (CLD-Fusion) first."
        )

    script = EFM5_DIR / "fusion" / "export_backbone.py"
    if not script.exists():
        raise RuntimeError(f"[Stage3b][ERR] export_backbone.py not found: {script}")

    log = LOG_DIR / "stage3b_export_gpr_backbone.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    with open(log, "w", encoding="utf-8") as f:
        subprocess.run(
            [PYTHON, str(script)],
            cwd=str(EFM5_DIR),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )

    out_ckpt = fusion_dir / "fused_backbone.pth"
    if not out_ckpt.exists():
        raise RuntimeError(
            "[Stage3b][ERR] Export finished but fused_backbone.pth not found."
        )

    print(f"[Stage3b] Done. Exported: {out_ckpt}")
    print(f"[Stage3b] Log saved to: {log}")


if __name__ == "__main__":
    run()
