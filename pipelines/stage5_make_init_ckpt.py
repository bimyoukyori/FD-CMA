# pipelines/stage5_make_init_ckpt.py
# ============================================================
# Stage 5 : 生成 RT-DETR 蒸馏初始化权重（Ultralytics 可用）
#
# 入口脚本位置（工程根目录）：
#   - ultralytics4/distill/make_ultralytics_init.py
#
# 输入：
#   - ultralytics4/distill/runs/exp_rtdetr_distill/rtdetr_distilled.pth
#   - ROOT/rtdetr-l.pt  （COCO pretrained，位于工程根目录）
#
# 输出：
#   - ultralytics4/distill/runs/exp_rtdetr_distill/rtdetr_l_distilled_init.pt
#
# 设计原则：
#   - pipelines 只做权重转换脚本的调度，不侵入 ultralytics 源码
#   - cwd 固定为 ROOT，保证脚本内部相对路径解析稳定
# ============================================================

from __future__ import annotations

import subprocess
from pipelines.config import ROOT, ULTRA_DIR, PYTHON, LOG_DIR


def run():
    print("[Stage5] Make RT-DETR distilled-init checkpoint")

    script = ULTRA_DIR / "distill" / "make_ultralytics_init.py"
    if not script.exists():
        raise FileNotFoundError(f"[Stage5] Script not found: {script}")

    log = LOG_DIR / "stage5_make_init_ckpt.log"

    with open(log, "w", encoding="utf-8") as f:
        subprocess.run(
            [PYTHON, str(script)],
            cwd=str(ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True
        )

    print(f"[Stage5] Done. Log: {log}")


if __name__ == "__main__":
    run()
