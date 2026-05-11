# pipelines/stage2_make_dct.py
# ============================================================
# Stage 2 : 配对（pairs.csv） + 2D-DCT 特征构建 + 特征体检
#
# 对应论文/工程主线（红框核心）：
#   2.1 make_pairs_pre_dct.py  : 生成 pairs.csv（写入真实 confidence；gpr_csv 推荐相对路径）
#   2.2 ert_to_2Ddct.py        : ERT 数值场 → nan_to_num → 2D-DCT → 32×32 → 1024D
#   2.3 gpr_to_2Ddct.py        : GPR B-scan CSV → 2D-DCT → 32×32 → 1024D
#   2.4 check_features.py      : 特征存在性/维度/NaNInf/匹配统计
#
# 输入依赖（运行前必须满足）：
#   - EFM5/data/ert/ert_lwf/*.dat
#   - EFM5/data/ert/ert_npy/*.npy
#   - EFM5/data/gpr/train/<class>/*.csv     （由 StageG 生成）
#   - gpr_generationV2/outputs/.../*_gpr.json（用于 pairs 写 confidence）
#
# 输出：
#   - EFM5/data/pairs.csv
#   - EFM5/features/ert/*.npy
#   - EFM5/features/gpr/*.npy
#   - 控制台/日志：匹配统计与体检结果
#
# 设计原则：
#   - pipelines 只做流程调度（subprocess），不 import EFM5 内部逻辑
#   - 所有脚本统一在 cwd=EFM5_DIR 下运行（保证相对路径稳定）
# ============================================================

from __future__ import annotations

import subprocess
from pipelines.config import EFM5_DIR, PYTHON, LOG_DIR


def run():
    print("[Stage2] Pairs + 2D-DCT + feature check")

    scripts = [
        EFM5_DIR / "scripts" / "make_pairs_pre_dct.py",
        EFM5_DIR / "scripts" / "ert_to_2Ddct.py",
        EFM5_DIR / "scripts" / "gpr_to_2Ddct.py",
        EFM5_DIR / "scripts" / "check_features.py",
    ]

    for s in scripts:
        if not s.exists():
            raise FileNotFoundError(f"[Stage2] Script not found: {s}")

    log = LOG_DIR / "stage2_make_dct.log"

    with open(log, "w", encoding="utf-8") as f:
        for script in scripts:
            print(f"[Stage2] Running: {script.name}")
            subprocess.run(
                [PYTHON, str(script)],
                cwd=str(EFM5_DIR),
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )

    print(f"[Stage2] Done. Log: {log}")


if __name__ == "__main__":
    run()
