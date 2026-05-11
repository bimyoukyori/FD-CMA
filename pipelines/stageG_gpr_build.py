# pipelines/stageG_gpr_build.py
# ============================================================
# StageG : GPR Generation V2 批量建模（ERT → GPR）
#
# 位置（工程根目录）：
#   - gpr_generationV2/run_batch.py
#
# 依赖输入：
#   - EFM5/data/ert/ert_meta/*.json
#
# 关键输出（主线依赖）：
#   A) gpr_generationV2/outputs/<class>/<id>/<id>_gpr.json   (含 confidence / mapping)
#   B) EFM5/data/gpr/train/<class>/<id>.csv                  (DCT 主数据池）
#   C) EFM5/data/gpr/gpr_samples_v2.csv                      (审计表，不参与训练对齐)
#
# 设计原则：
#   - pipelines 只做调度与路径检查，不侵入 gpr_generationV2 内部逻辑
#   - 使用当前 python 解释器（pipelines/config.py 的 PYTHON）
#   - cwd 固定为 ROOT（保证 gpr_generationV2 内部相对路径解析稳定）
# ============================================================

from __future__ import annotations

import subprocess
from pathlib import Path

from pipelines.config import ROOT, EFM5_DIR, GPRV2_DIR, PYTHON, LOG_DIR


def _assert_inputs():
    meta_dir = EFM5_DIR / "data" / "ert" / "ert_meta"
    if not meta_dir.exists():
        raise FileNotFoundError(f"[StageG] 缺少 ERT meta 目录: {meta_dir}")

    meta_files = sorted(meta_dir.glob("*.json"))
    if len(meta_files) == 0:
        raise FileNotFoundError(f"[StageG] ERT meta 目录为空: {meta_dir}")

    run_script = GPRV2_DIR / "run_batch.py"
    if not run_script.exists():
        raise FileNotFoundError(f"[StageG] 脚本不存在: {run_script}")

    return run_script, meta_dir


def run():
    print("[StageG] GPR Generation V2 batch build")

    run_script, meta_dir = _assert_inputs()

    log = LOG_DIR / "stageG_gpr_build.log"
    with log.open("w", encoding="utf-8") as f:
        f.write(f"[StageG] ROOT      : {ROOT}\n")
        f.write(f"[StageG] ERT meta   : {meta_dir}\n")
        f.write(f"[StageG] Script    : {run_script}\n\n")

        subprocess.run(
            [PYTHON, str(run_script)],
            cwd=str(ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )

            # ===== 完成性判据（非常关键） =====
    
    ert_meta_files = sorted(meta_dir.glob("*.json"))
    n_ert = len(ert_meta_files)

    gpr_json_files = list(
        (GPRV2_DIR / "outputs").glob("*/*/*_gpr.json")
    )
    n_gpr = len(gpr_json_files)

    if n_gpr < n_ert:
        raise RuntimeError(
            f"[StageG][ERR] GPR 输出数量不足："
            f"ERT meta={n_ert}, GPR json={n_gpr}"
        )

    print(
        f"[StageG] Completed: "
        f"{n_gpr} GPR samples generated (ERT meta={n_ert})"
    )

    print(f"[StageG] Done. Log: {log}")


if __name__ == "__main__":
    run()
