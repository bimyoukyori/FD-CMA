# pipelines/stage0_ert_build.py
# ============================================================
# Stage 0 : ERT 建模/正演（可选入口）
#
# 现实工程语义：
#   - 多数情况下 ERT 正演由你手动/GUI 完成（Console2 批量正演 + 汇总）
#   - pipelines 在此 Stage 仅做“可选启动入口”与“产物存在性判断”
#
# 判定逻辑：
#   - 若检测到以下任一产物存在，则直接跳过 Stage0：
#       A) EFM5/BatchModels/summary_result/
#       B) EFM5/data/ert/ert_meta/*.json
#
# 启动逻辑（可选）：
#   - 若产物不存在，则尝试在 EFM5 目录中寻找常见 GUI 入口脚本并启动
#   - 若仍找不到，则给出明确报错，提示需要先完成 ERT 正演产物准备
# ============================================================

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from pipelines.config import ROOT, EFM5_DIR, PYTHON


def _has_ert_products() -> bool:
    summary_dir = EFM5_DIR / "BatchModels" / "summary_result"
    meta_dir = EFM5_DIR / "data" / "ert" / "ert_meta"

    if summary_dir.exists():
        return True

    if meta_dir.exists():
        if any(meta_dir.glob("*.json")):
            return True

    return False


def _find_gui_entry() -> Path | None:
    # 常见 GUI 文件名候选（按优先级）
    candidates = [
        EFM5_DIR / "main_gui_psg_v2_2.py",
        EFM5_DIR / "efm_suite_gui.py",
        EFM5_DIR / "efm_gui.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def run():
    if _has_ert_products():
        print("[Stage0] ERT 产物已存在，跳过 Stage0（ERT 建模/正演）")
        return

    print("[Stage0] 未检测到 ERT 产物，尝试启动 ERT GUI...")

    gui_script = _find_gui_entry()
    if gui_script is None:
        raise FileNotFoundError(
            "[Stage0] 未找到 ERT GUI 入口脚本。\n"
            "请先完成 ERT 正演产物准备（BatchModels/summary_result 或 data/ert/ert_meta），\n"
            "或将 GUI 入口脚本放置在 EFM5 目录并加入候选列表。"
        )

    subprocess.run(
        [PYTHON, str(gui_script)],
        cwd=str(EFM5_DIR),
        check=True,
    )


if __name__ == "__main__":
    run()
