# pipelines/stage1_convert_ert.py
# ============================================================
# Stage 1 : ERT Console2 正演结果 → LWF 统一格式
#
# 功能：
#   - 调用 EFM5/scripts/console2_to_lwf.py
#   - 将 BatchModels/summary_result 中的 Console2 输出
#     转换为 data/ert_lwf + data/ert_meta
#
# 输入：
#   - EFM5/BatchModels/summary_result/
#
# 输出：
#   - EFM5/data/ert_lwf/*.dat
#   - EFM5/data/ert_meta/*.json
#
# 设计原则：
#   - pipelines 不 import EFM5 内部模块
#   - 仅通过 subprocess 调用
#   - cwd 必须设为 EFM5_DIR
# ============================================================

import subprocess
from pipelines.config import ROOT, EFM5_DIR, PYTHON, LOG_DIR


def run():
    print("[Stage1] Convert ERT results to LWF format")

    # 1️⃣ 正确的脚本路径（scripts，不是 tools）
    script = EFM5_DIR / "scripts" / "console2_to_lwf.py"
    if not script.exists():
        raise FileNotFoundError(f"[Stage1] Script not found: {script}")

    # 2️⃣ 日志
    log = LOG_DIR / "stage1_convert_ert.log"

    # 3️⃣ subprocess 调用（cwd 必须是 EFM5）
    subprocess.run(
        [PYTHON, str(script)],
        cwd=str(EFM5_DIR),              # ★ 关键修复点
        stdout=open(log, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        check=True
    )

    print(f"[Stage1] Done. Log: {log}")


if __name__ == "__main__":
    run()
