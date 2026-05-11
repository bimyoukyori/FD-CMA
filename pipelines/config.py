# pipelines/config.py
# ============================================================
# Pipeline 全局配置（MUL05）
#
# 目标：
#   - 为 pipelines 下所有 Stage 提供统一的路径锚点与运行环境
#   - 固化“工程根目录相对路径”口径，避免盘符/工作目录漂移
#
# 关键路径：
#   - ROOT          : MUL05 工程根目录（包含 EFM5 / gpr_generationV2 / ultralytics4 / pipelines）
#   - EFM5_DIR      : EFM5 工程目录
#   - GPRV2_DIR     : gpr_generationV2 目录（GPR Generation V2）
#   - ULTRA_DIR     : ultralytics4 目录（RT-DETR distill + 下游训练）
#
# 运行环境：
#   - PYTHON        : 当前运行 pipelines 的 python 解释器（保证环境一致）
#   - LOG_DIR       : pipelines 统一日志目录（ROOT/logs）
# ============================================================

from __future__ import annotations

from pathlib import Path
import sys


# ------------------------------------------------------------
# 工程根目录：MUL05
# pipelines/config.py 位于 ROOT/pipelines/config.py
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

# 工程子目录
EFM5_DIR = ROOT / "EFM5"
GPRV2_DIR = ROOT / "gpr_generationV2"
ULTRA_DIR = ROOT / "ultralytics4"

# Python 解释器（使用当前环境）
PYTHON = sys.executable

# 统一日志目录
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Debug（保留，便于定位“路径锚点是否正确”）
print(f"[PipelineConfig] ROOT      = {ROOT}")
print(f"[PipelineConfig] EFM5_DIR  = {EFM5_DIR}")
print(f"[PipelineConfig] GPRV2_DIR = {GPRV2_DIR}")
print(f"[PipelineConfig] ULTRA_DIR = {ULTRA_DIR}")
print(f"[PipelineConfig] PYTHON    = {PYTHON}")
print(f"[PipelineConfig] LOG_DIR   = {LOG_DIR}")
