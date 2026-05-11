# -*- coding: utf-8 -*-
"""
gpr_generationV2/config.py
============================================================
Stage GPR-V2 : 全局配置与路径管理（ROOT 相对路径）
【GPR 仿真配置模块（统一管理，不写死到脚本里）】

功能：
  1) 提供 find_project_root()：以工程 ROOT(MUL03) 为锚点的相对路径机制（跨平台/可迁移）
  2) 提供 GPRDomain / GPRGrid / GPRSimParams：gprMax 仿真域、网格、物理参数的统一默认值
  3) 提供 Paths：集中管理 ERT meta、GPR train、pairs.csv、outputs 等路径

设计原则：
  - 不依赖命令行参数（参数都写在此处，便于版本管理和回溯）
  - 默认参数尽量对齐你已验证的 run_gpr_simulation.py 稳定配置
  - 以“先跑通闭环 + 论文级可复现”为目标

关键修复（本次）：
  - [MOD] n_runs 默认从 1 改为 80：避免单道 A-scan 被“去直达波”处理减成全 0 导致黑图
  - [MOD] domain_z / dz 恢复为 0.0025（伪 2D），避免误改成 1.0 导致计算量爆炸
============================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def find_project_root() -> Path:
    """
    解析工程 ROOT（MUL03）：
    约定 gpr_generationV2/ 位于 MUL03/ 下一级目录，因此 ROOT = 当前文件目录的上一级。
    """
    return Path(__file__).resolve().parents[1]


# ============================================================
# GPR 仿真域与网格（gprMax: #domain / #dx_dy_dz）
# ============================================================

@dataclass(frozen=True)
class GPRDomain:
    """
    gprMax 仿真域（单位：m）
    注意：你当前采用“伪 2D”方式，z 方向非常薄（0.0025m），主要变化在 x-y 平面。
    """
    x: float = 2.5
    y: float = 5.0   # [MOD-400MHz] 拉深仿真域##400MHz主线1.5 改5.0
    #y: float = 10.  # [MOD-100MHz] 拉深仿真域#10.0 
    z: float = 0.0025   # ✅ 伪 2D，必须极薄 
   

@dataclass(frozen=True)
class GPRGrid:
    """
    网格步长（单位：m）
    需要与 domain 的尺度匹配，否则会导致网格层数异常。
    """
    #dx/dy=0.005 对 400MHz 可能偏细
    dx: float = 0.005#0.0025
    dy: float = 0.005#0.0025
    dz: float = 0.0025  # ✅ 与 domain.z 完全一致
    
# ============================================================
# gprMax 仿真物理/数值参数（对齐你原始可用脚本）
# ============================================================

@dataclass(frozen=True)
class GPRSimParams:
    """
    核心仿真参数：
      - time_window / freq：波形时窗与中心频率
      - tx/rx：天线位置（注意你目前 y≈1.45 的约定保持不变）
      - src_steps / rx_steps：B-scan 步进
      - soil_peplinski + fractal_box：背景介质与随机场
      - n_runs：B-scan 道数（最关键）
    """
   # ---- 空气层参数（由 simulate.py 使用）----
    air_gap_m: float = 0.05   # 5 cm，sanity
    
    #soil_thickness: float = 8.0   # [MOD-100MHz] 土体厚度
    soil_thickness: float = 3.0 # [MOD-400MHz] 土体厚度
    
    #新的：主线（400 MHz + 120 ns + 0–5m）
    #
    time_window: float = 1.2e-7 # [MOD-400MHz] 120 ns #8e-8  #旧3e-8
    freq: float = 400_000_000.0  #频率400 #旧80ns 900_000_000.0
    # time_window: float = 1.5e-7   # [MOD-100MHz] 150 ns
    # freq: float = 100_000_000.0   # [MOD-100MHz]
    
    # 天线（与 run_gpr_simulation.py 保持一致）
    tx_x: float = 0.20
    tx_y: float = 1.45 # ❗占位，simulate.py 会覆盖
    tx_z: float = 0.0

    rx_x: float = 0.25
    rx_y: float = 1.45  # ❗占位，simulate.py 会覆盖
    rx_z: float = 0.0

    # B-scan 步进（沿 x 扫描）
    src_steps_x: float = 0.02
    src_steps_y: float = 0.0
    src_steps_z: float = 0.0

    rx_steps_x: float = 0.02
    rx_steps_y: float = 0.0
    rx_steps_z: float = 0.0

    # Peplinski 土体参数
    sand: float = 0.5
    clay: float = 0.5
    density: float = 2.0
    sand_density: float = 2.66
    water: float = 0.15
    conductivity: float = 0.01

    # Fractal 背景
    fractal_n: float = 1.5
    fractal_ax: float = 1.0
    fractal_ay: float = 1.0
    fractal_az: float = 1.0
    materials: int = 50

    # 随机种子（会被 run_batch.py 按 ID 覆盖）
    seed: int = 9

    # gprMax API 参数
    geometry_only: bool = False
                                                                                                                                                                   
    # [MOD] 关键修复：默认 80 道（与你原始脚本 num_scan=80 一致）
    #n_runs: int = 120 #120  #旧80 #快速验证20
    n_runs: int = 80    # [MOD-100MHz] 先试形态
     #当前项目5米：120
    #验收用：建议先改成 n_runs=20（快 4 倍）

# ============================================================
# 项目路径（ROOT 相对路径，统一管理）
# ============================================================

@dataclass(frozen=True)
class Paths:
    """
    MUL03 工程路径集合（集中管理，避免散落在脚本里写死）
    """
    root: Path

    @property
    def ert_meta_dir(self) -> Path:
        return self.root / "EFM5" / "data" / "ert" / "ert_meta"

    @property
    def train_gpr_dir(self) -> Path:
        return self.root / "EFM5" / "data" / "gpr" / "train"

    @property
    def pairs_csv(self) -> Path:
        return self.root / "EFM5" / "data" / "pairs.csv"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "gpr_generationV2" / "outputs"
