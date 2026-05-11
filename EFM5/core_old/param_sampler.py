# -*- coding: utf-8 -*-
"""
param_sampler.py  【V2.2 最终稳定版｜论文可用】

------------------------------------------------------------
职责说明（非常重要）：
- 只负责：在给定“参数范围”内进行随机采样
- 不关心 GUI
- 不写死任何物理范围
- 不生成 anomaly_def.json
- 不涉及 ERT / FEM / 文件 IO

设计目标：
✔ 范围驱动（range-driven）
✔ 可复现（seed）
✔ 科研友好（每个参数可追溯）
✔ 与 GUI / 批量建模完全解耦
------------------------------------------------------------
"""

import random
from typing import Dict, Any, Optional


# ============================================================
# 随机种子控制（对外接口）
# ============================================================
def set_seed(seed: Optional[int] = None):
    """
    设置随机种子

    Parameters
    ----------
    seed : int or None
        - int  : 固定随机序列（可复现实验）
        - None : 完全随机
    """
    if seed is None:
        random.seed()
    else:
        random.seed(int(seed))


# ============================================================
# 内部工具函数
# ============================================================
def _uniform(vmin: float, vmax: float) -> float:
    """
    均匀分布采样（自动处理 vmin == vmax）

    Notes
    -----
    - 当 vmin == vmax 时，返回常数
    - 用于 angle / rho / r 等弱随机参数
    """
    if vmin == vmax:
        return vmin
    return random.uniform(vmin, vmax)


# ============================================================
# 主接口：按“范围字典”采样异常参数
# ============================================================
def sample_anomaly_params(
    anom_type: str,
    ranges: Dict[str, tuple],
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    在给定范围内，采样一组异常体参数

    Parameters
    ----------
    anom_type : str
        "normal" / "cavity" / "crack" / "loose"

    ranges : dict
        参数范围字典（由 GUI / 上层逻辑提供）
        例如：
        {
            "x":    (-40, 40),
            "z":    (10, 30),
            "r":    (4, 10),
            "rho":  (3000, 8000),
            "angle":(-45, 45)
        }

    seed : int or None
        可选随机种子（用于单次可复现采样）

    Returns
    -------
    params : dict
        {
            "anom_type": "crack",
            "enabled": True,
            "x0": 12.3,
            "z0": 18.7,
            "r":  2.4,
            "rho": 1350,
            "angle_deg": -23.6
        }
    """

    # -------------------------------
    # 种子控制（可选）
    # -------------------------------
    if seed is not None:
        random.seed(int(seed))

    # -------------------------------
    # normal：无异常，直接返回
    # -------------------------------
    if anom_type == "normal":
        return {
            "anom_type": "normal",
            "enabled": False
        }

    # -------------------------------
    # 基本校验
    # -------------------------------
    required_keys = ["x", "z", "r", "rho"]
    for k in required_keys:
        if k not in ranges:
            raise KeyError(f"缺少必要参数范围: {k}")

    # -------------------------------
    # 通用参数采样
    # -------------------------------
    x0 = _uniform(*ranges["x"])
    z0 = _uniform(*ranges["z"])
    r  = _uniform(*ranges["r"])
    rho = _uniform(*ranges["rho"])

    # -------------------------------
    # 裂隙专属：倾角
    # -------------------------------
    if anom_type == "crack":
        if "angle" not in ranges:
            raise KeyError("crack 类型必须提供 angle 范围")
        angle = _uniform(*ranges["angle"])
    else:
        angle = 0.0

    # -------------------------------
    # 输出（统一字段名）
    # -------------------------------
    params = {
        "anom_type": anom_type,
        "enabled": True,
        "x0": round(x0, 3),
        "z0": round(z0, 3),
        "r": round(r, 3),
        "rho": round(rho, 3),
        "angle_deg": round(angle, 3)
    }

    return params


# ============================================================
# 调试入口（仅本文件运行时）
# ============================================================
if __name__ == "__main__":

    demo_ranges = {
        "x": (-40, 40),
        "z": (10, 35),
        "r": (2, 3),
        "rho": (1000, 2000),
        "angle": (-45, 45)
    }

    print("DEBUG crack sample:")
    for i in range(3):
        print(sample_anomaly_params("crack", demo_ranges, seed=i))
