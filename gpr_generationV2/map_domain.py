# -*- coding: utf-8 -*-
"""
gpr_generationV2/map_domain.py
============================================================
Stage GPR-V2 : ERT → GPR 坐标映射（弱物理对齐的核心）

背景与目标：
  - ERT 模型域：x ∈ [x_min, x_max]（单位：m），z 为深度（单位：m）
  - GPR（gprMax）域：x ∈ [0, domain_x]，y ∈ [0, domain_y]
  - 本项目为“土石堤坝浅层异常体检测”，重点深度：
      主范围：0–20 m
      可选辅助：20–30 m（置信度更低，不作为主对象）

关键约定（与你当前天线位置一致）：
  - 你当前天线 y≈1.45（接近 domain_y=1.5）
  - 因此我们采用：浅层（z 小）→ y 更靠近 domain_y；深层（z 大）→ y 更靠近 0
  - 即：深度 z 与 y 方向“反向映射”（浅→上、深→下）

输出：
  - MapResult：包含映射后的 (x_gpr, y_gpr, r_gpr) 以及置信度 confidence

============================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class MapResult:
    x_gpr: float
    y_gpr: float
    r_gpr: float
    confidence: float
    z_map_range: Tuple[float, float]


def _linear_map(v: float, vmin: float, vmax: float, out_min: float, out_max: float) -> float:
    """线性映射 + 裁剪到 [0,1]。"""
    if vmax <= vmin:
        return (out_min + out_max) * 0.5
    t = (v - vmin) / (vmax - vmin)
    if t < 0.0:
        t = 0.0
    if t > 1.0:
        t = 1.0
    return out_min + t * (out_max - out_min)


def map_ert_to_gpr(
    x_ert: float,
    z_ert: float,
    r_ert: float,
    ert_domain: Dict,
    gpr_domain: Dict,
    *,
    shallow_max_m: float = 5.0,#深度Z方向“20 m 主线”改成“5 m 主线”
    aux_max_m: float = 8.0,# 30改8备选：浅层扩展（仅用于消融 / 低置信度样本）
    base_confidence: float = 0.9,
) -> MapResult:
    """
    将  ERT 的 (x,z,r) 映射到 GPR(gprMax) 的 (x,y,r)。

    [MOD-LOCAL] 支持“局部有效域”：
      - 若 ert_domain 中存在 x_local_min/x_local_max（或 x_extent），则优先使用
      - 否则回退到全局 x_min/x_max

    目的：
      - x 映射与 r 缩放由“局部域”决定，避免 anomaly 被全局域稀释成小点
    """
     # ----------------------------
    # [MOD-LOCAL-1] 解析 ERT x 域（局部优先，全局回退）
    # ----------------------------
    # 兼容两种注入方式：
    # 1) ert_domain["x_local_min"/"x_local_max"]
    # 2) ert_domain["x_extent"] = [xmin, xmax]
    if "x_local_min" in ert_domain and "x_local_max" in ert_domain:
        x_min = float(ert_domain["x_local_min"])
        x_max = float(ert_domain["x_local_max"])
    elif "x_extent" in ert_domain and isinstance(ert_domain["x_extent"], (list, tuple)) and len(ert_domain["x_extent"]) == 2:
        x_min = float(ert_domain["x_extent"][0])
        x_max = float(ert_domain["x_extent"][1])
    else:
        # x_min = float(ert_domain["x_min"])
        # x_max = float(ert_domain["x_max"])
        x_min = float(
            ert_domain.get("x_local_min", ert_domain["x_min"])
        )
        x_max = float(
            ert_domain.get("x_local_max", ert_domain["x_max"])
        )

    # ----------------------------
    # z_min 仍使用全局或默认 0（你的逻辑不变）
    # ----------------------------
    z_min = float(ert_domain.get("z_min", 0.0))

    domain_x = float(gpr_domain["x"])
    domain_y = float(gpr_domain["y"])

    # 土体上界面（地表）位置（你原逻辑不变）
    y_soil_top = float(gpr_domain.get("y_soil_top", domain_y))

    # 深度映射窗口（你原逻辑不变：用 gpr_domain 可覆盖）
    z_map_min = float(gpr_domain.get("z_map_min_m", 4.0))
    z_map_max = float(gpr_domain.get("z_map_max_m", 18.0))
    conf = float(base_confidence)

    # x：正向映射（现在用“局部 x_min/x_max”）
    xg = _linear_map(x_ert, x_min, x_max, 0.0, domain_x)

    # 深度仅保护下限（你原逻辑不变）
    z_ert_used = max(z_ert, z_map_min)

    # depth_sim（你原逻辑不变）
    depth_sim = _linear_map(
        z_ert_used, z_map_min, z_map_max,
        0.35,   # depth_sim_min_m
        0.90,   # depth_sim_max_m
    )
    yg = y_soil_top - depth_sim

    # r：按 ERT 横向范围比例缩放（关键：现在用“局部域宽度”）
    rg = (r_ert / max(1e-9, (x_max - x_min))) * domain_x
    if rg < 1e-4:
        rg = 1e-4

    # y 边界安全裁剪（你原逻辑不变）
    y_margin = float(gpr_domain.get("y_margin_m", 0.05))
    y_low = y_margin + rg
    y_high = y_soil_top - y_margin - rg

    if y_high <= y_low:
        yg = 0.5 * (y_low + y_high)
    else:
        if yg < y_low:
            yg = y_low
        elif yg > y_high:
            yg = y_high

    return MapResult(
        x_gpr=xg,
        y_gpr=yg,
        r_gpr=rg,
        confidence=conf,
        z_map_range=(z_map_min, z_map_max),
    )
    
    # """
    # 将 ERT 的 (x,z,r) 映射到 GPR(gprMax) 的 (x,y,r)。

    # 1) x 映射：ERT [x_min,x_max] → GPR [0,domain_x]
    # 2) z 映射：ERT 深度（越浅 z 越小）→ GPR y 越接近 domain_y（天线所在“上方”）
    #    [MOD] 使用反向映射：z 小 → y 大；z 大 → y 小
    # 3) r 映射：按 ERT 横向尺度比例映射到 GPR x 尺度
    # 4) 置信度：
    #    - z<=20m：置信度 = base_confidence
    #    - 20-30m：置信度降低（辅助样本）
    # """
    
    # x_min = float(ert_domain["x_min"])
    # x_max = float(ert_domain["x_max"])
    # z_min = float(ert_domain.get("z_min", 0.0))

    # domain_x = float(gpr_domain["x"])
    # domain_y = float(gpr_domain["y"])

    # # ============================================================
    # # [MOD-1] 土体上界面（地表）位置
    # # - 如果 gpr_domain 未提供，则退化为 domain_y（兼容旧逻辑）
    # # - 后续 y 坐标应围绕 y_soil_top 计算，而不是直接用 domain_y
    # # ============================================================
    # y_soil_top = float(gpr_domain.get("y_soil_top", domain_y))

    # # # 深度映射范围选择（浅层优先）【置信度（浅层）】
    # # if z_ert <= shallow_max_m:
    # #     # Shallow region: physically visible by GPR (mainline assumption)
    # #     z_map_min, z_map_max = z_min, shallow_max_m
    # #     conf = float(base_confidence)
    # # else:
    # #        # Deeper region: weak or invisible for GPR, keep with reduced confidence
    # #     z_map_min, z_map_max = z_min, aux_max_m
    # #     conf = min(float(base_confidence) * 0.6, 0.6)

    # # ============================================================
    # # [TEST-2] 统一 ERT → GPR 深度映射窗口（不再分段）
    # # 主线深度：4–18 m（与你 PPT 中的有效深度一致）
    # # ============================================================
    # # z_map_min = 4.0
    # # z_map_max = 18.0
    # # conf = float(base_confidence)
    # z_map_min = float(gpr_domain.get("z_map_min_m", 4.0))
    # z_map_max = float(gpr_domain.get("z_map_max_m", 18.0))
    # conf = float(base_confidence)



    # # x：正向映射
    # xg = _linear_map(x_ert, x_min, x_max, 0.0, domain_x)

    # # # ============================================================
    # # # [MOD-2A] 显式裁剪深度，避免 z > aux_max_m 被压到 y=0
    # # # ============================================================
    # # z_ert_used = z_ert
    # # depth_clipped = False
    # # if z_ert_used < z_map_min:
    # #     z_ert_used = z_map_min
    # # if z_ert_used > z_map_max:
    # #     z_ert_used = z_map_max
    # #     depth_clipped = True
    
    # # ============================================================
    # # [TEST-2] 仅保护下限，不再压缩上限
    # # ============================================================
    # z_ert_used = max(z_ert, z_map_min)

    # # y：反向映射（浅层→更靠近 domain_y）
    # # [MOD] 关键：out_min=domain_y, out_max=0.0
    # # z 小（浅） → y 接近 domain_y（天线附近）
    # # z 大（深） → y 接近 0
    # #yg = _linear_map(z_ert, z_map_min, z_map_max, domain_y, 0.0)

    # # ============================================================
    # # [MOD-2B] 深度 → 仿真埋深带 → 地表坐标
    # # - 先把 ERT 深度映射到一个“可见埋深区间”
    # # - 再用 y = y_soil_top - depth_sim
    # # ============================================================
    # # depth_sim = _linear_map(
    # #     z_ert_used, z_map_min, z_map_max,
    # #     # gpr_domain.get("depth_sim_min_m", 0.4),
    # #     # gpr_domain.get("depth_sim_max_m", 0.8),
    # #     #埋深拉深
    # #     # depth_sim_min_m = 0.8,
    # #     # depth_sim_max_m = 1.2,
    # #     gpr_domain.get("depth_sim_min_m", 0.6),
    # #     gpr_domain.get("depth_sim_max_m", 1.6),
    # # )
    
    # depth_sim = _linear_map(
    # z_ert_used, z_map_min, z_map_max,
    # # 0.25,   # depth_sim_min_m
    # # 1.05,   # depth_sim_max_m（< y_soil_top - margin）
    # 0.35,   # depth_sim_min_m
    # 0.90,   # depth_sim_max_m
    # )

    # yg = y_soil_top - depth_sim


    # # r：按 ERT 横向范围比例缩放
    # rg = (r_ert / max(1e-9, (x_max - x_min))) * domain_x
    # if rg < 1e-4:
    #     rg = 1e-4
        
    # # ============================================================
    # # [MOD-3] y 边界安全裁剪（防止异常体进入空气层）
    # # 物理语义：
    # #   - 异常体中心必须位于土体层内
    # #   - soil 区间定义为: (rg + margin, y_soil_top - rg - margin)
    # # ============================================================
    # y_margin = float(gpr_domain.get("y_margin_m", 0.05))

    # # soil 内允许的最小 / 最大 y（以异常体“完整落入土体”为准）
    # y_low  = y_margin + rg
    # y_high = y_soil_top - y_margin - rg

    # # [MOD-3A] 若 soil 区间本身非法（极端参数），退化为居中
    # if y_high <= y_low:
    #     yg = 0.5 * (y_low + y_high)
    # else:
    #     # [MOD-3B] 强制限制异常体中心位于 soil 区间
    #     if yg < y_low:
    #         yg = y_low
    #     elif yg > y_high:
    #         yg = y_high


    # return MapResult(
    #     x_gpr=xg,
    #     y_gpr=yg,
    #     r_gpr=rg,
    #     confidence=conf,
    #     z_map_range=(z_map_min, z_map_max),
    # )
