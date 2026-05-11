# -*- coding: utf-8 -*-
"""
gpr_generationV2/build_anomaly.py
============================================================
Stage GPR-V2 : 异常体构造（ERT meta → gprMax 几何语句）

【修改说明】
1. [CRITICAL FIX] loose 由 #box 改为 #fractal_box：
   - 根因：#soil_peplinski 创建的是“混合模型/材料混合器”，不是 #material。
     因此 my_soil_loose 不能作为 #box 的 str1(material id) 使用，否则会触发
     "material(s) ['my_soil_loose'] do not exist"。
   - 方案：保持你既有的“Peplinski + 分形土体”物理语义不变：
     loose 异常区域用第二个 #fractal_box 覆盖背景 #fractal_box。
   - 该修改属于“最小侵入”：不改 simulate.py 的材料定义逻辑，仅替换
     loose 的几何表达方式。

============================================================
"""
from __future__ import annotations#这个需要在第一行导入包

from typing import Dict, List, Tuple
import math
#from venv import logger#修改日志文件包
import logging
logger = logging.getLogger(__name__)

from map_domain import map_ert_to_gpr
import random

def _ensure_order(a: float, b: float) -> Tuple[float, float]:
    """确保 a<=b。"""
    return (a, b) if a <= b else (b, a)


def build_geometry_lines(
    meta: Dict,
    *,
    gpr_domain: Dict,
    base_sim_params: Dict,
) -> Tuple[List[str], Dict]:
    """
    输入：meta, gpr_domain, base_sim_params
    输出：lines (仅几何指令), extra_meta
    """
    cls = str(meta.get("class", "")).strip()
    sampled = meta.get("sampled", {}) or {}
    ert_domain = meta.get("domain", {}) or {}

    extra_meta: Dict = {"class": cls, "mapping": {}, "confidence": None}

    if cls == "normal" or not bool(sampled.get("enabled", False)):
        extra_meta["confidence"] = float(
            (meta.get("alignment_meta", {}) or {})
            .get("alignment_hint", {})
            .get("confidence_init", 0.3)
        )
        return [], extra_meta

    # 提取几何参数
    anomalies = (meta.get("ert", {}) or {}).get("anomalies", [])
    anom = anomalies[0] if anomalies else {"type": cls, "geometry": {}}
    geom = anom.get("geometry", {}) or {}

    x0 = float(geom.get("x0", sampled.get("x0", 0.0)))
    z0 = float(geom.get("z0", sampled.get("z0", 0.0)))
    r = float(geom.get("r", sampled.get("r", 1.0)))
    angle_deg = float(geom.get("angle_deg", sampled.get("angle_deg", 0.0)))

    # NOTE:
    # ERT anomaly_def.json angles are defined in degrees by design.
    # This is a project-level assumption and enforced here.
    
    #“在 GPR 几何构建阶段，裂隙倾角参数统一假设为角度制（degree），
    #并在代码中显式进行角度–弧度转换，以保证工程实现与物理建模假设一致。”
    angle_rad = math.radians(angle_deg)

    if abs(angle_deg) < 1.0:
        # 小角度是允许的（接近竖直裂隙），但记录以便审计
        # 防止 rad 单位误混入
        # 例如：0.3 如果被误当成 rad，将对应 ~17 deg
        try:
            logger.debug(
                f"[ANGLE] small crack angle detected: {angle_deg} deg (assumed)"
            )
        except NameError:
            pass
    
    # ============================================================
    # [SURG-LOCAL-1] 注入“局部有效域”到 ert_domain（供 map_domain.py 使用）
    # ============================================================
    ert_domain_local = dict(ert_domain)

    anchor = (meta.get("alignment_meta", {}) or {}).get("ert_anchor", {}) or {}
    x_ext = anchor.get("x_extent", None)
    z_ext = anchor.get("z_extent", None)

    # 可调：局部 padding（以 r 为尺度，避免贴边）
    pad_x = 1.0 * r
    pad_z = 1.0 * r

    if isinstance(x_ext, (list, tuple)) and len(x_ext) == 2:
        ert_domain_local["x_local_min"] = float(x_ext[0]) - pad_x
        ert_domain_local["x_local_max"] = float(x_ext[1]) + pad_x

    # z_extent 这里不直接进入 x/r 缩放，但保留用于审计或未来扩展
    if isinstance(z_ext, (list, tuple)) and len(z_ext) == 2:
        ert_domain_local["z_local_min"] = float(z_ext[0]) - pad_z
        ert_domain_local["z_local_max"] = float(z_ext[1]) + pad_z

    
    mr = map_ert_to_gpr(
        x0, z0, r,
        #ert_domain=ert_domain,
        ert_domain=ert_domain_local,   # [MOD] 改这里
        gpr_domain=gpr_domain,
        base_confidence=float(
            (meta.get("alignment_meta", {}) or {})
            .get("alignment_hint", {})
            .get("confidence_init", 0.9)
        ),
    )

    extra_meta["confidence"] = mr.confidence
    extra_meta["mapping"] = {
        "x0_ert": x0, "z0_ert": z0, "r_ert": r, "angle_deg": angle_deg,
        "x0_gpr": mr.x_gpr, "y0_gpr": mr.y_gpr, "r_gpr": mr.r_gpr,
        "z_map_range_m": list(mr.z_map_range),
    }

    domain_z = float(gpr_domain["z"])
    lines: List[str] = []
    

    # === Cavity ===
    if cls == "cavity":
        lines.append(
            f"#cylinder: {mr.x_gpr:.6f} {mr.y_gpr:.6f} 0 "
            f"{mr.x_gpr:.6f} {mr.y_gpr:.6f} {domain_z:.6f} {mr.r_gpr:.6f} free_space"
        )
        return lines, extra_meta

    # === Loose ===
    # ============================================================
    # loose：疏松/渗漏带
    # ============================================================
    if cls == "loose":
        # [MOD] loose 必须用 #fractal_box，而不是 #box:
        #       - my_soil_loose 来自 #soil_peplinski，是混合模型 ID，不是 #material ID
        #       - #box 只能接收 #material 定义的 material id
        #       因此这里用第二个 #fractal_box 覆盖背景土体，实现“疏松区更高含水范围”的语义。

        # [MOD] 显式清空列表（loose 只返回一个覆盖用的 fractal_box）
        lines = []

        # 计算几何范围 (保持原逻辑不变)
        anchor = (meta.get("alignment_meta", {}) or {}).get("ert_anchor", {}) or {}
        x_ext = anchor.get("x_extent", [x0 - 2 * r, x0 + 2 * r])
        z_ext = anchor.get("z_extent", [z0 - 2 * r, z0 + 2 * r])

        x1_ert, x2_ert = float(x_ext[0]), float(x_ext[1])
        z1_ert, z2_ert = float(z_ext[0]), float(z_ext[1])

        # [MOD] 修复 keyword-only 参数调用（map_ert_to_gpr 的 ert_domain/gpr_domain 为关键字参数）
        # x1g = map_ert_to_gpr(x1_ert, z0, r, ert_domain=ert_domain, gpr_domain=gpr_domain).x_gpr
        # x2g = map_ert_to_gpr(x2_ert, z0, r, ert_domain=ert_domain, gpr_domain=gpr_domain).x_gpr
        # y1g = map_ert_to_gpr(x0, z1_ert, r, ert_domain=ert_domain, gpr_domain=gpr_domain).y_gpr
        # y2g = map_ert_to_gpr(x0, z2_ert, r, ert_domain=ert_domain, gpr_domain=gpr_domain).y_gpr
        x1g = map_ert_to_gpr(x1_ert, z0, r, ert_domain=ert_domain_local, gpr_domain=gpr_domain).x_gpr
        x2g = map_ert_to_gpr(x2_ert, z0, r, ert_domain=ert_domain_local, gpr_domain=gpr_domain).x_gpr
        y1g = map_ert_to_gpr(x0, z1_ert, r, ert_domain=ert_domain_local, gpr_domain=gpr_domain).y_gpr
        y2g = map_ert_to_gpr(x0, z2_ert, r, ert_domain=ert_domain_local, gpr_domain=gpr_domain).y_gpr

        # # x 方向轻微扰动（打破完全平直）新增
        # dx_jitter = rnd.uniform(-0.03, 0.03)   # 2–5 cm 级
        # x1s = x1g + dx_jitter
        # x2s = x2g - dx_jitter
        
        x1g, x2g = _ensure_order(x1g, x2g)
        y1g, y2g = _ensure_order(y1g, y2g)
        
        #x1s, x2g = _ensure_order(x1g, x2s)
        
        # ============================================================
        # [SURG-LOOSE-1] 多段 loose：打散水平边界相干性
        # ============================================================
        #新增
        n_segments = 3  # 建议 3–4
        total_height = y2g - y1g
        seg_height = total_height / n_segments

        # y 扰动幅度（米）
        # 2–5 cm 是 GPR 中“足够打散但不过度”的经验范围
        jitter_min = 0.02
        jitter_max = 0.05


        # [MOD] loose 区域用 #fractal_box 覆盖背景 #fractal_box
        #       关键参数从 base_sim_params 读取，确保与 simulate.py 生成背景分形土体一致
        fractal_n = float(base_sim_params.get("fractal_n", 1.5))

        # [MOD] 与 simulate.py / config.py 的 key 对齐（fractal_ax/fractal_ay/fractal_az）
        ax = float(base_sim_params.get("fractal_ax", base_sim_params.get("ax", 1.0)))
        ay = float(base_sim_params.get("fractal_ay", base_sim_params.get("ay", 1.0)))
        az = float(base_sim_params.get("fractal_az", base_sim_params.get("az", 1.0)))
        n_materials = int(base_sim_params.get("materials", 50))

        # [MOD] 为 loose 分形盒提供一个独立且可复现的 seed
        #       - 优先使用本样本的 seed（run_batch.py 会传入）
        #       - 若缺失，则退化为 0（仍可运行，但随机性会由 gprMax 内部决定）
        seed_bg = int(base_sim_params.get("seed", 0))
        seed_loose = seed_bg + 100000  # 与背景 seed 分离，避免同一随机场导致对比度不明显

        # [MOD] 生成 fractal_box 标识符（必须是无空格字符串）
        sample_id = str(meta.get("id", meta.get("sample_id", ""))).strip() or "unknown"
        fractal_id = f"my_fractal_loose_{sample_id}"

        # lines.append(
        #     f"#fractal_box: {x1g:.6f} {y1g:.6f} 0 {x2g:.6f} {y2g:.6f} {domain_z:.6f} "
        #     f"{fractal_n:.6f} {ax:.6f} {ay:.6f} {az:.6f} {n_materials:d} my_soil_loose {fractal_id} {seed_loose:d}"
        # )
        # return lines, extra_meta
        rnd = random.Random(seed_loose)

        for i in range(n_segments):

            # 基本分段
            y1_seg = y1g + i * seg_height
            y2_seg = y1g + (i + 1) * seg_height

            # y 方向扰动（上下边界都扰动，但不交叉）
            dy1 = rnd.uniform(-jitter_max, jitter_max)
            dy2 = rnd.uniform(-jitter_max, jitter_max)
            # sign = rnd.choice([-1.0, 1.0])
            # dy1 = sign * rnd.uniform(jitter_min, jitter_max)
            # dy2 = sign * rnd.uniform(jitter_min, jitter_max)

            y1s = y1_seg + dy1
            y2s = y2_seg + dy2

            # 保证几何合法
            y1s, y2s = _ensure_order(y1s, y2s)

            # 防止越界
            y_soil_top = float(gpr_domain.get("y_soil_top", gpr_domain["y"]))
            y1s = max(0.0, y1s)
            #y2s = min(float(gpr_domain["y"]), y2s)
            y2s = min(y_soil_top, y2s)
            
             # ===== 手术-3核心：x 方向轻微错动 =====
             #新增“轻微 x 偏移”
            # [SURG-LOOSE-3A] x 方向轻微错动（2–4 cm），破坏垂直对齐
            dx_shift = rnd.uniform(-0.03, 0.03)
            
            # 每段使用略微不同的 seed
            seg_seed = seed_loose + (i + 1) * 1000

            seg_id = f"{fractal_id}_seg{i}"
            
            #append 前加二次 clamp：
            domain_x = float(gpr_domain["x"])

            x1s = max(0.0, min(x1g + dx_shift, domain_x))
            x2s = max(0.0, min(x2g + dx_shift, domain_x))
            if x2s <= x1s:
                continue
            
            # lines.append(
            #     # f"#fractal_box: {x1g:.6f} {y1s:.6f} 0 "
            #     # f"{x2g:.6f} {y2s:.6f} {domain_z:.6f} "
            #     f"#fractal_box: {(x1g + dx_shift):.6f} {y1s:.6f} 0 "
            #     f"{(x2g + dx_shift):.6f} {y2s:.6f} {domain_z:.6f} "

            #     f"{fractal_n:.6f} {ax:.6f} {ay:.6f} {az:.6f} "
            #     f"{n_materials:d} my_soil_loose {seg_id} {seg_seed:d}"
            # )
            lines.append(
                f"#fractal_box: {x1s:.6f} {y1s:.6f} 0 "
                f"{x2s:.6f} {y2s:.6f} {domain_z:.6f} "
                f"{fractal_n:.6f} {ax:.6f} {ay:.6f} {az:.6f} "
                f"{n_materials:d} my_soil_loose {seg_id} {seg_seed:d}"
            )

        return lines, extra_meta 
        
    
    # === Crack ===
    if cls == "crack":
        width = mr.r_gpr
        anchor = (meta.get("alignment_meta", {}) or {}).get("ert_anchor", {}) or {}
        depth_focus = anchor.get("depth_focus", None)

        if depth_focus and len(depth_focus) == 2:
            zf1, zf2 = float(depth_focus[0]), float(depth_focus[1])
            L_ert = max(2.0, abs(zf2 - zf1))
        else:
            L_ert = max(4.0, 8.0 * r)

        z_map_min, z_map_max = mr.z_map_range
        domain_y = float(gpr_domain["y"])
        scale_y = domain_y / max(1e-9, (z_map_max - z_map_min))
        L_gpr = L_ert * scale_y

        #theta = math.radians(angle_deg)
        theta = angle_rad
        dx = (L_gpr * 0.5) * math.cos(theta)
        dy = (L_gpr * 0.5) * math.sin(theta)

        x_start = mr.x_gpr - dx
        y_start = mr.y_gpr - dy
        x_end = mr.x_gpr + dx
        y_end = mr.y_gpr + dy

        # --------------------------------------------------
        # 段级统计：用于判断“整条 crack 是否完全退化”
        # --------------------------------------------------
        valid_seg_count = 0 #判断“整条 crack 是否完全退化”
        n_seg = 10
        for i in range(n_seg):
            t1 = i / n_seg
            t2 = (i + 1) / n_seg
            xs1 = x_start + (x_end - x_start) * t1
            ys1 = y_start + (y_end - y_start) * t1
            xs2 = x_start + (x_end - x_start) * t2
            ys2 = y_start + (y_end - y_start) * t2

            x_min = min(xs1, xs2) - width * 0.5
            x_max = max(xs1, xs2) + width * 0.5
            y_min = min(ys1, ys2) - width * 0.5
            y_max = max(ys1, ys2) + width * 0.5

            x_min = max(0.0, x_min)
            y_min = max(0.0, y_min)
            x_max = min(float(gpr_domain["x"]), x_max)
            y_max = min(float(gpr_domain["y"]), y_max)

            # --------------------------------------------------
            # 几何合法性约束（段级）
            # --------------------------------------------------
            dx = float(base_sim_params.get("dx", 0.0025))
            dy = float(base_sim_params.get("dy", 0.0025))
            k = 2.0  # 最小 2 个网格

            if (x_max - x_min) < k * dx or (y_max - y_min) < k * dy:
                # 当前 crack 段几何退化，跳过该段
                continue

            lines.append(
                f"#box: {x_min:.6f} {y_min:.6f} 0 {x_max:.6f} {y_max:.6f} {domain_z:.6f} free_space"
            )
            
            valid_seg_count += 1
        if valid_seg_count == 0:
                #返回前，加一个明确失败信号
            raise RuntimeError(
                f"All crack segments degenerated (geometry too small): "
                f"id={meta.get('id', 'unknown')}"
                )

        return lines, extra_meta

    return [], extra_meta
