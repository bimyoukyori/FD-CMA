# -*- coding: utf-8 -*-
"""
gpr_generationV2/simulate.py
============================================================
Stage GPR-V2 : 单样本 gprMax 仿真内核（不含 UI / 不读命令行）

【架构更新说明】
1. [策略升级] 实施“材料定义与几何形状分离”策略：
   - simulate.py 负责定义所有材料（Physics）：包括背景 my_soil 和异常 my_soil_loose。
   - build_anomaly.py 只负责生成几何形状（Geometry）：如 #box, #cylinder。
   - 材料定义被移至 .in 文件头部，确保先定义后使用。

2. [进程隔离] 使用 subprocess 替代 api() 直接调用：
   - 彻底隔离 GPU 显存，防止单次模拟崩溃导致的显存脏数据残留。

功能：
  1) 生成 gprMax 输入文件 .in
  2) 调用 gprMax API 执行正演（Subprocess 模式）
  3) 合并/后处理/绘图/归档
  
  
# NOTE: CSV 为训练/特征提取主数据；PNG 仅用于预览展示。
英文版（论文可直接引用）
# NOTE: CSV is the primary data for DCT/features; PNG is for visualization only.
============================================================
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib

# [MOD] 防止 Windows/无显示环境绘图闪退
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gprMax.gprMax import api
from tools.outputfiles_merge import merge_files
from tools.plot_Bscan import get_output_data

import subprocess  # <--- 必须导入，用于独立进程运行 gprMax


# ============================================================
# 1）写入 gprMax 输入文件（.in）
# ============================================================

def write_in_file(
    in_path: Path,
    sample_id: str,
    domain: Dict,
    grid: Dict,
    sim_params: Dict,
    geometry_lines: List[str],
) -> None:
    """
    生成单个样本的 gprMax 输入文件（.in）

    [重要逻辑确认] 写入顺序说明 (优化版)：
      1. 头部信息（#domain, #dx_dy_dz, #time_window）
      2. [移到这里] 材料定义 (#soil_peplinski)
         - 确保在定义激励源和几何体之前，材料已经存在。
         - 包含背景土 my_soil 和 疏松土 my_soil_loose (如有)。
      3. 激励源与接收器 (#waveform, #hertzian_dipole, #rx)
      4. 步进信息 (#src_steps, #rx_steps)
      5. 背景几何体 (#fractal_box)
      6. 异常体相关语句 (geometry_lines)
    """
    lines: List[str] = []

    # ----------------------------
    # 1. 头部信息
    # ----------------------------
    lines.append(f"#title: {sample_id}")
    lines.append(f"#domain: {domain['x']} {domain['y']} {domain['z']}")
    lines.append(f"#dx_dy_dz: {grid['dx']} {grid['dy']} {grid['dz']}")
    lines.append(f"#time_window: {sim_params['time_window']}")
    
    # ============================================================
    # [SURG-AIR-1] 地表/空气层语义
    # - y_soil_top: 土体上界（地表），土体填充范围 y ∈ [0, y_soil_top]
    # - (y_soil_top, domain['y']] 为自由空间（空气）
    # - air_gap_m: 天线离地间距（在空气中），典型 0.05~0.10 m
    # ============================================================
    y_domain = float(domain["y"])
    y_soil_top = float(domain.get("y_soil_top", y_domain))  # 未传则退化为“无空气层”
    air_gap = float(sim_params.get("air_gap_m", 0.05))

    # ----------------------------
    # 2. [关键修复] 材料定义 (Materials) - 必须最先定义！
    # ----------------------------
    
    # 2.1 定义背景土 (my_soil)
    lines.append(
        f"#soil_peplinski: {sim_params['sand']} {sim_params['clay']} "
        f"{sim_params['density']} {sim_params['sand_density']} "
        f"{sim_params['conductivity']} {sim_params['water']} my_soil"
    )

    # 2.2 定义疏松土 (my_soil_loose) - 强力兜底逻辑
    # 只要 geometry_lines 里有引用 my_soil_loose，这里就必须定义。
    # 我们检查 sim_params 是否有 water_loose，或者为了保险直接判断是否需要定义。
    # 这里做一个强力兜底：如果 sim_params 里没有 water_loose，但我们预感可能要用，就用默认值造一个。
    if "water_loose" in sim_params:
        w_loose = sim_params["water_loose"]
    else:
        # 默认比背景高 0.1，防止报错 (Fallback default)
        w_loose = min(float(sim_params['water']) + 0.1, 0.35)
    
    # 无论如何，定义它！定义了不用没关系，用了没定义才会报错。
    lines.append(
        f"#soil_peplinski: {sim_params['sand']} {sim_params['clay']} "
        f"{sim_params['density']} {sim_params['sand_density']} "
        f"{sim_params['conductivity']} {w_loose} my_soil_loose"
    )

    # ----------------------------
    # 3. 激励源与接收器
    # ----------------------------
    # lines.append(f"#waveform: ricker 1 {sim_params['freq']} my_ricker")
    # lines.append(f"#hertzian_dipole: z {sim_params['tx_x']} {sim_params['tx_y']} {sim_params['tx_z']} my_ricker")
    # lines.append(f"#rx: {sim_params['rx_x']} {sim_params['rx_y']} {sim_params['rx_z']}")
    lines.append(f"#waveform: ricker 1 {sim_params['freq']} my_ricker")

    # ============================================================
    # [SURG-AIR-2] 天线与接收器放在空气层中（贴近地表）
    # - y_tx = y_soil_top + air_gap
    # - y_rx = y_soil_top + air_gap
    # - 仅覆盖 y 坐标，x/z 仍使用外部传入的 sim_params（保持你现有步进逻辑）
    # ============================================================
    tx_y_air = y_soil_top + air_gap
    rx_y_air = y_soil_top + air_gap

    lines.append(f"#hertzian_dipole: z {sim_params['tx_x']} {tx_y_air} {sim_params['tx_z']} my_ricker")
    lines.append(f"#rx: {sim_params['rx_x']} {rx_y_air} {sim_params['rx_z']}")


    # ----------------------------
    # 4. 步进信息
    # ----------------------------
    lines.append(f"#src_steps: {sim_params['src_steps_x']} {sim_params['src_steps_y']} {sim_params['src_steps_z']}")
    lines.append(f"#rx_steps: {sim_params['rx_steps_x']} {sim_params['rx_steps_y']} {sim_params['rx_steps_z']}")

    # ----------------------------
    # 5. 几何结构 (Geometry)
    # ----------------------------
    
    # 5.1 背景分形盒
    # lines.append(
    #     f"#fractal_box: 0 0 0 {domain['x']} {domain['y']} {domain['z']} "
    #     f"{sim_params['fractal_n']} {sim_params['fractal_ax']} {sim_params['fractal_ay']} {sim_params['fractal_az']} "
    #     f"{sim_params['materials']} my_soil my_fractal {sim_params['seed']}"
    # )

    # ============================================================
    # [SURG-AIR-3] 背景土体仅填充到 y_soil_top（上方留为空气层）
    # - 若 y_soil_top == domain['y']，则退化为旧行为（无空气层）
    # ============================================================
    lines.append(
        f"#fractal_box: 0 0 0 {domain['x']} {y_soil_top} {domain['z']} "
        f"{sim_params['fractal_n']} {sim_params['fractal_ax']} {sim_params['fractal_ay']} {sim_params['fractal_az']} "
        f"{sim_params['materials']} my_soil my_fractal {sim_params['seed']}"
    )

    # 5.2 异常体几何 (追加在最后)
    # 此时 geometry_lines 应该只包含 #box 或 #cylinder，
    # 且因为材料已在第2步定义，这里引用 my_soil_loose 是安全的。
    for g in geometry_lines:
        lines.append(g)

    # ----------------------------
    # 6. 诊断
    # ----------------------------
    lines.append("Model_diag n")

    in_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 2）绘制 B-scan（灰度 + 99% 分位归一化）
# ============================================================

def save_bscan_png(outputdata: np.ndarray, png_path: Path) -> None:
    """
    保存灰度 B-scan 图（核心：避免全黑）
    - vmax 用 99% 分位，而不是 max(abs)
    """
    absdata = np.abs(outputdata)
    vmax = float(np.percentile(absdata, 99))
    if vmax <= 0:
        vmax = 1.0

    plt.figure(figsize=(6, 4))
    plt.imshow(
        outputdata,
        extent=[0, outputdata.shape[1], outputdata.shape[0], 0],
        interpolation="bicubic",
        aspect="auto",
        cmap="gray",
        vmin=-vmax,
        vmax=vmax,
    )
    plt.axis("off")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 3）主接口：运行单个样本
# ============================================================

def run_one(
    sample_id: str,
    out_dir: Path,
    sim_params: Dict,
    domain: Dict,
    grid: Dict,
    geometry_lines: List[str],
    extra_meta: Dict,
    gpu_id: Optional[int] = None,
) -> Dict:
    """
    执行单个样本的完整仿真流程（使用 subprocess 隔离模式）
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    in_path = out_dir / f"{sample_id}.in"
    csv_path = out_dir / f"{sample_id}.csv"
    png_path = out_dir / f"{sample_id}.png"
    meta_path = out_dir / f"{sample_id}_gpr.json"
    base_prefix = str(out_dir / sample_id)

    # 1. 写入 .in 文件
    write_in_file(
        in_path=in_path,
        sample_id=sample_id,
        domain=domain,
        grid=grid,
        sim_params=sim_params,
        geometry_lines=geometry_lines,
    )

    #新增插入
        # ============================================================
    # [SURG-NRUNS-1] 自动反推安全 n_runs（防止 B-scan 越界）
    # 原则：
    #   tx_x + (n_runs - 1) * step_x <= domain.x - safety_margin
    # ============================================================

    # 初始值（来自 config）
    n_runs_cfg = int(sim_params.get("n_runs", 1))
    if n_runs_cfg <= 0:
        n_runs_cfg = 1

    # 几何参数
    tx_x = float(sim_params.get("tx_x", 0.0))
    rx_x = float(sim_params.get("rx_x", tx_x))
    step_x = float(sim_params.get("src_steps_x", 0.0))
    domain_x = float(domain["x"])

    # 安全边界（避免贴 PML / 边界）
    safety_margin = 0.05  # 5 cm，工程级稳妥值

    # 可用扫描长度
    usable_length = domain_x - max(tx_x, rx_x) - safety_margin

    if step_x <= 0:
        # 不步进，退化为单道
        n_runs_max = 1
    else:
        n_runs_max = int(usable_length // step_x) + 1

    if n_runs_max < 1:
        n_runs_max = 1

    # 最终采用的 n_runs
    n_runs_final = min(n_runs_cfg, n_runs_max)

    if n_runs_final < n_runs_cfg:
        print(
            f"[WARN] n_runs clipped for safety: "
            f"{n_runs_cfg} -> {n_runs_final} "
            f"(domain.x={domain_x}, tx_x={tx_x}, step={step_x})"
        )

    # 覆盖 sim_params（关键）
    sim_params["n_runs"] = n_runs_final


    # 2. 调用 gprMax (使用 subprocess 隔离进程，防止 GPU 显存泄漏)
    n_runs = int(sim_params.get("n_runs", 1))
    if n_runs <= 0: n_runs = 1

    # 构造命令行参数
    # python -m gprMax path/to/file.in -n 1 -gpu 0
    cmd = [sys.executable, "-m", "gprMax", str(in_path), "-n", str(n_runs)]
    
    # GPU 参数控制
    env = os.environ.copy()
    if gpu_id is not None:
        cmd.extend(["-gpu", str(gpu_id)])
        # 确保子进程也能拿到正确的环境变量 (C++14 + 64位修复)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYCUDA_DEFAULT_NVCC_FLAGS"] = '-m64 -Xcompiler "/std:c++14"'
    
    print(f"[RUN] Subprocess: {' '.join(cmd)}")
    
    try:
        # 启动子进程运行 gprMax
        subprocess.run(cmd, env=env, check=True, cwd=out_dir)
        print(f"[OK] gprMax done: {sample_id}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] gprMax failed (return code {e.returncode})")
        raise RuntimeError(f"gprMax simulation failed for {sample_id}")

    # 3. 合并输出 (n_runs > 1)
    if n_runs > 1:
        merge_files(base_prefix, removefiles=True)
        out_file = f"{base_prefix}_merged.out"
    else:
        out_file = f"{base_prefix}.out"

    # 4. 后处理 (读取 Ez -> 去均值 -> 绘图)
   # ============================================================
    # 4. 后处理 (读取 Ez -> 去均值 -> 预处理 -> 保存)
    # ============================================================

    outputdata, dt = get_output_data(out_file, 1, "Ez")
    if outputdata is None or outputdata.size == 0:
        raise RuntimeError(f"Ez output is empty: {out_file}")

    # 保证是二维 (nt, nx)
    if outputdata.ndim == 1:
        outputdata = outputdata[:, np.newaxis]

    # ------------------------------------------------------------
    # [MOD-1] 基础直达波抑制（保持你原有逻辑）
    # ------------------------------------------------------------
    if outputdata.shape[1] > 1:
        outputdata = outputdata - np.mean(outputdata, axis=1, keepdims=True)
    else:
        outputdata = outputdata - np.mean(outputdata, axis=0, keepdims=True)

    # ------------------------------------------------------------
    # [SEMANTIC] GPR preprocess 参数（单一权威来源）
    # ------------------------------------------------------------
    preprocess_cfg = {
        "mute_ns": 8.0,
        "apply_gain": True,
        "gain_type": "power",
        "gain_power": 2.0,
        "apply_dewow": True,
        "dewow_win": 50
    }
    # ------------------------------------------------------------
    # [MOD-2] 时间域预处理（Test-1 核心）
    # ------------------------------------------------------------
    outputdata = preprocess_gpr(
        # NOTE:
        # Time-gate 会改变时间零点，当前实现只影响幅值处理，
        # 若后续需要物理深度标定，应同步记录 mute_ns 偏移量。
        outputdata,
        dt,
        **preprocess_cfg #这里已经包含下面的参数！
        # mute_ns=8.0,          # 静默前 8 ns（可调 5–10）
        # apply_gain=True,
        # gain_type="power",
        # gain_power=2.0,
        # apply_dewow=True,
        # dewow_win=50,
    )
    

    # ------------------------------------------------------------
    # [MOD-3] 保存 CSV（用于 DCT / 多模态）
    # ------------------------------------------------------------
    np.savetxt(csv_path, outputdata, delimiter=",")

    # ------------------------------------------------------------
    # [MOD-4] PNG：百分位归一化，避免浅层压顶
    # ------------------------------------------------------------
    save_bscan_png(
        outputdata,
        png_path
        #percentile=99      # 如果你的 save_bscan_png 支持
    )

    # 5. 保存 meta
    meta = {
        "id": sample_id,
        "class": extra_meta.get("class", None),
        "confidence": extra_meta.get("confidence", None),
        "mapping": extra_meta.get("mapping", {}),
        "sim_params": sim_params,
        "domain": domain,
        "grid": grid,
        "gpu": gpu_id,
        "out_file": out_file,
        # [SEMANTIC] GPR preprocess bookkeeping
        "preprocess": preprocess_cfg
        # [SEMANTIC] GPR preprocess bookkeeping
        # "preprocess": {
        #     "mute_ns": 8.0,
        #     "gain_type": "power",
        #     "gain_power": 2.0,
        #     "dewow_win": 50
        # }
    }
    
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "in": str(in_path),
        "out": str(out_file),
        "csv": str(csv_path),
        "png": str(png_path),
        "meta": str(meta_path),
    }
    
   
def preprocess_gpr(
    data,
    dt,
    mute_ns=8.0,
    apply_gain=True,
    gain_type="power",
    gain_power=2.0,
    apply_dewow=True,
    dewow_win=50
):
   # """
    #data: (nt, nx)
    #dt: time step (s)
   # """

    nt, nx = data.shape
    t = np.arange(nt) * dt  # seconds

    # --------------------------------------------------
    # 1. Time-gate（静默前 N ns）
    # --------------------------------------------------
    if mute_ns is not None and mute_ns > 0:
        mute_samples = int((mute_ns * 1e-9) / dt)
        if mute_samples < nt:
            data = data[mute_samples:, :]
    time_offset_ns = mute_ns if mute_ns is not None else 0.0
    # NOTE:
    # Time-gate 会改变时间零点（t=0 被整体后移 mute_ns），
    # 当前实现仅用于幅值预处理，不做物理深度标定。
    # 若后续需要深度反演 / 时间轴对齐，应显式引入 time_offset_ns。
    
    
    # --------------------------------------------------
    # 2. Dewow（滑动均值高通）
    # --------------------------------------------------
    #if apply_dewow and dewow_win > 1:
    if apply_dewow and dewow_win > 1 and data.shape[0] > dewow_win:
        kernel = np.ones(dewow_win) / dewow_win
        for ix in range(data.shape[1]):
            trend = np.convolve(data[:, ix], kernel, mode="same")
            data[:, ix] = data[:, ix] - trend

    # --------------------------------------------------
    # 3. Time-gain（随时间递增）
    # --------------------------------------------------
    if apply_gain:
        nt2 = data.shape[0]
        t2 = np.arange(nt2) * dt
        
        tmax = t2.max()
        if tmax <= 0:
            # 极端情况兜底：不做增益
            gain = np.ones_like(t2)
        else:    
            if gain_type == "power":
                gain = (t2 / t2.max()) ** gain_power
            elif gain_type == "linear":
                gain = t2 / t2.max()
            elif gain_type == "exp":
                gain = np.exp(t2 / t2.max()) - 1
            else:
                raise ValueError("Unknown gain_type")

        gain = gain.reshape(-1, 1)
        data = data * gain

    return data
