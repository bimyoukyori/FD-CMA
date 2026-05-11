# -*- coding: utf-8 -*-
from __future__ import annotations
"""
run_batch.py
============================================================
Stage GPR-V2 : 批量 GPR 样本生成调度器

功能说明：
1. 扫描 ERT meta 文件（EFM5/data/ert/ert_meta）
2. 调用 build_anomaly 构建 GPR 几何体（ERT → GPR 映射）
3. 调用 simulate.run_one 执行 gprMax 正演
4. 生成并整理输出：
   - .in / .out / .png / .csv / meta.json
5. 维护样本审计表 gpr_samples_v2.csv

版本信息：
- v2.0  初始批量生成版本
- v2.1  引入 y_soil_top，明确空气层/土体语义
- v2.2  批量稳定性修复（文件头 / future import）

注意事项：
- 本文件必须保证 from __future__ import annotations 位于所有 import 之前
- 不得在 future import 之前插入任何代码
============================================================
"""
import os
import sys

# ====================================================================
# 🟢 [修改点] 万能启动头 (必须在 import gprMax/simulate 之前执行)
# ====================================================================
print("[INIT] 正在配置 GPU 编译环境...")
os.environ['PYCUDA_DEFAULT_NVCC_FLAGS'] = '-m64 -Xcompiler "/std:c++14"'

# 自动挂载 VS 编译器路径
vs_path = r"S:\1software\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"
if os.path.exists(vs_path):
    if vs_path not in os.environ['PATH']:
        os.environ['PATH'] = vs_path + ";" + os.environ['PATH']
    print(f"[INIT] VS 编译器路径确认: {vs_path}")
else:
    print("[INIT] 警告: 未找到指定的 VS 路径，尝试使用系统默认...")
# ====================================================================


import csv
import json
from pathlib import Path
from typing import Dict, List
import random
import shutil

from config import find_project_root, GPRDomain, GPRGrid, GPRSimParams, Paths
from build_anomaly import build_geometry_lines
from simulate import run_one


# ============================================================
# 运行时配置
# ============================================================

USE_GPU = True   
GPU_ID = 0       
LIMIT_SAMPLES = 20000   
DRY_RUN = False      


# ============================================================
# 工具函数
# ============================================================

def _id_from_meta(meta: Dict) -> str:
    mid = meta.get("id", "")
    if isinstance(mid, str) and mid.startswith("model_"):
        return mid.split("_", 1)[1]
    return str(mid).strip()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _write_gpr_samples_csv(csv_path: Path, rows: List[Dict]) -> None:
    """
    写入 GPR 样本生成阶段的审计 CSV（不用于训练对齐）

    语义说明：
    - 一行 = 一个 ERT → GPR 映射尝试
    - 不要求全部成功
    - 用于统计失败率、可见性、深度裁剪情况
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "class",
        "ert_json",
        "gpr_csv",       # [MOD] 原 gpr_png -> gpr_csv
        "confidence",
        "visible",
        "depth_clipped",
        "status",
        "error",
    ]

    exists = csv_path.exists()

    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()

        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def _deterministic_seed(short_id: str) -> int:
    try:
        return int(short_id)
    except Exception:
        return abs(hash(short_id)) % 10_000_000


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    gpu_id = GPU_ID if USE_GPU else None
    limit = LIMIT_SAMPLES

    root = find_project_root()
    paths = Paths(root=root)

    ert_meta_dir = paths.ert_meta_dir
    outputs_root = paths.outputs_dir
    train_root = paths.train_gpr_dir
    pairs_csv_path = paths.pairs_csv

    domain = GPRDomain()
    grid = GPRGrid()
    base = GPRSimParams()

    # ============================================================
    # [SURG-DOMAIN-1] 土体 / 空气层语义（GPR-V2）
    # - y_soil_top: 土体上界（地表）
    # - domain.y   : 总高度 = 土体 + 空气
    # ============================================================
    #domain.y_soil_top = 1.2   # 土体厚度（m）
    # 约定：
    #   soil  : y ∈ [0, 1.2]
    #   air   : y ∈ [1.2, domain.y]

    
    if not ert_meta_dir.exists():
        raise FileNotFoundError(f"ERT meta dir not found: {ert_meta_dir}")

    # meta_files = sorted(ert_meta_dir.glob("*.json"))
    # if limit is not None:
    #     meta_files = meta_files[: int(limit)]
    
    #按文件名读取ERT里面的参数*.json文件
    # ============================================================
    # [MOD] 文件驱动的 ERT meta 读取方式（替代编号递增）
    # ============================================================
    meta_files = sorted(ert_meta_dir.glob("*.json"))
    if limit is not None:
        meta_files = meta_files[: int(limit)]

    #print(f"[INFO] ERT meta    : {ert_meta_dir}  (files={len(meta_files)})")



    print(f"[INFO] Project root: {root}")
    print(f"[INFO] ERT meta    : {ert_meta_dir}  (files={len(meta_files)})")
    print(f"[INFO] Outputs     : {outputs_root}")
    print(f"[INFO] Train pool  : {train_root}")
    print(f"[INFO] GPU         : {gpu_id}")

    #修改日志CSV写入定义
    gpr_sample_rows: List[Dict] = []

    for meta_path in meta_files:
        # --- 1. 元数据解析 ---
        meta = _read_json(meta_path)
        cls = str(meta.get("class", "")).strip()
        short_id = meta_path.stem

        # meta_id = _id_from_meta(meta)
        # if meta_id and meta_id.isdigit() and len(meta_id) == len(short_id):
        #     short_id = meta_id
        meta_id = _id_from_meta(meta)  # 仅用于记录，不参与 short_id 决策
        
        if cls not in {"cavity", "loose", "crack", "normal"}:
            print(f"[SKIP] {short_id}: unknown class={cls}")
            continue

        # --- 2. 断点续传检测 ---
        train_cls_dir = train_root / cls
        dst_csv  = train_cls_dir / f"{short_id}.csv"#对data下面的汇总结果文件 .png修改为 .csv

        if dst_csv.exists():
            print(f"[SKIP] {short_id} already exists. ({dst_csv.name})")
            continue

        # --- 3. 准备仿真参数 ---
        sim_params = base.__dict__.copy()
        seed = _deterministic_seed(short_id)
        sim_params["seed"] = seed
        
        # [修改确认] 针对 loose 类，提前计算参数
        if cls == "loose":
            water_base = float(base.water)
            sim_params["water_loose"] = min(max(water_base + 0.10, 0.01), 0.35)
            
        if cls == "normal":
            rnd = random.Random(seed)
            sim_params["seed"] = rnd.randint(1, 10_000_000)
            sim_params["water"] = max(0.01, min(0.35, base.water + rnd.uniform(-0.03, 0.03)))

        # --- 4. 生成几何 ---
        # 注意：这里的 geometry_lines 将只包含 #box 等形状，不含材料定义
        geom_lines, extra_meta = build_geometry_lines(
            meta,
            #gpr_domain={"x": domain.x, "y": domain.y, "z": domain.z},
            gpr_domain={
                "x": domain.x,
                "y": domain.y,
                "z": domain.z,
                "y_soil_top": base.soil_thickness,   # [MOD] 关键修复
            },


            base_sim_params={
                "sand": sim_params["sand"],
                "clay": sim_params["clay"],
                "density": sim_params["density"],
                "sand_density": sim_params["sand_density"],
                "conductivity": sim_params["conductivity"],
                "water": sim_params["water"],

                # [MOD] 同步传入 fractal 参数与 seed：
                #       - loose 现改为 #fractal_box，需要这些参数生成“最小侵入”的疏松区分形盒
                #       - 与 simulate.py 的背景 #fractal_box 保持物理语义一致（Peplinski + fractal soil）
                "fractal_n": sim_params.get("fractal_n", base.fractal_n),
                "fractal_ax": sim_params.get("fractal_ax", base.fractal_ax),
                "fractal_ay": sim_params.get("fractal_ay", base.fractal_ay),
                "fractal_az": sim_params.get("fractal_az", base.fractal_az),
                "materials": sim_params.get("materials", base.materials),
                "seed": sim_params.get("seed", seed),
            },
        )

        out_dir = outputs_root / cls / short_id
        _ensure_dir(out_dir)

        # --- 5. 运行仿真 ---
        try:
            print(f"[RUN] {cls}/{short_id} (gpu={gpu_id})")

            if DRY_RUN:
                print(f"      - out_dir: {out_dir}")
                continue

            result = run_one(
                sample_id=short_id,
                out_dir=out_dir,
                sim_params=sim_params,
                #domain={"x": domain.x, "y": domain.y, "z": domain.z},
                domain={
                    "x": domain.x,
                    "y": domain.y,
                    "z": domain.z,
                    # ============================================================
                    # [FUNC] 土体/空气层语义闭合（run_batch -> simulate）
                    # 目的：
                    #   1) 让 simulate.py 写入 .in 时明确区分：土体区 y∈[0, y_soil_top]、空气区 y∈(y_soil_top, domain.y]
                    #   2) 保证天线位置 y = y_soil_top + air_gap，严格位于空气层
                    # 约定：
                    #   y_soil_top 来源于 config.GPRSimParams.soil_thickness
                    # ============================================================
                    "y_soil_top": float(base.soil_thickness),

                    # 可选：若后续要在 map_domain 中统一 margin，也可以在此处集中注入
                    # "y_margin_m": 0.05,
                                
                },
                            
                grid={"dx": grid.dx, "dy": grid.dy, "dz": grid.dz},
                geometry_lines=geom_lines,
                extra_meta=extra_meta,
                gpu_id=gpu_id,
            )


                        
            _ensure_dir(train_cls_dir)
            shutil.copy2(Path(result["csv"]), dst_csv)

            gpr_sample_rows.append({
                "id": short_id,
                "class": cls,
                "ert_json": str(meta_path.relative_to(root)).replace("\\", "/"),
               # "gpr_png": str(dst_png.relative_to(root)).replace("\\", "/"),
                "gpr_csv": str(dst_csv.relative_to(root)).replace("\\", "/"),
                "confidence": float(extra_meta.get("confidence", 0.0)),

                # —— 审计字段（关键）——
                "visible": bool(extra_meta.get("confidence", 0.0) >= 0.5),
                "depth_clipped": bool(
                    extra_meta.get("mapping", {})
                    .get("z_map_range_m", [0, 0])[1] > 5.0
                ),

                "status": "ok",
                "error": "",
            })


            #print(f"[OK ] {cls}/{short_id} -> {dst_png}")
            print(f"[OK ] {cls}/{short_id} -> {dst_csv} (CSV saved)")

        except Exception as e:
            print(f"❌ [ERROR] Sample {short_id} failed.")
            print(f"   Reason: {str(e)}")

            gpr_sample_rows.append({
                "id": short_id,
                "class": cls,
                "ert_json": str(meta_path.relative_to(root)).replace("\\", "/"),
                # "gpr_png": "",
                "gpr_csv": "",
                "confidence": 0.0,
                "visible": False,
                "depth_clipped": "",
                "status": "failed",
                "error": str(e),
            })

            continue

    if gpr_sample_rows:
        gpr_csv_path = (
            paths.root
            / "EFM5"
            / "data"
            / "gpr"
            / "gpr_samples_v2.csv"
        )

        _write_gpr_samples_csv(gpr_csv_path, gpr_sample_rows)
        print(f"[DONE] gpr_samples_v2.csv updated: {gpr_csv_path}")
    else:
        print("[DONE] No GPR samples recorded.")



if __name__ == "__main__":
    main()
