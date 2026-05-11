#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pathlib import Path

# ============================================================
# 读取 result.dat（兼容 x z rhoa / idx x z rhoa）
# ============================================================
def process_result_dat(result_path):
    xs, zs, vs = [], [], []
    with open(result_path, "r", encoding="latin-1", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                nums = [float(p) for p in parts]
            except ValueError:
                continue

            if len(nums) >= 4:
                xi, zi, vi = nums[1], nums[2], nums[3]
            elif len(nums) == 3:
                xi, zi, vi = nums[0], nums[1], nums[2]
            else:
                continue

            xs.append(xi)
            zs.append(zi)
            vs.append(vi)

    if not xs:
        return np.array([]), np.array([]), np.array([])

    return np.asarray(xs), np.asarray(zs), np.asarray(vs)


# ============================================================
# ⭐ Wenner 伪剖面（论文级：可控色标 + 体检）
# ============================================================
def plot_pseudo_section(
        x, z, rho_a, save_path,
        levels=64,
        cmap="jet",
        grid_res_x=400,
        grid_res_z=200,
        display_mode="physical",      # [MOD]
        vlim=None                     # [MOD] (vmin, vmax)
):
    if x.size == 0:
        print(f"[INFO] 无数据，跳过：{save_path}")
        return

    x = np.asarray(x, float)
    z = np.asarray(z, float)
    v = np.asarray(rho_a, float)

    # -----------------------------
    # [MOD] 数据体检
    # -----------------------------
    if not np.isfinite(v).all():
        print(f"[WARN] {save_path} 中存在 NaN/Inf，已自动忽略")
    v = v[np.isfinite(v)]
    x = x[:len(v)]
    z = z[:len(v)]

    if np.allclose(v, v[0]):
        print(f"[WARN] {save_path} 为常数场（ρa={v[0]}），跳过绘图")
        return

    # -----------------------------
    # 1）按 x 排序
    # -----------------------------
    order = np.argsort(x)
    x, z, v = x[order], z[order], v[order]

    # -----------------------------
    # 2）z 翻转（Wenner 几何）
    # -----------------------------
    z = -z

    xmin, xmax = x.min(), x.max()
    zmin, zmax = z.min(), z.max()

    XI, YI = np.meshgrid(
        np.linspace(xmin, xmax, grid_res_x),
        np.linspace(zmin, zmax, grid_res_z)
    )

    ZI = griddata((x, z), v, (XI, YI), method="linear")

    if ZI is None or np.all(np.isnan(ZI)):
        print(f"[INFO] 插值失败：{save_path}")
        return
    # ============================================================
    # 【新增】[MOD] 保存二维伪剖面数值矩阵（用于 DCT / AI）
    # 【插入“二维伪剖面数值导出”】model_00001_pseudosection_wenner.npy
    # ============================================================
    npy_path = save_path.replace(".png", ".npy")
    np.save(npy_path, ZI)

    # -----------------------------
    # [MOD] 色标策略
    # -----------------------------
    if vlim is not None:
        vmin, vmax = vlim
    else:
        vmin, vmax = np.nanmin(ZI), np.nanmax(ZI)

    # -----------------------------
    # 绘图
    # -----------------------------
    plt.figure(figsize=(23, 5))
    cf = plt.contourf(
        XI, YI, ZI,
        levels=levels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    plt.gca().invert_yaxis()
    plt.axis("off")
    plt.gca().set_position([0, 0, 1, 1])

    cbar = plt.colorbar(cf)
    cbar.set_label("Apparent Resistivity (Ω·m)")
    cbar.mappable.set_clim(vmin, vmax)

    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"[OK] Wenner 伪剖面已保存：{save_path}")


# ============================================================
# 批处理
# ============================================================
def batch_plot_pseudosections(parent_dir="BatchModels"):
    model_dirs = sorted(d for d in os.listdir(parent_dir) if d.startswith("model_"))

    for model_name in model_dirs:
        print(f"[INFO] 处理：{model_name}")
        model_path = os.path.join(parent_dir, model_name)
        result_path = os.path.join(model_path, "result.dat")

        if not os.path.exists(result_path):
            print(f"[INFO] 未找到 {result_path}，跳过")
            continue

        x, z, rho = process_result_dat(result_path)

        if x.size == 0:
            print(f"[INFO] 无有效数据：{model_name}")
            continue

        out_path = os.path.join(
            model_path,
            f"{model_name}_pseudosection_wenner.png"
        )

        plot_pseudo_section(
            x, z, rho, out_path,
            display_mode="physical"
        )


if __name__ == "__main__":
    batch_plot_pseudosections("BatchModels")

# #调试
# if __name__ == "__main__":
#     #BatchModels = r"S:\1software\Microsoft VS Code\VSCode_Data\MUL05\EFM5\BatchModels"
#     ROOT = Path(__file__).resolve().parents[1]  # EFM5
#     BatchModels = ROOT / "BatchModels"
#     batch_plot_pseudosections(BatchModels)