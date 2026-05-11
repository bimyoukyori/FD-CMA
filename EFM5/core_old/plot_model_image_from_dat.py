#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_model_image_from_dat.py
正确显示 Console2 Model.dat 的真实网格（Triangulation）
⭐ 永久取消 imshow（规则网格）——避免伪影
⭐ 使用 tripcolor + gouraud = 标准有限元网格显示方式
⭐ 色标使用真实 min~max ρ（避免异常体被压缩）
"""

import os
import re
import sys
import math
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


_num_re = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

def read_text_auto(path: str) -> str:
    encs = ["utf-8","gbk","cp936","latin-1"]
    data = pathlib.Path(path).read_bytes()
    for enc in encs:
        try:
            return data.decode(enc)
        except Exception:
            pass
    return data.decode("latin-1", errors="ignore")


# ------------------------------------------------------------
# 精确解析 Model.dat → (x, z, rho)
# ------------------------------------------------------------
def parse_xzr_from_model_dat(text: str):
    xs, zs, rhos = [], [], []

    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("*"):
            continue

        nums = _num_re.findall(s)
        if len(nums) >= 4:
            try:
                xs.append(float(nums[-3]))
                zs.append(float(nums[-2]))
                rhos.append(float(nums[-1]))
            except:
                continue

    # 兼容 3 列 x z rho
    if not xs:
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("*"):
                continue
            nums = _num_re.findall(s)
            if len(nums) == 3:
                xs.append(float(nums[0]))
                zs.append(float(nums[1]))
                rhos.append(float(nums[2]))

    if not xs:
        raise ValueError("❌ Model.dat 格式异常：未找到 x z rho")

    return np.asarray(xs,float), np.asarray(zs,float), np.asarray(rhos,float)


# ------------------------------------------------------------
# ⭐ Console2 正确模型显示方式：Triangulation
# ------------------------------------------------------------
def plot_triangulation_correct(x, z, rho, outdir):

    # ----- 色标范围：真实物理 ρ 范围 -----
    rho_clean = rho[np.isfinite(rho) & (rho > 0)]
    vmin = float(rho_clean.min())
    vmax = float(rho_clean.max())

    # ----- 建三角剖分 -----
    triang = mtri.Triangulation(x, z)

    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    png = outdir / "rho_model_triangulated.png"
    svg = outdir / "rho_model_triangulated.svg"

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)

    # ⭐ gouraud = 真实 FEM 网格填色（不会伪影）
    tpc = ax.tripcolor(
        triang, rho,
        shading='gouraud',
        vmin=vmin, vmax=vmax,
        cmap="viridis"
    )

    cbar = fig.colorbar(tpc, ax=ax)
    cbar.set_label("ρ (Ω·m)")

    ax.set_title("ρ model (finite-element triangulated)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.invert_yaxis()

    # 可选：加散点辅助显示网格
    ax.scatter(x, z, s=6, color="white", alpha=0.3)

    fig.savefig(png, dpi=220)
    fig.savefig(svg)
    plt.close()
    print(f"[OK] Triangulated 模型图已输出：{png}")


# ------------------------------------------------------------
# 主函数
# ------------------------------------------------------------
def main():
    inpath = sys.argv[1] if len(sys.argv)>=2 else DEFAULT_INPUT
    outdir = sys.argv[2] if len(sys.argv)>=3 else DEFAULT_OUTDIR

    if not os.path.isfile(inpath):
        raise FileNotFoundError(inpath)

    text = read_text_auto(inpath)
    x, z, rho = parse_xzr_from_model_dat(text)

    # ⭐ 永远使用 Triangulation（Console2 是三角网！）
    plot_triangulation_correct(x, z, rho, outdir)


if __name__ == "__main__":
    main()
