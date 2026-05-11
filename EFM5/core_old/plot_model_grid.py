#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# 读文本：兼容多编码（utf-8 / gbk / gb2312 / latin-1）
# =========================================================
def _read_text_multi_encoding(path):
    with open(path, "rb") as f:
        raw = f.read()

    encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"]
    for enc in encodings:
        try:
            text = raw.decode(enc)
            print(f"[INFO] Model.dat 编码尝试成功：{enc}")
            return text
        except Exception:
            pass

    print("[WARN] 所有编码尝试失败，使用 latin-1(ignore) 兜底")
    return raw.decode("latin-1", errors="ignore")


# =========================================================
# 读取 Model.dat（节点段 + 可选“编号→真实ρ”映射）
# =========================================================
def load_model_nodes_with_mapping(model_path):
    """
    从 Model.dat 中读取：
      - 节点段： idx, x, z, rho_or_id
      - 若文件末尾存在“不同电阻率信息”映射表：
            id -> rho(Ω·m)
        则自动将 rho_id 映射为真实电阻率
      - 若不存在映射表：
        则认为节点段最后一列已是真实 rho(Ω·m)

    返回：
        x (np.ndarray)
        z (np.ndarray)
        rho (np.ndarray)   # 始终是真实 ρ(Ω·m)
    """
    text = _read_text_multi_encoding(model_path)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    def _is_section_title(ln):
        return ln.startswith("*")

    def _parse_int(ln):
        return int(ln.strip())

    def _is_terminator(ln):
        p = ln.split()
        return len(p) >= 3 and p[0] == "0" and p[1] == "0" and p[2] == "0"

    i = 0

    # -----------------------------------------------------
    # 1) 跳过范围段：标题 + 一行范围
    # -----------------------------------------------------
    while i < len(lines) and not _is_section_title(lines[i]):
        i += 1
    if i >= len(lines):
        raise ValueError("未找到任何段标题")
    i += 1  # 跳过范围标题
    if i < len(lines):
        i += 1  # 跳过范围数值行

    # -----------------------------------------------------
    # 2) 电极段
    # -----------------------------------------------------
    while i < len(lines) and not _is_section_title(lines[i]):
        i += 1
    if i >= len(lines):
        raise ValueError("未找到电极段标题")
    i += 1

    n_elec = _parse_int(lines[i])
    i += 1
    i += n_elec
    if i >= len(lines) or not _is_terminator(lines[i]):
        raise ValueError("电极段未找到 0 0 0 结束符")
    i += 1

    # -----------------------------------------------------
    # 3) 装置段
    # -----------------------------------------------------
    while i < len(lines) and not _is_section_title(lines[i]):
        i += 1
    if i >= len(lines):
        raise ValueError("未找到装置段标题")
    i += 1

    n_dev = _parse_int(lines[i])
    i += 1
    i += n_dev
    if i >= len(lines) or not _is_terminator(lines[i]):
        raise ValueError("装置段未找到 0 0 0 结束符")
    i += 1

    # -----------------------------------------------------
    # 4) 节点段
    # -----------------------------------------------------
    while i < len(lines) and not _is_section_title(lines[i]):
        i += 1
    if i >= len(lines):
        raise ValueError("未找到节点段标题")
    i += 1

    n_nodes = _parse_int(lines[i])
    i += 1

    xs, zs, rho_raw = [], [], []
    read_cnt = 0
    while i < len(lines) and read_cnt < n_nodes:
        parts = lines[i].split()
        if len(parts) >= 4:
            try:
                _, x, z, r = map(float, parts[:4])
                xs.append(x)
                zs.append(z)
                rho_raw.append(r)
                read_cnt += 1
            except Exception:
                pass
        i += 1

    if read_cnt != n_nodes:
        raise ValueError(f"节点段读取数量不足：期望 {n_nodes}，实际 {read_cnt}")

    if i < len(lines) and _is_terminator(lines[i]):
        i += 1

    # -----------------------------------------------------
    # 5) 尝试解析“编号 → 真实ρ”映射表（若存在）
    # -----------------------------------------------------
    rho_map = {}
    j = i
    while j < len(lines):
        parts = lines[j].split()
        # 寻找一个“类型数”（单个整数）
        if len(parts) == 1 and parts[0].isdigit():
            n_types = int(parts[0])
            tmp = {}
            ok = True
            for k in range(1, n_types + 1):
                if j + k >= len(lines):
                    ok = False
                    break
                p = lines[j + k].split()
                if len(p) < 2:
                    ok = False
                    break
                try:
                    rid = int(float(p[0]))
                    rval = float(p[1])
                    tmp[rid] = rval
                except Exception:
                    ok = False
                    break

            end_line = j + n_types + 1
            if ok and end_line < len(lines) and _is_terminator(lines[end_line]):
                rho_map = tmp
                print(f"[INFO] 发现电阻率映射表：{rho_map}")
            break
        j += 1

    x = np.array(xs, dtype=float)
    z = np.array(zs, dtype=float)
    rho_raw = np.array(rho_raw, dtype=float)

    # -----------------------------------------------------
    # 6) 若存在映射表：rho_id → 真实ρ
    # -----------------------------------------------------
    if rho_map:
        rid = np.rint(rho_raw).astype(int)
        rho = np.array([rho_map.get(v, float(v)) for v in rid], dtype=float)
    else:
        rho = rho_raw

    return x, z, rho


# =========================================================
# ⭐ 最终版：严格规则网格绘图（真实ρ，Ω·m）
# =========================================================
def plot_model_grid(model_dir):
    model_path = os.path.join(model_dir, "Model.dat")
    if not os.path.exists(model_path):
        print(f"[plot_model_grid] 未找到 Model.dat：{model_path}")
        return

    try:
        x, z, rho = load_model_nodes_with_mapping(model_path)
    except Exception as e:
        print(f"[plot_model_grid] Model.dat 解析失败：{e}")
        return

    if rho.size == 0:
        print("[plot_model_grid] 未读取到任何节点")
        return

    # 规则网格检查
    x2 = np.round(x, 6)
    z2 = np.round(z, 6)
    x_unique = np.unique(x2)
    z_unique = np.unique(z2)
    nx = len(x_unique)
    nz = len(z_unique)

    print(f"[DEBUG] nx={nx}, nz={nz}, nx*nz={nx*nz}, n_nodes={rho.size}")

    if nx * nz != rho.size:
        print("[ERR] 节点数与规则网格不一致，拒绝绘图")
        return

    # 排序并 reshape
    order = np.lexsort((x2, z2))
    RHO = rho[order].reshape(nz, nx)

    save_path = os.path.join(
        model_dir,
        os.path.basename(model_dir) + "_model_grid.png"
    )

    plt.figure(figsize=(14, 6))
    plt.imshow(
        RHO,
        origin="lower",
        cmap="viridis",
        extent=[x_unique.min(), x_unique.max(),
                z_unique.min(), z_unique.max()],
        aspect="auto"
    )
    plt.colorbar(label="ρ (Ω·m)")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title("ρ model (FEM regular grid)")
    plt.gca().invert_yaxis()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[plot_model_grid] 已保存模型网格图：{save_path}")


# =========================================================
# 批处理入口（可选）
# =========================================================
if __name__ == "__main__":
    parent = "BatchModels"
    for d in sorted(os.listdir(parent)):
        if d.startswith("model_"):
            plot_model_grid(os.path.join(parent, d))
