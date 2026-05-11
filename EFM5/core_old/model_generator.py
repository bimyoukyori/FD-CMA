#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_generator.py (V2.2 dual-mode)
-----------------------------------
支持两种 ERT 建模模式：

mode = "id"
    → 编号制（Console2.exe）
    → 节点最后一列 = 材料编号（1,2,3,...）
    → 材料表：编号 -> 电阻率

mode = "real"
    → 真实值制（Console3.exe）
    → 节点最后一列 = 电阻率真实值（Ω·m）
    → 材料表仍然写出（方便后期调试 / 论文记录）

说明：
- 不插入电极节点
- 规则网格
- 四类异常：normal / cavity / crack / loose
- crack 支持 angle_deg
"""

import os
import json
import numpy as np


# ============================================================
# 读取 anomaly_def.json
# ============================================================
def load_anomaly_def(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 电极与 Wenner 装置
# ============================================================
def generate_electrodes(Pnum, spacing):
    half = Pnum // 2
    xs = np.linspace(-half * spacing, half * spacing, Pnum)
    return [(i + 1, x, 0.0) for i, x in enumerate(xs)]


def generate_devices_full(Pnum):
    devices = []
    count = 0
    for A in range(1, Pnum - 2):
        for spacing in range(1, (Pnum - A) // 3 + 1):
            B = A + 3 * spacing
            M = A + spacing
            N = A + 2 * spacing
            if B <= Pnum:
                count += 1
                devices.append((count, A, B, M, N))
    return devices


# ============================================================
# 构造材料表（编号 -> rho）
# ============================================================
def build_rho_table(anom_def):
    rho_bg = float(anom_def["ert"]["rho_background"])
    anomalies = anom_def["ert"].get("anomalies", [])

    rho_list = [rho_bg]  # id=1 背景
    seen = {rho_bg}

    for anom in anomalies:
        if anom.get("type", "").lower() == "normal":
            continue
        rho_a = float(anom["rho"])
        if rho_a not in seen:
            rho_list.append(rho_a)
            seen.add(rho_a)

    rho2id = {rho: i + 1 for i, rho in enumerate(rho_list)}
    return rho_list, rho2id


# ============================================================
# 规则网格 + 异常体建模
# ============================================================
def generate_nodes_from_json(anom_def, rho2id, mode="id"):
    """
    [MOD] 新增 mode 参数
    mode = "id"   → Console2（编号制）
    mode = "real" → Console3（真实值制）
    """
    dom = anom_def["domain"]
    dx, dz = dom["dx"], dom["dz"]

    x = np.arange(dom["x_min"], dom["x_max"] + dx, dx)
    z = np.arange(dom["z_min"], dom["z_max"] + dz, dz)

    rho_bg = float(anom_def["ert"]["rho_background"])
    anomalies = anom_def["ert"].get("anomalies", [])

    nodes = []
    idx = 1

    for zi in z:
        for xi in x:
            rho_val = rho_bg

            for anom in anomalies:
                t = anom["type"].lower()
                g = anom["geometry"]
                rho_a = float(anom["rho"])
                x0, z0, r = float(g["x0"]), float(g["z0"]), float(g["r"])

                if t == "cavity":
                    if (xi - x0) ** 2 + (zi - z0) ** 2 <= r ** 2:
                        rho_val = rho_a

                elif t == "crack":
                    angle = float(g.get("angle_deg", 0.0))
                    theta = np.deg2rad(angle)
                    xr = (xi - x0) * np.cos(theta) + (zi - z0) * np.sin(theta)
                    zr = -(xi - x0) * np.sin(theta) + (zi - z0) * np.cos(theta)
                    if abs(xr) <= r and abs(zr) <= 4 * r:
                        rho_val = rho_a

                elif t == "loose":
                    if (xi - x0) ** 2 + (zi - z0) ** 2 <= r ** 2:
                        rho_val = rho_a

                elif t == "normal":
                    pass

            # ==============================
            # [MOD] 关键：按 mode 决定输出
            # ==============================
            if mode == "id":
                out_val = rho2id.get(float(rho_val))
                if out_val is None:
                    raise ValueError(f"[ERR] rho={rho_val} 未在材料表中")
            else:
                out_val = float(rho_val)

            nodes.append((idx, xi, zi, out_val))
            idx += 1

    return nodes


# ============================================================
# 写 cmd.par（不区分模式）
# ============================================================
def write_cmd_par(path, dim, array_type, survey_type,
                  Pnum, Dnum, numnode, numR):
    with open(os.path.join(path, "cmd.par"), "w", encoding="utf-8") as f:
        f.write("---计算模型相关总数信息----------------------\n")
        f.write("*维数\n")
        f.write(f"{dim}\n")
        f.write("*观测装置类型\n")
        f.write(f"{array_type} {survey_type}\n")
        f.write("*模拟用到的总电极数\n")
        f.write(f"{Pnum}\n")
        f.write("*模拟总数\n")
        f.write(f"{Dnum}\n")
        f.write("*全局节点总数\n")
        f.write(f"{numnode}\n")
        f.write("*计算域中不同电阻率个数\n")
        f.write(f"{numR}\n")
        f.write("*地形点数（当地形水平时为0）\n")
        f.write("0\n")


# ============================================================
# 写 Model.dat（双模式）
# ============================================================
def write_model_dat(path, electrodes, devices, nodes, dom, rho_list, mode="id"):
    with open(os.path.join(path, "Model.dat"), "w", encoding="utf-8") as f:
        f.write("*全局(矩形)的范围\n")
        f.write(f"{dom['x_min']:.2f} {dom['x_max']:.2f} "
                f"{dom['z_min']:.2f} {dom['z_max']:.2f}\n")

        f.write("*电极信息\n")
        f.write(f"{len(electrodes)}\n")
        for eid, x, z in electrodes:
            f.write(f"{eid} {x:.2f} {z:.2f}\n")
        f.write("0 0 0\n")

        f.write("*对称四极测深模拟信息\n")
        f.write(f"{len(devices)}\n")
        for d in devices:
            f.write(" ".join(map(str, d)) + "\n")
        f.write("0 0 0\n")

        f.write("*全局节点信息\n")
        f.write(f"{len(nodes)}\n")
        for nid, x, z, v in nodes:
            if mode == "id":
                f.write(f"{nid} {x:.2f} {z:.2f} {int(v)}\n")
            else:
                f.write(f"{nid} {x:.2f} {z:.2f} {float(v):.6f}\n")
        f.write("0 0 0\n")

        # [MOD] 两种模式都写材料表（你要求）
        f.write("*计算域中不同电阻率信息\n")
        f.write(f"{len(rho_list)}\n")
        for i, rho in enumerate(rho_list, 1):
            f.write(f"{i} {float(rho):.2f}\n")
        f.write("0 0 0\n")


# ============================================================
# 对外入口【模式切换】【编号版】/【真实值版】
# ============================================================
def generate_model_from_json(model_dir, json_path,
                             Pnum=91, spacing=2.0, mode="id"): # Console2【编号版】
                             #Pnum=91, spacing=2.0, mode="real"): # Console3【真实值版】
    
    
    anom_def = load_anomaly_def(json_path)
    dom = anom_def["domain"]

    electrodes = generate_electrodes(Pnum, spacing)
    devices = generate_devices_full(Pnum)

    rho_list, rho2id = build_rho_table(anom_def)
    nodes = generate_nodes_from_json(anom_def, rho2id, mode=mode)

    os.makedirs(model_dir, exist_ok=True)

# ArrayType = 4 → 四极装置

# SurveyType = 1 → 对称四极装置（AB/4）

# SurveyType = 2 → 偶极/非对称类装置（AM/4 等）

    write_cmd_par(
        model_dir,
        dim=2,
        array_type=4,
        survey_type=1, #对称四极装置（AB/4）
        Pnum=Pnum,
        Dnum=len(devices),
        numnode=len(nodes),
        numR=len(rho_list)
    )

    write_model_dat(
        model_dir,
        electrodes,
        devices,
        nodes,
        dom,
        rho_list,
        mode=mode
    )

    print(f"[OK] Model generated ({mode}) → {model_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        generate_model_from_json(
            sys.argv[1],
            sys.argv[2],
            Pnum=91,
            spacing=2.0,
            mode="id"   # ← 本地测试默认编号制
        )
    else:
        print("Usage: python model_generator.py <model_dir> <anomaly_def.json>")
