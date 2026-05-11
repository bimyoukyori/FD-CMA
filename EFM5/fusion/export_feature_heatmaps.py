#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.fftpack import idct

from fusion_net import FusionNet
from train_fusion import (
    FusionPairDataset,
    PAIRS_CSV,
    GPR_DCT_HW,
)

# ============================================================
# 配置
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "fusion" / "runs"

BASELINE_RUN = os.getenv("FUSION_BASELINE_RUN", "exp_fusion_baseline")
MAIN_RUN = os.getenv("FUSION_MAIN_RUN", "exp_fusion_main")

BASELINE_CKPT = RUNS_DIR / BASELINE_RUN / "fusion_best.pth"
MAIN_CKPT = RUNS_DIR / MAIN_RUN / "fusion_best.pth"

OUT_DIR = RUNS_DIR / "paired_heatmaps_paper"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GPR_RAW_DIR = ROOT / "data" / "gpr" / "train"

# 每类导出几个样本
MAX_PER_CLASS = 4


# ============================================================
# 工具函数
# ============================================================

def normalize(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    mn, mx = float(a.min()), float(a.max())
    if mx - mn < 1e-8:
        return np.zeros_like(a, dtype=np.float32)
    return (a - mn) / (mx - mn)


def idct2(a: np.ndarray) -> np.ndarray:
    return idct(idct(a.T, norm="ortho").T, norm="ortho")


def resize_to_shape(arr: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    t = torch.tensor(arr, dtype=torch.float32).view(1, 1, arr.shape[0], arr.shape[1])
    t = F.interpolate(t, size=shape_hw, mode="bilinear", align_corners=False)
    return t[0, 0].cpu().numpy()


def load_raw_bscan(sample_id: str, label_name: str) -> Optional[np.ndarray]:
    p1 = GPR_RAW_DIR / label_name / f"{sample_id}.csv"
    if p1.exists():
        return np.loadtxt(p1, delimiter=",", dtype=np.float32)

    cands = list(GPR_RAW_DIR.rglob(f"{sample_id}.csv"))
    if cands:
        return np.loadtxt(cands[0], delimiter=",", dtype=np.float32)

    return None


def parse_confidence(row: Dict[str, str]) -> float:
    try:
        return float(row.get("confidence", 1.0))
    except Exception:
        return 1.0


# ============================================================
# 模型加载
# ============================================================

def build_model(dataset: FusionPairDataset, ckpt_path: Path) -> FusionNet:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    model = FusionNet(
        ert_dim=dataset.ert_dim,
        num_classes=len(dataset.class_names),
        emb_dim=256,
    ).to(DEVICE)

    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ============================================================
# Grad-CAM hook
# ============================================================

class FeatureHook:
    def __init__(self, module: torch.nn.Module):
        self.features = None
        self.grads = None
        self.h1 = module.register_forward_hook(self.forward_hook)
        self.h2 = module.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, inp, out):
        self.features = out

    def backward_hook(self, module, grad_input, grad_output):
        self.grads = grad_output[0]

    def close(self):
        self.h1.remove()
        self.h2.remove()


def compute_gradcam_2d(model: FusionNet, ert_feat: torch.Tensor, gpr_feat: torch.Tensor) -> Tuple[np.ndarray, float]:
    """
    对 GPR 分支最后卷积块 b3 做 2D Grad-CAM
    解释目标：正样本对相似度 sim(e1, e2)
    """
    hook = FeatureHook(model.gpr_branch.b3)

    x1 = ert_feat.to(DEVICE).unsqueeze(0)       # (1, D_ert)
    x2 = gpr_feat.to(DEVICE).unsqueeze(0)       # (1, 1, H, W)

    model.zero_grad(set_to_none=True)
    _, _, e1, e2 = model(x1, x2)
    score = F.cosine_similarity(e1, e2).mean()
    score.backward()

    feat = hook.features[0]   # (C, h, w)
    grad = hook.grads[0]      # (C, h, w)

    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * feat).sum(dim=0))
    cam = cam / (cam.max() + 1e-8)

    cam = cam.detach().cpu().numpy().astype(np.float32)
    sim_val = float(score.detach().cpu().item())

    hook.close()
    return cam, sim_val


# ============================================================
# 样本选择
# ============================================================

def select_rows(dataset: FusionPairDataset) -> List[Tuple[int, Dict[str, str]]]:
    cls_map = defaultdict(list)
    for i, row in enumerate(dataset.rows):
        sid = (row.get("id") or "").strip()
        label_name = (row.get("label") or "unknown").strip() or "unknown"
        raw_bscan = load_raw_bscan(sid, label_name)
        if raw_bscan is None:
            continue
        cls_map[label_name].append((i, row))

    picked = []
    for cls, items in cls_map.items():
        items = sorted(items, key=lambda x: parse_confidence(x[1]), reverse=True)
        picked.extend(items[:MAX_PER_CLASS])
    return picked


def suppress_ultra_lowfreq(cam: np.ndarray, k: int = 3) -> np.ndarray:
    cam = cam.copy()
    cam[:k, :k] = 0.0     # 去掉左上角超低频块
    cam[0, :] = 0.0       # 去掉 DC / 最低水平频率
    cam[:, 0] = 0.0       # 去掉最低垂向频率
    return cam

def suppress_direct_wave(heat: np.ndarray, top_rows: int = 6) -> np.ndarray:
    heat = heat.copy()
    heat[:top_rows, :] = 0.0
    return heat

# ============================================================
# 绘图
# ============================================================

def save_paired_panel(
    raw_bscan: np.ndarray,
    base_dct: np.ndarray,
    main_dct: np.ndarray,
    base_bscan: np.ndarray,
    main_bscan: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    raw = normalize(raw_bscan)
    base_d = normalize(base_dct)
    main_d = normalize(main_dct)
    base_b = normalize(base_bscan)
    main_b = normalize(main_bscan)
    diff_b = normalize(np.maximum(main_b - base_b, 0.0))

    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.2), dpi=220)

    axes[0, 0].imshow(raw, cmap="gray", aspect="auto", origin="upper")
    axes[0, 0].set_title("Raw GPR B-scan", fontsize=9)

    axes[0, 1].imshow(base_d, cmap="jet", aspect="auto", origin="upper")
    axes[0, 1].set_title("Baseline DCT heatmap", fontsize=9)

    axes[0, 2].imshow(main_d, cmap="jet", aspect="auto", origin="upper")
    axes[0, 2].set_title("Prior-constrained DCT heatmap", fontsize=9)

    axes[0, 3].imshow(normalize(np.maximum(main_d - base_d, 0.0)), cmap="jet", aspect="auto", origin="upper")
    axes[0, 3].set_title("DCT heatmap difference", fontsize=9)

    axes[1, 0].imshow(raw, cmap="gray", aspect="auto", origin="upper")
    axes[1, 0].imshow(base_b, cmap="jet", alpha=0.45, aspect="auto", origin="upper")
    axes[1, 0].set_title("Baseline overlay", fontsize=9)

    axes[1, 1].imshow(raw, cmap="gray", aspect="auto", origin="upper")
    axes[1, 1].imshow(main_b, cmap="jet", alpha=0.45, aspect="auto", origin="upper")
    axes[1, 1].set_title("Prior-constrained overlay", fontsize=9)

    axes[1, 2].imshow(diff_b, cmap="jet", aspect="auto", origin="upper")
    axes[1, 2].set_title("Back-projected difference", fontsize=9)

    axes[1, 3].axis("off")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ============================================================
# 主流程
# ============================================================

def main():
    if not BASELINE_CKPT.exists():
        raise FileNotFoundError(f"baseline checkpoint not found: {BASELINE_CKPT}")
    if not MAIN_CKPT.exists():
        raise FileNotFoundError(f"main checkpoint not found: {MAIN_CKPT}")

    dataset = FusionPairDataset(PAIRS_CSV)

    model_baseline = build_model(dataset, BASELINE_CKPT)
    model_main = build_model(dataset, MAIN_CKPT)

    selected = select_rows(dataset)
    print(f"[INFO] selected samples for paired export: {len(selected)}")

    summary_rows = []

    for k, (idx, row) in enumerate(selected, start=1):
        sid = (row.get("id") or "").strip()
        label_name = (row.get("label") or "unknown").strip() or "unknown"
        conf = parse_confidence(row)

        ert_feat, gpr_feat, _, _ = dataset[idx]
        raw_bscan = load_raw_bscan(sid, label_name)
        if raw_bscan is None:
            print(f"[WARN] raw B-scan not found for sample id={sid}, skip")
            continue

        cam_base_small, sim_base = compute_gradcam_2d(model_baseline, ert_feat, gpr_feat)
        cam_main_small, sim_main = compute_gradcam_2d(model_main, ert_feat, gpr_feat)

        cam_base_dct = resize_to_shape(cam_base_small, (GPR_DCT_HW, GPR_DCT_HW))
        cam_main_dct = resize_to_shape(cam_main_small, (GPR_DCT_HW, GPR_DCT_HW))

        cam_base_dct_bp = suppress_ultra_lowfreq(cam_base_dct, k=3)
        cam_main_dct_bp = suppress_ultra_lowfreq(cam_main_dct, k=3)

        cam_base_bscan = resize_to_shape(np.abs(idct2(cam_base_dct_bp)), raw_bscan.shape)
        cam_main_bscan = resize_to_shape(np.abs(idct2(cam_main_dct_bp)), raw_bscan.shape)

        cam_base_bscan = suppress_direct_wave(cam_base_bscan, top_rows=6)
        cam_main_bscan = suppress_direct_wave(cam_main_bscan, top_rows=6)

        title = (
            f"{label_name} / {sid} | confidence={conf:.3f} | "
            f"sim_base={sim_base:.3f}, sim_main={sim_main:.3f}"
        )

        base = f"{k:02d}_{label_name}_{sid}"
        save_paired_panel(
            raw_bscan,
            cam_base_dct,
            cam_main_dct,
            cam_base_bscan,
            cam_main_bscan,
            OUT_DIR / f"{base}_paired_panel.png",
            title=title,
        )

        summary_rows.append({
            "idx": k,
            "id": sid,
            "label": label_name,
            "confidence": conf,
            "sim_baseline": sim_base,
            "sim_main": sim_main,
            "delta_sim": sim_main - sim_base,
            "heat_mean_baseline": float(normalize(cam_base_bscan).mean()),
            "heat_mean_main": float(normalize(cam_main_bscan).mean()),
        })

        print(
            f"[OK] id={sid}, label={label_name}, "
            f"sim_baseline={sim_base:.4f}, sim_main={sim_main:.4f}"
        )

    out_csv = OUT_DIR / "summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx", "id", "label", "confidence",
                "sim_baseline", "sim_main", "delta_sim",
                "heat_mean_baseline", "heat_mean_main",
            ]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[DONE] paired heatmaps saved to: {OUT_DIR}")
    print(f"[DONE] summary csv: {out_csv}")


if __name__ == "__main__":
    main()