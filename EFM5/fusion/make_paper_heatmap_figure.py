#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
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
# Config
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "fusion" / "runs"
GPR_RAW_DIR = ROOT / "data" / "gpr" / "train"

BASELINE_RUN = os.getenv("FUSION_BASELINE_RUN", "exp_fusion_baseline")
MAIN_RUN = os.getenv("FUSION_MAIN_RUN", "exp_fusion_main")

BASELINE_CKPT = RUNS_DIR / BASELINE_RUN / "fusion_best.pth"
MAIN_CKPT = RUNS_DIR / MAIN_RUN / "fusion_best.pth"

OUT_DIR = RUNS_DIR / "paper_heatmap_single_column"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SPECS = [
    ("cavity", "00001"),
    ("crack", "10001"),
]

# Figure style: 单栏友好
FIG_W = 7.2
FIG_H = 2.2
DPI = 300

FONT = 8
TITLE = 8
SUPTITLE = 9

CMAP_HEAT = "viridis"
CMAP_DIFF = "OrRd"

LOWFREQ_SUPPRESS_K = 3
TOP_ROWS_SUPPRESS = 6

# 单栏正文建议默认不加 colorbar
ADD_COLORBAR = False


# ============================================================
# Font
# ============================================================

def pick_serif_font() -> str:
    candidates = [
        "Times New Roman",
        "Nimbus Roman",
        "Liberation Serif",
        "STIXGeneral",
        "DejaVu Serif",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "DejaVu Serif"


SERIF_FONT = pick_serif_font()


# ============================================================
# Utils
# ============================================================

def normalize(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    mn, mx = float(a.min()), float(a.max())
    if mx - mn < 1e-8:
        return np.zeros_like(a, dtype=np.float32)
    return (a - mn) / (mx - mn)


def robust_positive_rescale(a: np.ndarray, q: float = 99.0) -> np.ndarray:
    """
    用正值分位数做显示缩放，避免 difference 图大面积发黑。
    """
    a = np.asarray(a, dtype=np.float32)
    a = np.maximum(a, 0.0)
    pos = a[a > 0]
    if pos.size == 0:
        return np.zeros_like(a, dtype=np.float32)
    vmax = np.percentile(pos, q)
    vmax = max(float(vmax), 1e-8)
    return np.clip(a / vmax, 0.0, 1.0)


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


def suppress_ultra_lowfreq(cam: np.ndarray, k: int = 3) -> np.ndarray:
    cam = cam.copy()
    cam[:k, :k] = 0.0
    cam[0, :] = 0.0
    cam[:, 0] = 0.0
    return cam


def suppress_direct_wave(heat: np.ndarray, top_rows: int = 6) -> np.ndarray:
    heat = heat.copy()
    heat[:top_rows, :] = 0.0
    return heat


# ============================================================
# Model
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
    hook = FeatureHook(model.gpr_branch.b3)

    x1 = ert_feat.to(DEVICE).unsqueeze(0)
    x2 = gpr_feat.to(DEVICE).unsqueeze(0)

    model.zero_grad(set_to_none=True)
    _, _, e1, e2 = model(x1, x2)
    score = F.cosine_similarity(e1, e2).mean()
    score.backward()

    feat = hook.features[0]
    grad = hook.grads[0]

    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * feat).sum(dim=0))
    cam = cam / (cam.max() + 1e-8)

    cam = cam.detach().cpu().numpy().astype(np.float32)
    sim_val = float(score.detach().cpu().item())

    hook.close()
    return cam, sim_val


# ============================================================
# Data helpers
# ============================================================

def find_row_index(dataset: FusionPairDataset, label_name: str, sample_id: str) -> int:
    for i, row in enumerate(dataset.rows):
        sid = (row.get("id") or "").strip()
        lab = (row.get("label") or "unknown").strip() or "unknown"
        if sid == sample_id and lab == label_name:
            return i
    raise ValueError(f"sample not found in pairs.csv: label={label_name}, id={sample_id}")


def prepare_sample_outputs(
    dataset: FusionPairDataset,
    model_baseline: FusionNet,
    model_main: FusionNet,
    label_name: str,
    sample_id: str,
) -> Dict[str, np.ndarray]:
    idx = find_row_index(dataset, label_name, sample_id)
    ert_feat, gpr_feat, _, _ = dataset[idx]

    raw_bscan = load_raw_bscan(sample_id, label_name)
    if raw_bscan is None:
        raise FileNotFoundError(f"raw B-scan not found for label={label_name}, id={sample_id}")

    cam_base_small, sim_base = compute_gradcam_2d(model_baseline, ert_feat, gpr_feat)
    cam_main_small, sim_main = compute_gradcam_2d(model_main, ert_feat, gpr_feat)

    cam_base_dct = resize_to_shape(cam_base_small, (GPR_DCT_HW, GPR_DCT_HW))
    cam_main_dct = resize_to_shape(cam_main_small, (GPR_DCT_HW, GPR_DCT_HW))

    cam_base_dct_bp = suppress_ultra_lowfreq(cam_base_dct, k=LOWFREQ_SUPPRESS_K)
    cam_main_dct_bp = suppress_ultra_lowfreq(cam_main_dct, k=LOWFREQ_SUPPRESS_K)

    cam_base_bscan = resize_to_shape(np.abs(idct2(cam_base_dct_bp)), raw_bscan.shape)
    cam_main_bscan = resize_to_shape(np.abs(idct2(cam_main_dct_bp)), raw_bscan.shape)

    cam_base_bscan = suppress_direct_wave(cam_base_bscan, top_rows=TOP_ROWS_SUPPRESS)
    cam_main_bscan = suppress_direct_wave(cam_main_bscan, top_rows=TOP_ROWS_SUPPRESS)

    dct_diff = np.maximum(normalize(cam_main_dct) - normalize(cam_base_dct), 0.0)
    bscan_diff = np.maximum(normalize(cam_main_bscan) - normalize(cam_base_bscan), 0.0)

    return {
        "raw": normalize(raw_bscan),
        "base_dct": normalize(cam_base_dct),
        "main_dct": normalize(cam_main_dct),
        "dct_diff": robust_positive_rescale(dct_diff),
        "bscan_diff": robust_positive_rescale(bscan_diff),  # 这里只算，不放正文
        "sim_base": sim_base,
        "sim_main": sim_main,
        "delta_sim": sim_main - sim_base,
    }


# ============================================================
# Plot helpers
# ============================================================

def style_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.8)


def make_single_figure(
    label_name: str,
    sample_id: str,
    data: Dict[str, np.ndarray],
) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [SERIF_FONT],
        "mathtext.fontset": "stix",
        "font.size": FONT,
        "axes.titlesize": TITLE,
        "axes.labelsize": FONT,
        "xtick.labelsize": FONT - 1,
        "ytick.labelsize": FONT - 1,
    })

    if ADD_COLORBAR:
        fig, axes = plt.subplots(
            1, 5,
            figsize=(FIG_W + 0.7, FIG_H),
            dpi=DPI,
            gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.045]}
        )
        ax0, ax1, ax2, ax3, cax = axes
    else:
        fig, axes = plt.subplots(1, 4, figsize=(FIG_W, FIG_H), dpi=DPI)
        ax0, ax1, ax2, ax3 = axes
        cax = None

    # 主图
    im0 = ax0.imshow(data["raw"], cmap="gray", aspect="auto", origin="upper", vmin=0, vmax=1)
    im1 = ax1.imshow(data["base_dct"], cmap=CMAP_HEAT, aspect="auto", origin="upper", vmin=0, vmax=1)
    im2 = ax2.imshow(data["main_dct"], cmap=CMAP_HEAT, aspect="auto", origin="upper", vmin=0, vmax=1)
    im3 = ax3.imshow(data["dct_diff"], cmap=CMAP_DIFF, aspect="auto", origin="upper", vmin=0.02, vmax=0.95)

    titles = [
        "Raw B-scan",
        "Baseline DCT",
        "Prior-constrained DCT",
        "DCT difference",
    ]
    for ax, t in zip([ax0, ax1, ax2, ax3], titles):
        ax.set_title(t, pad=6)
        style_ax(ax)

    # 单样本标题：放在整张图上方，不压子图
    fig.suptitle(
        f"{label_name.capitalize()} sample (ID: {sample_id}, $\\Delta$sim = {data['delta_sim']:+.3f})",
        y=0.98,
        fontsize=SUPTITLE
    )

    if ADD_COLORBAR and cax is not None:
        cb = fig.colorbar(im2, cax=cax)
        cb.set_label("Normalized response", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    fig.subplots_adjust(left=0.03, right=0.98, top=0.82, bottom=0.08, wspace=0.05)

    out_base = OUT_DIR / f"{label_name}_{sample_id}_single_column"
    fig.savefig(out_base.with_suffix(".png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[DONE] saved: {out_base.with_suffix('.png')}")
    print(f"[DONE] saved: {out_base.with_suffix('.pdf')}")


# ============================================================
# Main
# ============================================================

def main():
    dataset = FusionPairDataset(PAIRS_CSV)
    model_baseline = build_model(dataset, BASELINE_CKPT)
    model_main = build_model(dataset, MAIN_CKPT)

    for label_name, sample_id in SAMPLE_SPECS:
        data = prepare_sample_outputs(
            dataset=dataset,
            model_baseline=model_baseline,
            model_main=model_main,
            label_name=label_name,
            sample_id=sample_id,
        )
        make_single_figure(label_name, sample_id, data)


if __name__ == "__main__":
    main()