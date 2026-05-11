#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_bar_2x2.py
============================================================
Standalone bar chart (2x2) for E1~E4 final detection metrics.

- Read Ultralytics results.csv from:
    <ROOT>/train/E1_baseline_coco/results.csv
    <ROOT>/train/E2_ours_dinit/results.csv
    <ROOT>/train/E3_ours_dinit_freeze/results.csv
    <ROOT>/train/E4_ours_enhanced/results.csv

- Use best epoch picked by max(mAP@0.5:0.95).
- Plot 2x2 layout:
    (a) mAP@0.5
    (b) mAP@0.5:0.95
    (c) Precision
    (d) Recall

Outputs:
    <ROOT>/results/fig5_6_bar_2x2.png
    <ROOT>/results/raw/bar_metrics_used.csv
============================================================
"""

from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# =========================
# Config
# =========================
ROOT = Path(__file__).resolve().parents[1]  # tools/plot_bar_2x2.py -> MUL06/
TRAIN_DIR = ROOT / "train"
OUTDIR = ROOT / "results"
RAWDIR = OUTDIR / "raw"

OUTDIR.mkdir(exist_ok=True)
RAWDIR.mkdir(exist_ok=True)

EXP_META = OrderedDict(
    {
        "E1": dict(run_name="E1_baseline_coco", label="E1\nBaseline"),
        "E2": dict(run_name="E2_ours_dinit", label="E2\nOurs-init"),
        "E3": dict(run_name="E3_ours_dinit_freeze", label="E3\nOurs-freeze"),
        "E4": dict(run_name="E4_ours_enhanced", label="E4\nOurs-Enhanced"),
    }
)

# 只用于区分实验柱的颜色：不强制论文配色，你需要与全文统一再改
BAR_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


# =========================
# Helpers
# =========================
def _safe_float(v: object, default: float = float("nan")) -> float:
    try:
        return float(str(v).strip())
    except Exception:
        return default


def _norm_col(c: str) -> str:
    return "".join(ch for ch in str(c).lower() if ch.isalnum())


def _find_col(cols: Sequence[str], candidates: Sequence[Sequence[str]]) -> Optional[str]:
    norm = {c: _norm_col(c) for c in cols}
    for keys in candidates:
        for c, nc in norm.items():
            if all(k in nc for k in keys):
                return c
    return None


def read_results_csv(csv_path: Path) -> Dict[str, object]:
    if not csv_path.exists():
        raise FileNotFoundError(f"results.csv not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = reader.fieldnames or []

    if not rows:
        raise RuntimeError(f"results.csv is empty: {csv_path}")

    col_epoch = _find_col(cols, (("epoch",),))
    col_prec = _find_col(cols, (("metricsprecisionb",), ("precision",)))
    col_rec = _find_col(cols, (("metricsrecallb",), ("recall",)))
    col_map95 = _find_col(cols, (("metricsmap5095b",), ("map5095",), ("map50", "95")))
    col_map50 = _find_col(cols, (("metricsmap50b",), ("map50b",), ("map50",)))

    if col_map95 is None:
        raise RuntimeError(f"Cannot find mAP50-95 column in {csv_path}")

    if col_map50 == col_map95:
        col_map50 = _find_col(cols, (("metricsmap50b",), ("map50b",)))

    epochs, prec, rec, map50, map95 = [], [], [], [], []
    for i, r in enumerate(rows):
        ep = _safe_float(r.get(col_epoch, i), default=float(i)) if col_epoch else float(i)
        epochs.append(ep)
        prec.append(_safe_float(r.get(col_prec, np.nan)))
        rec.append(_safe_float(r.get(col_rec, np.nan)))
        map50.append(_safe_float(r.get(col_map50, np.nan)))
        map95.append(_safe_float(r.get(col_map95, np.nan)))

    epochs = np.array(epochs, dtype=float)
    if np.nanmin(epochs) <= 0.0:
        epochs += 1.0

    map95_arr = np.array(map95, dtype=float)
    valid = np.where(np.isfinite(map95_arr))[0]
    best_idx = int(valid[np.argmax(map95_arr[valid])]) if valid.size else int(len(map95_arr) - 1)

    return dict(
        epochs=epochs,
        precision=np.array(prec, dtype=float),
        recall=np.array(rec, dtype=float),
        map50=np.array(map50, dtype=float),
        map50_95=map95_arr,
        best_idx=best_idx,
    )


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for r in rows:
            w.writerow(list(r))


# =========================
# Main Plot
# =========================
def main() -> None:
    # 1) load best metrics
    metrics = OrderedDict()
    raw_rows: List[List[object]] = []

    for eid, meta in EXP_META.items():
        run_dir = TRAIN_DIR / meta["run_name"]
        rcsv = run_dir / "results.csv"
        data = read_results_csv(rcsv)
        bi = int(data["best_idx"])

        m = dict(
            best_epoch=int(round(float(data["epochs"][bi]))),
            map50=float(data["map50"][bi]),
            map50_95=float(data["map50_95"][bi]),
            precision=float(data["precision"][bi]),
            recall=float(data["recall"][bi]),
        )
        metrics[eid] = m
        raw_rows.append([eid, meta["run_name"], m["best_epoch"], m["map50"], m["map50_95"], m["precision"], m["recall"]])

    write_csv(
        RAWDIR / "bar_metrics_used.csv",
        ["exp", "run_name", "best_epoch", "map50", "map50_95", "precision", "recall"],
        raw_rows,
    )

    # 2) build 2x2 bars
    labels = [EXP_META[e]["label"] for e in EXP_META.keys()]
    vals_map50 = [metrics[e]["map50"] for e in EXP_META.keys()]
    vals_map95 = [metrics[e]["map50_95"] for e in EXP_META.keys()]
    vals_prec = [metrics[e]["precision"] for e in EXP_META.keys()]
    vals_rec = [metrics[e]["recall"] for e in EXP_META.keys()]
    colors = BAR_COLORS[: len(labels)]

    plt.rcParams.update(
        {
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.30,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.2))
    fig.suptitle("Final Detection Performance Comparison (E1–E4)", fontsize=13, fontweight="bold", y=0.99)

    def _bar(ax, title: str, values: List[float]) -> None:
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=colors, width=0.55, edgecolor="white", linewidth=1.2, alpha=0.92)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(max(0.0, float(np.nanmin(values)) * 0.90), 1.02)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() * 0.5, v + 0.004, f"{v:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        # 标注最优
        best_i = int(np.nanargmax(values))
        bars[best_i].set_edgecolor("gold")
        bars[best_i].set_linewidth(2.4)

    _bar(axes[0, 0], "(a) mAP@0.5", vals_map50)
    _bar(axes[0, 1], "(b) mAP@0.5:0.95", vals_map95)
    _bar(axes[1, 0], "(c) Precision", vals_prec)
    _bar(axes[1, 1], "(d) Recall", vals_rec)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = OUTDIR / "fig5_6_bar_2x2.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"[OK] saved: {out_path.relative_to(ROOT)}")
    print(f"[OK] raw metrics: {(RAWDIR / 'bar_metrics_used.csv').relative_to(ROOT)}")
    for eid in EXP_META.keys():
        m = metrics[eid]
        print(f"  {eid}: epoch={m['best_epoch']}  mAP50={m['map50']:.4f}  mAP50-95={m['map50_95']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")


if __name__ == "__main__":
    main()