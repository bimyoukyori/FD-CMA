# pipelines/stage6b_export_training_artifacts.py
from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:
    from pipelines.config import ROOT
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipelines.config import ROOT


RUNS_ROOT = ROOT / "ultralytics4" / "runs_paper_rtdetr"
TRAIN_DIR = RUNS_ROOT / "train"
ARTIFACT_DIR = RUNS_ROOT / "artifacts"
WEIGHT_DIR = RUNS_ROOT / "weights_release"

EXP_PREFIX = {
    "E1": ("E1_baseline_coco",),
    "E2": ("E2_ours_dinit",),
    "E3": ("E3_ours_dinit_freeze",),
    "E4": ("E4_ours_enhanced",),
}


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


def _read_best_metrics(results_csv: Path) -> Dict[str, float]:
    with results_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = reader.fieldnames or []

    if not rows:
        raise RuntimeError(f"empty results.csv: {results_csv}")

    col_epoch = _find_col(cols, (("epoch",),))
    col_p = _find_col(cols, (("metricsprecisionb",), ("precision",)))
    col_r = _find_col(cols, (("metricsrecallb",), ("recall",)))
    col_m50 = _find_col(cols, (("metricsmap50b",), ("map50b",), ("map50",)))
    col_m95 = _find_col(cols, (("metricsmap5095b",), ("map5095",), ("map50", "95")))
    if col_m95 is None:
        raise RuntimeError(f"mAP50-95 column not found: {results_csv}")
    if col_m50 == col_m95:
        col_m50 = _find_col(cols, (("metricsmap50b",), ("map50b",)))

    best_i = 0
    best_v = -1.0
    for i, r in enumerate(rows):
        v = _safe_float(r.get(col_m95, "nan"))
        if v == v and v > best_v:
            best_v = v
            best_i = i
    br = rows[best_i]

    return dict(
        best_epoch=int(_safe_float(br.get(col_epoch, best_i), default=float(best_i))),
        precision=_safe_float(br.get(col_p, "nan")),
        recall=_safe_float(br.get(col_r, "nan")),
        map50=_safe_float(br.get(col_m50, "nan")),
        map50_95=_safe_float(br.get(col_m95, "nan")),
    )


def _pick_best_run(exp: str, prefixes: Sequence[str]) -> Path:
    cands: List[Path] = []
    for d in TRAIN_DIR.iterdir():
        if d.is_dir() and any(d.name.startswith(px) for px in prefixes):
            cands.append(d)
    if not cands:
        raise RuntimeError(f"[Stage6b] no run found for {exp}, prefixes={prefixes}")

    scored = []
    for d in cands:
        r = d / "results.csv"
        if not r.exists():
            continue
        try:
            m = _read_best_metrics(r)
            score = float(m["map50_95"])
        except Exception:
            score = float("-inf")
        scored.append((score, d.stat().st_mtime, d))

    if not scored:
        raise RuntimeError(f"[Stage6b] runs for {exp} found, but none has valid results.csv")
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


def run() -> None:
    print("[Stage6b] Export canonical training artifacts for E1~E4")
    if not TRAIN_DIR.exists():
        raise RuntimeError(f"[Stage6b] train dir not found: {TRAIN_DIR}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)

    binding_rows = []
    metrics_rows = []
    for exp, prefixes in EXP_PREFIX.items():
        rd = _pick_best_run(exp, prefixes)
        rcsv = rd / "results.csv"
        best_pt = rd / "weights" / "best.pt"
        if not best_pt.exists():
            raise RuntimeError(f"[Stage6b] {exp} best.pt not found: {best_pt}")

        metrics = _read_best_metrics(rcsv)
        binding_rows.append([exp, rd.name, str(rd), str(rcsv), str(best_pt)])
        metrics_rows.append(
            [
                exp,
                metrics["best_epoch"],
                f"{metrics['precision']:.10f}",
                f"{metrics['recall']:.10f}",
                f"{metrics['map50']:.10f}",
                f"{metrics['map50_95']:.10f}",
            ]
        )

        shutil.copy2(best_pt, WEIGHT_DIR / f"{exp}_best.pt")
        shutil.copy2(rcsv, ARTIFACT_DIR / f"{exp}_results.csv")

    with (ARTIFACT_DIR / "run_binding.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["exp", "run_name", "run_dir", "results_csv", "best_weight"])
        w.writerows(binding_rows)

    with (ARTIFACT_DIR / "best_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["exp", "best_epoch", "precision", "recall", "map50", "map50_95"])
        w.writerows(metrics_rows)

    print(f"[Stage6b] Done. artifacts: {ARTIFACT_DIR}")
    print(f"[Stage6b] Done. weights  : {WEIGHT_DIR}")


if __name__ == "__main__":
    run()

