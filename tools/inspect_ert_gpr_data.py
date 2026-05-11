#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect ERT/GPR artifacts in this repo (MUL06).

What it checks:
1) ERT/GPR feature npy counts, IDs, dims, NaN/Inf, basic stats
2) Existence of raw ERT LWF, ERT NPY, GPR CSV files
3) Consistency with EFM5/data/pairs.csv (if present)

Designed for migrated projects where some artifacts already exist.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass(frozen=True)
class NpySummary:
    count: int
    ids: List[str]
    dims: Dict[int, int]  # dim -> count
    bad_nan: int
    bad_inf: int
    non_1d: int
    min_v: float
    max_v: float
    mean_v: float


def _extract_id_from_stem(stem: str) -> Optional[str]:
    s = stem.strip()
    if s.isdigit():
        return s
    # common patterns: "00001" / "xxx_00001_yyy"
    parts = [p for p in s.replace("-", "_").split("_") if p]
    for p in parts:
        if p.isdigit():
            return p
    return None


def _list_npy_ids(dir_path: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not dir_path.exists():
        return out
    for p in sorted(dir_path.glob("*.npy")):
        sid = _extract_id_from_stem(p.stem)
        if not sid:
            continue
        # keep first occurrence
        out.setdefault(sid, p)
    return out


def _finite_stats(arr: np.ndarray) -> Tuple[float, float, float]:
    a = arr.astype(np.float64, copy=False).reshape(-1)
    m = np.isfinite(a)
    if not np.any(m):
        return (float("nan"), float("nan"), float("nan"))
    aa = a[m]
    return (float(np.min(aa)), float(np.max(aa)), float(np.mean(aa)))


def summarize_npy_dir(dir_path: Path, sample_limit: int = 50) -> NpySummary:
    id_map = _list_npy_ids(dir_path)
    ids = sorted(id_map.keys(), key=lambda x: int(x))

    dims: Dict[int, int] = {}
    bad_nan = 0
    bad_inf = 0
    non_1d = 0

    min_list: List[float] = []
    max_list: List[float] = []
    mean_list: List[float] = []

    # sample a subset for speed
    sample_ids = ids[:sample_limit] if sample_limit > 0 else ids
    for sid in sample_ids:
        p = id_map[sid]
        try:
            arr = np.load(p, allow_pickle=False)
        except Exception:
            continue

        if arr.ndim != 1:
            non_1d += 1
            arr = arr.reshape(-1)

        dims[int(arr.shape[0])] = dims.get(int(arr.shape[0]), 0) + 1

        if np.isnan(arr).any():
            bad_nan += 1
        if np.isinf(arr).any():
            bad_inf += 1

        mn, mx, mu = _finite_stats(arr)
        min_list.append(mn)
        max_list.append(mx)
        mean_list.append(mu)

    def _agg(xs: Sequence[float], fn) -> float:
        a = np.array(xs, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return float("nan")
        return float(fn(a))

    return NpySummary(
        count=len(ids),
        ids=ids,
        dims=dims,
        bad_nan=bad_nan,
        bad_inf=bad_inf,
        non_1d=non_1d,
        min_v=_agg(min_list, np.min),
        max_v=_agg(max_list, np.max),
        mean_v=_agg(mean_list, np.mean),
    )


def _read_pairs(pairs_csv: Path) -> List[Dict[str, str]]:
    if not pairs_csv.exists():
        return []
    with pairs_csv.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _existing_ids_by_pattern(dir_path: Path, glob_pat: str) -> Set[str]:
    if not dir_path.exists():
        return set()
    ids: Set[str] = set()
    for p in dir_path.glob(glob_pat):
        sid = _extract_id_from_stem(p.stem)
        if sid:
            ids.add(sid)
    return ids


def _fmt_dims(dims: Dict[int, int]) -> str:
    if not dims:
        return "(no samples read)"
    items = sorted(dims.items(), key=lambda x: (-x[1], x[0]))
    return ", ".join([f"{d}×{c}" for d, c in items[:6]]) + (" ..." if len(items) > 6 else "")


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect ERT/GPR data & feature artifacts (MUL06)")
    ap.add_argument(
        "--root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="Repo root path (default: inferred from this script location).",
    )
    ap.add_argument("--sample-limit", type=int, default=50, help="Max npy files to sample for stats per dir.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    efm5 = root / "EFM5"

    ert_feat_dir = efm5 / "features" / "ert"
    gpr_feat_dir = efm5 / "features" / "gpr"
    pairs_csv = efm5 / "data" / "pairs.csv"

    ert_lwf_dir = efm5 / "data" / "ert" / "ert_lwf"
    ert_npy_dir = efm5 / "data" / "ert" / "ert_npy"
    gpr_train_dir = efm5 / "data" / "gpr" / "train"

    print("============================================================")
    print("  MUL06 ERT/GPR Artifacts Inspector")
    print("============================================================")
    print(f"[ROOT]  {root}")
    print(f"[EFM5]  {efm5}")
    print("")

    # Features
    ert_sum = summarize_npy_dir(ert_feat_dir, sample_limit=args.sample_limit)
    gpr_sum = summarize_npy_dir(gpr_feat_dir, sample_limit=args.sample_limit)

    print("[FEATURES] ERT")
    print(f"  dir        : {ert_feat_dir}")
    print(f"  count(ids) : {ert_sum.count}")
    print(f"  dims(sample): {_fmt_dims(ert_sum.dims)}")
    print(f"  non_1d(sample): {ert_sum.non_1d}  nan(sample): {ert_sum.bad_nan}  inf(sample): {ert_sum.bad_inf}")
    print(f"  stats(sample): min={ert_sum.min_v:.6g} max={ert_sum.max_v:.6g} mean={ert_sum.mean_v:.6g}")
    print(f"  ids(head)  : {', '.join(ert_sum.ids[:12])}{' ...' if ert_sum.count > 12 else ''}")
    print("")

    print("[FEATURES] GPR")
    print(f"  dir        : {gpr_feat_dir}")
    print(f"  count(ids) : {gpr_sum.count}")
    print(f"  dims(sample): {_fmt_dims(gpr_sum.dims)}")
    print(f"  non_1d(sample): {gpr_sum.non_1d}  nan(sample): {gpr_sum.bad_nan}  inf(sample): {gpr_sum.bad_inf}")
    print(f"  stats(sample): min={gpr_sum.min_v:.6g} max={gpr_sum.max_v:.6g} mean={gpr_sum.mean_v:.6g}")
    print(f"  ids(head)  : {', '.join(gpr_sum.ids[:12])}{' ...' if gpr_sum.count > 12 else ''}")
    print("")

    # Raw dirs existence
    ert_lwf_ids = _existing_ids_by_pattern(ert_lwf_dir, "*_lwf.dat")
    ert_npy_ids = _existing_ids_by_pattern(ert_npy_dir, "*.npy")

    # gpr ids by scanning all class subfolders
    gpr_csv_ids: Set[str] = set()
    if gpr_train_dir.exists():
        for class_dir in gpr_train_dir.iterdir():
            if not class_dir.is_dir():
                continue
            gpr_csv_ids |= _existing_ids_by_pattern(class_dir, "*.csv")

    print("[RAW] ERT/GPR file presence")
    print(f"  ERT LWF  : {len(ert_lwf_ids)} ids  ({ert_lwf_dir})")
    print(f"  ERT NPY  : {len(ert_npy_ids)} ids  ({ert_npy_dir})")
    print(f"  GPR CSV  : {len(gpr_csv_ids)} ids  ({gpr_train_dir})")
    print("")

    # Overlaps
    feat_ert_ids = set(ert_sum.ids)
    feat_gpr_ids = set(gpr_sum.ids)
    inter_feat = feat_ert_ids & feat_gpr_ids
    print("[OVERLAP] Feature ID overlap (ERT ∩ GPR)")
    print(f"  overlap  : {len(inter_feat)}")
    if inter_feat:
        head = sorted(list(inter_feat), key=lambda x: int(x))[:20]
        print(f"  head     : {', '.join(head)}{' ...' if len(inter_feat) > 20 else ''}")
    print("")

    # pairs.csv
    rows = _read_pairs(pairs_csv)
    if not rows:
        print("[PAIRS] pairs.csv not found or empty -> skip")
        return 0

    pair_ids = []
    missing_ert_feat = 0
    missing_gpr_feat = 0
    for r in rows:
        sid = (r.get("id") or "").strip()
        if not sid:
            continue
        pair_ids.append(sid)
        if sid not in feat_ert_ids:
            missing_ert_feat += 1
        if sid not in feat_gpr_ids:
            missing_gpr_feat += 1

    pair_set = set(pair_ids)
    print("[PAIRS] Consistency check")
    print(f"  pairs.csv : {pairs_csv}")
    print(f"  rows(ids) : {len(pair_ids)}")
    print(f"  uniq(ids) : {len(pair_set)}")
    print(f"  missing ERT feature : {missing_ert_feat}")
    print(f"  missing GPR feature : {missing_gpr_feat}")

    # show some mismatches
    miss_ert = sorted(list(pair_set - feat_ert_ids), key=lambda x: int(x))[:20]
    miss_gpr = sorted(list(pair_set - feat_gpr_ids), key=lambda x: int(x))[:20]
    if miss_ert:
        print(f"  ids missing ERT feat (head): {', '.join(miss_ert)}{' ...' if (len(pair_set - feat_ert_ids) > 20) else ''}")
    if miss_gpr:
        print(f"  ids missing GPR feat (head): {', '.join(miss_gpr)}{' ...' if (len(pair_set - feat_gpr_ids) > 20) else ''}")
    print("")

    print("[HINT]")
    print("  - Stage3 (fusion) uses: EFM5/data/pairs.csv + EFM5/features/ert/*.npy + EFM5/features/gpr/*.npy")
    print("  - To increase usable pairs, ensure SAME IDs exist on both sides.")
    print("============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

