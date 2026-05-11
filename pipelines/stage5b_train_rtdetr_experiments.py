# pipelines/stage5b_train_rtdetr_experiments.py
from __future__ import annotations

import os
import subprocess
from pathlib import Path

try:
    from pipelines.config import ROOT, PYTHON, ULTRA_DIR, LOG_DIR
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipelines.config import ROOT, PYTHON, ULTRA_DIR, LOG_DIR


EXPECTED_PREFIX = {
    "E1": "E1_baseline_coco",
    "E2": "E2_ours_dinit",
    "E3": "E3_ours_dinit_freeze",
    "E4": "E4_ours_enhanced",
}


def _resolve_data_yaml_from_env() -> Path:
    p = os.getenv(
        "MUL05_DATA_YAML",
        str(ULTRA_DIR / "ultralytics" / "cfg" / "datasets" / "gpr5-detect.yaml"),
    )
    path = Path(p)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _check_data_yaml_exists() -> None:
    yaml_path = _resolve_data_yaml_from_env()
    if not yaml_path.exists():
        raise FileNotFoundError(f"[Stage5b] DATA_YAML not found: {yaml_path}")
    print(f"[Stage5b] DATA_YAML = {yaml_path}")
    try:
        import yaml  # type: ignore

        with yaml_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if isinstance(cfg, dict):
            for key in ("train", "val", "test"):
                v = str(cfg.get(key, "")).strip()
                if not v:
                    continue
                p = Path(v)
                if p.is_absolute() and not p.exists():
                    print(f"[Stage5b][WARN] {key} path not found: {p}")
    except Exception as e:
        print(f"[Stage5b][WARN] dataset path precheck skipped: {e}")


def run() -> None:
    print("[Stage5b] Train RT-DETR E1~E4 experiments (real runs)")
    _check_data_yaml_exists()

    script = ULTRA_DIR / "ultralytics" / "train_rtdetr_e2_paper10.py"
    if not script.exists():
        raise FileNotFoundError(f"[Stage5b] training script not found: {script}")

    log = LOG_DIR / "stage5b_train_rtdetr_experiments.log"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault(
        "RUN_EXPERIMENTS",
        "E1_baseline_coco,E2_ours_dinit,E3_ours_dinit_freeze,E4_ours_enhanced",
    )

    with log.open("w", encoding="utf-8") as f:
        subprocess.run(
            [PYTHON, str(script)],
            cwd=str(ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )

    train_dir = ROOT / "ultralytics4" / "runs_paper_rtdetr" / "train"
    if not train_dir.exists():
        raise RuntimeError(f"[Stage5b] train run dir not found: {train_dir}")

    for k, prefix in EXPECTED_PREFIX.items():
        matched = [d for d in train_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
        if not matched:
            raise RuntimeError(f"[Stage5b] missing run for {k}, prefix={prefix}")
        if not any((d / "results.csv").exists() for d in matched):
            raise RuntimeError(f"[Stage5b] {k} matched run exists but no results.csv found")

    print("[Stage5b] Done.")
    print(f"[Stage5b] Log: {log}")


if __name__ == "__main__":
    run()
