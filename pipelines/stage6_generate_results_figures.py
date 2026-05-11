# pipelines/stage6_generate_results_figures.py
from __future__ import annotations

import subprocess
import os

try:
    from pipelines.config import ROOT, PYTHON, LOG_DIR
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipelines.config import ROOT, PYTHON, LOG_DIR


EXPECTED_FIGS = [
    "fig0_summary.png",
    "fig1_training_dynamics.png",
    "fig2_val_loss.png",
    "fig3_pr_curves.png",
    "fig4_bar_comparison.png",
    "fig5_radar.png",
    "fig6_confusion_matrix.png",
    "fig7_per_class_map.png",
    "fig8_panorama_detection.png",
    "fig8b_panorama_caption.png",
    "fig9_single_profiles.png",
    "fig10_e1_vs_e4_comparison.png",
]


def run():
    print("[Stage6] Generate result figures from main.py")

    script = ROOT / "main.py"
    if not script.exists():
        raise FileNotFoundError(f"[Stage6] Script not found: {script}")

    log = LOG_DIR / "stage6_generate_results_figures.log"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    with log.open("w", encoding="utf-8") as f:
        subprocess.run(
            [PYTHON, str(script)],
            cwd=str(ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )

    out_dir = ROOT / "results"
    if not out_dir.exists():
        raise RuntimeError(f"[Stage6] results dir not found: {out_dir}")

    missing = [x for x in EXPECTED_FIGS if not (out_dir / x).exists()]
    if missing:
        raise RuntimeError(
            "[Stage6] Missing expected figures:\n  - " + "\n  - ".join(missing)
        )

    print(f"[Stage6] Done. Figure count={len(EXPECTED_FIGS)}")
    print(f"[Stage6] Log: {log}")


if __name__ == "__main__":
    run()
