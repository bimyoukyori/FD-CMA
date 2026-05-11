#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build RT-DETR distilled initialization checkpoint.

Logic:
1) Load COCO pretrained `rtdetr-l.pt`
2) Load distillation checkpoint from Stage4
3) Extract backbone weights from distilled student state
4) Overlay backbone onto COCO model (neck/head remain COCO pretrained)
5) Save final init weights for downstream detection training
"""

from __future__ import annotations

from pathlib import Path
import sys
import torch


ROOT = Path(__file__).resolve().parents[2]  # .../MUL05
ULTRA_ROOT = ROOT / "ultralytics4"
sys.path.insert(0, str(ULTRA_ROOT))

from ultralytics import RTDETR  # noqa: E402


def _extract_student_state(ckpt: dict) -> dict:
    if "student_det_state" in ckpt and isinstance(ckpt["student_det_state"], dict):
        return ckpt["student_det_state"]
    if "student_state" in ckpt and isinstance(ckpt["student_state"], dict):
        return ckpt["student_state"]
    if all(isinstance(k, str) for k in ckpt.keys()) and any(str(k).startswith("model.") for k in ckpt.keys()):
        # In case checkpoint itself is a plain RT-DETR state_dict.
        return ckpt
    raise KeyError("Cannot find student detector state in distilled checkpoint.")


def main():
    distilled_ckpt = ROOT / "ultralytics4" / "distill" / "runs" / "exp_rtdetr_distill" / "rtdetr_distilled.pth"
    out_dir = ROOT / "ultralytics4" / "distill" / "runs" / "exp_rtdetr_distill"
    out_ckpt = out_dir / "rtdetr_l_distilled_init.pt"
    device = "cpu"

    if not distilled_ckpt.exists():
        raise FileNotFoundError(f"Distilled checkpoint not found: {distilled_ckpt}")

    print("[INFO] Loading COCO pretrained RT-DETR-L from rtdetr-l.pt")
    model = RTDETR("rtdetr-l.pt")
    det_model = model.model.to(device)

    print(f"[INFO] Loading distilled checkpoint: {distilled_ckpt}")
    ckpt = torch.load(distilled_ckpt, map_location="cpu", weights_only=True)
    student_state = _extract_student_state(ckpt)

    backbone_state = {k: v for k, v in student_state.items() if k.startswith("model.0.")}
    if not backbone_state:
        raise RuntimeError("No backbone weights found in student state (expected prefix: model.0.)")

    print(f"[INFO] Distilled backbone params: {len(backbone_state)}")
    missing, unexpected = det_model.load_state_dict(backbone_state, strict=False)
    print(f"[CHECK] Missing keys after overlay: {len(missing)}")
    print(f"[CHECK] Unexpected keys after overlay: {len(unexpected)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(det_model.state_dict(), out_ckpt)
    print("[OK] Distilled init checkpoint saved:")
    print(f"  {out_ckpt}")


if __name__ == "__main__":
    main()

