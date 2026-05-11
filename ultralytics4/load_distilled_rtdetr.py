#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility loader: convert distilled RT-DETR checkpoint into init `.pt` weights.
"""

from __future__ import annotations

from pathlib import Path
import torch
from ultralytics import RTDETR


def _extract_student_state(ckpt: dict) -> dict:
    if "student_det_state" in ckpt and isinstance(ckpt["student_det_state"], dict):
        return ckpt["student_det_state"]
    if "student_state" in ckpt and isinstance(ckpt["student_state"], dict):
        return ckpt["student_state"]
    raise KeyError("Expected `student_det_state` or `student_state` in checkpoint.")


def main():
    distilled_ckpt = Path("distill/runs/exp_rtdetr_distill/rtdetr_distilled.pth")
    out_ckpt = Path("distill/runs/exp_rtdetr_distill/rtdetr_l_distilled_init.pt")
    device = "cpu"

    print("[INFO] Loading RT-DETR model from YAML...")
    model = RTDETR("rtdetr-l.yaml")
    det_model = model.model.to(device)

    print(f"[INFO] Loading distilled checkpoint: {distilled_ckpt}")
    ckpt = torch.load(distilled_ckpt, map_location="cpu", weights_only=True)
    student_state = _extract_student_state(ckpt)

    print("[INFO] Loading distilled state into RT-DETR (strict=False)...")
    missing, unexpected = det_model.load_state_dict(student_state, strict=False)
    print(f"[CHECK] Missing keys: {len(missing)}")
    print(f"[CHECK] Unexpected keys: {len(unexpected)}")

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(det_model.state_dict(), out_ckpt)
    print("[OK] Distilled RT-DETR init checkpoint saved to:")
    print(f"  {out_ckpt}")


if __name__ == "__main__":
    main()

