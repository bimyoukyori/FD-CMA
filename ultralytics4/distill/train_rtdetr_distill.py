#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RT-DETR backbone distillation from fusion GPR encoder.

Teacher:
  - EFM5/fusion/runs/exp_fusion_main/fused_backbone.pth
Student:
  - Ultralytics RT-DETR backbone feature projection
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Training config
# ============================================================

EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42
STUDENT_INPUT_HW = (224, 224)


# ============================================================
# Path anchors
# ============================================================

def _find_efm5_root(this_file: Path) -> Path:
    for p in this_file.parents:
        cand = p / "EFM5"
        if cand.exists() and cand.is_dir():
            return cand
    raise RuntimeError(f"Cannot locate EFM5 root from: {this_file}")


THIS_FILE = Path(__file__).resolve()
EFM5_ROOT = _find_efm5_root(THIS_FILE)
ULTRA_ROOT = THIS_FILE.parents[1]  # .../MUL05/ultralytics4

DATA_ROOT = EFM5_ROOT / "data"
FEATURE_ROOT = EFM5_ROOT / "features"
FUSION_RUN_DIR = EFM5_ROOT / "fusion" / "runs" / "exp_fusion_main"
DISTILL_RUN_DIR = ULTRA_ROOT / "distill" / "runs" / "exp_rtdetr_distill"

PAIRS_CSV = DATA_ROOT / "pairs.csv"
GPR_FEATURE_DIR = FEATURE_ROOT / "gpr"
FUSION_BACKBONE_PATH = FUSION_RUN_DIR / "fused_backbone.pth"
FUSION_META_PATH = FUSION_RUN_DIR / "meta.json"

assert PAIRS_CSV.exists(), f"pairs.csv missing: {PAIRS_CSV}"
assert GPR_FEATURE_DIR.exists(), f"gpr feature dir missing: {GPR_FEATURE_DIR}"
assert FUSION_BACKBONE_PATH.exists(), f"fused_backbone missing: {FUSION_BACKBONE_PATH}"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_fusion_meta(meta_path: Path) -> Dict[str, object]:
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


FUSION_META = load_fusion_meta(FUSION_META_PATH)
EMBED_DIM = int(FUSION_META.get("emb_dim", 256))
GPR_IN_CHANNELS = int(FUSION_META.get("gpr_input_channels", 3))
GPR_BASE_CHANNELS = int(FUSION_META.get("gpr_base_channels", 32))
GPR_DCT_HW = int(FUSION_META.get("gpr_dct_hw", 32))
_hw = FUSION_META.get("gpr_image_hw", [128, 128])
if not isinstance(_hw, list) or len(_hw) != 2:
    _hw = [128, 128]
GPR_IMAGE_HW = (int(_hw[0]), int(_hw[1]))  # (H, W)


def _resolve_gpr_feat_path(sample_id: str) -> Optional[Path]:
    sid = str(sample_id).strip()
    cands = [
        GPR_FEATURE_DIR / f"{sid}.npy",
        GPR_FEATURE_DIR / f"{sid.zfill(5)}.npy",
    ]
    for p in cands:
        if p.exists():
            return p
    return None


def preprocess_gpr_feature_to_image(x: np.ndarray) -> torch.Tensor:
    """
    Convert gpr feature array to image-like tensor (C,H,W).
    """
    if x.ndim == 1:
        expect = GPR_DCT_HW * GPR_DCT_HW
        if x.shape[0] != expect:
            raise RuntimeError(f"GPR 1D size mismatch: {x.shape[0]} vs {expect}")
        x2d = x.reshape(GPR_DCT_HW, GPR_DCT_HW)
    elif x.ndim == 2:
        x2d = x
    elif x.ndim == 3 and x.shape[0] in {1, 3}:
        t = torch.from_numpy(x.astype("float32"))
        if tuple(t.shape[1:]) != GPR_IMAGE_HW:
            t = F.interpolate(t.unsqueeze(0), size=GPR_IMAGE_HW, mode="bilinear", align_corners=False).squeeze(0)
        if t.shape[0] == 1 and GPR_IN_CHANNELS == 3:
            t = t.repeat(3, 1, 1)
        if t.shape[0] == 3 and GPR_IN_CHANNELS == 1:
            t = t.mean(dim=0, keepdim=True)
        return t
    else:
        raise RuntimeError(f"Unsupported gpr feature shape: {x.shape}")

    lo, hi = np.percentile(x2d, [1, 99])
    if hi <= lo:
        lo = float(x2d.min())
        hi = float(x2d.max()) + 1e-6
    x2d = np.clip((x2d - lo) / (hi - lo + 1e-6), 0.0, 1.0)

    t = torch.from_numpy(x2d.astype("float32")).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    if tuple(x2d.shape) != GPR_IMAGE_HW:
        t = F.interpolate(t, size=GPR_IMAGE_HW, mode="bilinear", align_corners=False)
    t = t.squeeze(0)  # (1,H,W)
    if GPR_IN_CHANNELS == 3:
        t = t.repeat(3, 1, 1)
    return t


class GPRDistillDataset(Dataset):
    """
    Output:
      x_gpr: (C, H, W), image-like tensor
    """

    def __init__(self, csv_path: Path):
        rows: List[Dict[str, str]] = []
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        if not rows:
            raise RuntimeError("pairs.csv is empty")

        valid = []
        missing = 0
        for r in rows:
            sid = (r.get("id") or "").strip()
            p = _resolve_gpr_feat_path(sid)
            if p is None:
                missing += 1
                continue
            valid.append((sid, p))
        if not valid:
            raise RuntimeError("No valid gpr features for distillation.")

        self.valid_items: List[Tuple[str, Path]] = valid
        print(f"[INFO] distill pairs: total={len(rows)} valid={len(valid)} missing_gpr={missing}")
        print(f"[INFO] teacher input: channels={GPR_IN_CHANNELS} image_hw={GPR_IMAGE_HW} dct_hw={GPR_DCT_HW}")

    def __len__(self) -> int:
        return len(self.valid_items)

    def __getitem__(self, idx: int):
        _, p = self.valid_items[idx]
        x = np.load(p).astype("float32")
        return preprocess_gpr_feature_to_image(x)


def build_teacher(device: torch.device) -> nn.Module:
    """
    Teacher architecture is imported from EFM5 fusion_net to ensure strict homology.
    """
    fusion_dir = EFM5_ROOT / "fusion"
    sys.path.insert(0, str(fusion_dir))
    from fusion_net import GPRImageEncoder  # pylint: disable=import-error

    teacher = GPRImageEncoder(
        in_ch=GPR_IN_CHANNELS,
        base=GPR_BASE_CHANNELS,
        out_dim=EMBED_DIM,
    ).to(device)

    state = torch.load(FUSION_BACKBONE_PATH, map_location="cpu", weights_only=True)
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print("[INFO] teacher loaded from:", FUSION_BACKBONE_PATH)
    return teacher


def _pick_last_valid_4d(obj, min_c: int = 32) -> Optional[torch.Tensor]:
    last = None

    def walk(x):
        nonlocal last
        if torch.is_tensor(x):
            if x.ndim == 4 and int(x.shape[1]) >= min_c:
                last = x
            return
        if isinstance(x, (list, tuple)):
            for t in x:
                walk(t)
            return
        if isinstance(x, dict):
            for _, v in x.items():
                walk(v)
            return

    walk(obj)
    return last


def _resolve_rtdetr_backbone(det_model: nn.Module) -> nn.Module:
    if hasattr(det_model, "backbone") and isinstance(getattr(det_model, "backbone"), nn.Module):
        return getattr(det_model, "backbone")
    if hasattr(det_model, "model"):
        m = getattr(det_model, "model")
        if isinstance(m, (nn.ModuleList, nn.Sequential, list, tuple)) and len(m) > 0 and isinstance(m[0], nn.Module):
            return m[0]
    raise RuntimeError("Cannot resolve RT-DETR backbone module.")


class RTDETRStudent(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        from ultralytics import RTDETR

        wrapper = RTDETR("rtdetr-l.yaml")
        self.det = wrapper.model.to(device)

        self.backbone = _resolve_rtdetr_backbone(self.det)
        self._feat4d: Optional[torch.Tensor] = None

        def _backbone_hook(_module, _inp, out):
            self._feat4d = _pick_last_valid_4d(out, min_c=32)

        self.backbone.register_forward_hook(_backbone_hook)

        with torch.no_grad():
            dummy = torch.zeros((1, 3, STUDENT_INPUT_HW[0], STUDENT_INPUT_HW[1]), device=device)
            _ = self.det(dummy)
            if self._feat4d is None:
                raise RuntimeError("Backbone hook failed to capture valid 4D feature map.")
            c = int(self._feat4d.shape[1])

        self.proj = nn.Linear(c, EMBED_DIM).to(device)
        print(f"[INFO] student backbone channels={c}, proj_out={EMBED_DIM}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.det(x)
        feat4d = self._feat4d
        if feat4d is None:
            raise RuntimeError("No 4D backbone feature captured in forward.")
        feat = feat4d.mean(dim=(2, 3))
        feat = self.proj(feat)
        return F.normalize(feat, dim=1)


def main():
    set_seed(SEED)
    DISTILL_RUN_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    dataset = GPRDistillDataset(PAIRS_CSV)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    teacher = build_teacher(device)
    student = RTDETRStudent(device=device).to(device)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_loss = float("inf")
    history: List[float] = []

    for epoch in range(1, EPOCHS + 1):
        student.train()
        loss_sum = 0.0

        for x_gpr in loader:
            x_gpr = x_gpr.to(device)

            with torch.no_grad():
                t = teacher(x_gpr)
                t = F.normalize(t, dim=1)

            dummy_img = torch.zeros(
                (x_gpr.size(0), 3, STUDENT_INPUT_HW[0], STUDENT_INPUT_HW[1]),
                device=device,
            )

            s = student(dummy_img)
            loss = F.mse_loss(s, t)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())

        avg_loss = loss_sum / max(1, len(loader))
        history.append(avg_loss)
        print(f"[Epoch {epoch:03d}] distill_loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            det_state = student.det.state_dict()
            torch.save(
                {
                    "student_det_state": det_state,
                    "student_proj_state": student.proj.state_dict(),
                    # Compatibility key for legacy converters:
                    "student_state": det_state,
                    "best_loss": best_loss,
                    "embed_dim": EMBED_DIM,
                    "student_input_hw": list(STUDENT_INPUT_HW),
                },
                DISTILL_RUN_DIR / "rtdetr_distilled.pth",
            )
            print("[INFO] Saved best distilled checkpoint")

    meta = {
        "efm5_root": str(EFM5_ROOT),
        "pairs_csv": str(PAIRS_CSV),
        "gpr_feature_dir": str(GPR_FEATURE_DIR),
        "fusion_backbone": str(FUSION_BACKBONE_PATH),
        "fusion_meta": str(FUSION_META_PATH),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "embed_dim": EMBED_DIM,
        "gpr_input_channels": GPR_IN_CHANNELS,
        "gpr_image_hw": list(GPR_IMAGE_HW),
        "gpr_dct_hw": GPR_DCT_HW,
        "student_input_hw": list(STUDENT_INPUT_HW),
        "best_loss": best_loss,
        "history": history,
    }
    with (DISTILL_RUN_DIR / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[DONE] RT-DETR distillation completed")


if __name__ == "__main__":
    main()

