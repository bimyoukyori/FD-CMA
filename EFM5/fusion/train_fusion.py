#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_fusion.py（CNN-GPR 主线｜最终可跑版）

功能说明
--------
训练 ERT–GPR 双分支融合网络（FusionNet）：
- ERT 分支：MLP（输入 1D 特征：features/ert 或 features/ert_blockdct_pos）
- GPR 分支：CNN（输入 2D DCT 频谱：需 reshape 为 (1, H, W)）

训练目标
--------
- 主损失：对称 InfoNCE（支持样本级 confidence 权重）
- 辅助损失：分类交叉熵（若 pairs.csv 中存在 label 且类别数>1）

工程定位
--------
- Stage3：CLD-Fusion 跨模态对齐
- 输出：
  - fusion_best.pth        ：融合网络最优权重
  - fused_backbone.pth     ：仅 GPR CNN 分支权重（用于后续 RT-DETR 蒸馏）
  - meta.json              ：本次训练配置与统计信息（便于论文复现）

重要约束（与你当前主线对齐）
--------
- 不使用命令行参数，所有开关写死在脚本顶部
- 以脚本位置反推工程 ROOT，彻底避免 cwd 引发的路径错误
- GPR 特征按 CNN 语义读取：允许 .npy 为 (1024,) 或 (H,W)，最终统一为 (1,H,W)
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset


# ============================================================
# 工程根目录（唯一锚点）
# ============================================================

ROOT = Path(__file__).resolve().parents[1]  # .../EFM5


# ============================================================
# 实验配置区（唯一权威来源）
# ============================================================

# ----------------------
# 消融与策略开关
# ----------------------

# False：主线（features/ert）
# True ：ERT 分块 DCT + 位置编码消融（features/ert_blockdct_pos）
USE_ERT_BLOCK_DCT = False

# 是否启用样本级 confidence 加权（强烈建议 True）
USE_CONFIDENCE_WEIGHT = True

# 是否分层切分（按 label 分层；若 label 缺失或类别不足，将自动回退普通切分）
USE_STRATIFIED_SPLIT = True

# 验证集比例
VAL_RATIO = 0.2

# ----------------------
# GPR 2D 频谱尺寸（CNN-GPR 主线关键参数）
# ----------------------
# 你的 GPR 特征若来自 2D-DCT 低频块 32×32 且 flatten 为 1024，此处应为 32
# 若你后续改成 64×64，请同步改为 64
GPR_DCT_HW = 32

# ----------------------
# 路径配置（全部基于 ROOT）
# ----------------------

PAIRS_CSV = ROOT / "data" / "pairs.csv"

ERT_FEATURE_DIR = (
    ROOT / "features" / "ert_blockdct_pos"
    if USE_ERT_BLOCK_DCT
    else ROOT / "features" / "ert"
)

GPR_FEATURE_DIR = ROOT / "features" / "gpr"

RUN_NAME = os.getenv("FUSION_RUN_NAME", "exp_fusion_main")

# 是否启用样本级 confidence 加权（支持环境变量覆盖）
USE_CONFIDENCE_WEIGHT = os.getenv("FUSION_USE_CONF", "1") != "0"

SAVE_DIR = ROOT / "fusion" / "runs" / RUN_NAME

# SAVE_DIR = ROOT / "fusion" / "runs" / "exp_fusion_main"

# ----------------------
# 训练超参数
# ----------------------

EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
TEMPERATURE = 0.07
ALPHA_CE = 1.0
SEED = 42

# DataLoader
NUM_WORKERS = 0  # Windows 下设 0 最稳
PIN_MEMORY = True


# ============================================================
# 随机种子固定（保证可复现）
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 对称 InfoNCE（支持样本级权重）
# ============================================================

def info_nce_weighted(
    e1: torch.Tensor,
    e2: torch.Tensor,
    weight: Optional[torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    """
    e1, e2: (B, D)，已 L2 normalize
    weight: (B,) 或 None
    """
    bsz = e1.shape[0]
    sim = (e1 @ e2.T) / temperature
    target = torch.arange(bsz, device=e1.device)

    loss_a = F.cross_entropy(sim, target, reduction="none")
    loss_b = F.cross_entropy(sim.T, target, reduction="none")

    if weight is None:
        return 0.5 * (loss_a.mean() + loss_b.mean())

    weight = weight.float().clamp(min=0.0)
    denom = weight.sum().clamp(min=1e-12)

    loss_a = (loss_a * weight).sum() / denom
    loss_b = (loss_b * weight).sum() / denom
    return 0.5 * (loss_a + loss_b)


# ============================================================
# 数据集：pairs.csv → (ERT 1D, GPR 2D, label, confidence)
# ============================================================

class FusionPairDataset(Dataset):
    """
    pairs.csv 职责冻结：仅提供语义配对（id / label / confidence）
    特征读取由代码统一控制：
      - ERT: features/ert/{id}.npy 或 features/ert_blockdct_pos/{id}.npy  （1D）
      - GPR: features/gpr/{id}.npy（允许 1D flatten 或 2D 矩阵）
    """

    def __init__(self, csv_path: Path):
        if not csv_path.exists():
            raise FileNotFoundError(f"pairs.csv not found: {csv_path}")

        if not ERT_FEATURE_DIR.exists():
            raise FileNotFoundError(f"ERT feature dir not found: {ERT_FEATURE_DIR}")

        if not GPR_FEATURE_DIR.exists():
            raise FileNotFoundError(f"GPR feature dir not found: {GPR_FEATURE_DIR}")

        # 读取 CSV 原始行
        raw_rows: List[Dict[str, str]] = []
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                raw_rows.append(r)

        if not raw_rows:
            raise RuntimeError("pairs.csv is empty")

        # 过滤掉缺失特征的样本（避免训练中途随机爆炸）
        self.rows: List[Dict[str, str]] = []
        missing_ert = 0
        missing_gpr = 0

        for r in raw_rows:
            sid = (r.get("id") or "").strip()
            if not sid:
                continue

            ep = ERT_FEATURE_DIR / f"{sid}.npy"
            gp = GPR_FEATURE_DIR / f"{sid}.npy"

            ok = True
            if not ep.exists():
                missing_ert += 1
                ok = False
            if not gp.exists():
                missing_gpr += 1
                ok = False

            if ok:
                self.rows.append(r)

        if not self.rows:
            raise RuntimeError(
                "No valid pairs after filtering. "
                f"missing_ert={missing_ert}, missing_gpr={missing_gpr}"
            )

        print(f"[INFO] pairs.csv rows={len(raw_rows)} valid_pairs={len(self.rows)} "
              f"missing_ert={missing_ert} missing_gpr={missing_gpr}")

        # label 字段（可选）
        # 若不存在 label 或全为空，则 class_names=unknown
        label_set = sorted({(r.get("label") or "").strip() for r in self.rows if (r.get("label") or "").strip()})
        self.class_names = label_set if label_set else ["unknown"]
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        # 推断 ERT 维度（1D）
        self.ert_dim = None
        for r in self.rows:
            sid = r["id"].strip()
            ep = ERT_FEATURE_DIR / f"{sid}.npy"
            arr = np.load(ep)
            if arr.ndim != 1:
                # 允许 ERT 特征是 2D，但当前主线应为 1D；此处直接 flatten 以保证可跑
                arr = arr.reshape(-1)
            self.ert_dim = int(arr.shape[0])
            break

        if self.ert_dim is None:
            raise RuntimeError("Failed to infer ERT feature dim")

        # 记录 GPR 目标 HW（CNN 输入）
        self.gpr_hw = int(GPR_DCT_HW)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_gpr_as_2d(self, sid: str) -> np.ndarray:
        """
        将 features/gpr/{id}.npy 统一加载为 (1, H, W)

        允许两种输入：
        - 1D: (H*W,)  → reshape
        - 2D: (H, W)  → 加 channel 维
        """
        gp = GPR_FEATURE_DIR / f"{sid}.npy"
        arr = np.load(gp).astype("float32")

        H = W = self.gpr_hw

        if arr.ndim == 1:
            if arr.size != H * W:
                raise RuntimeError(f"GPR 1D feature size mismatch: id={sid} size={arr.size} expected={H*W}")
            arr2d = arr.reshape(H, W)
            return arr2d[None, :, :]  # (1,H,W)

        if arr.ndim == 2:
            if arr.shape != (H, W):
                raise RuntimeError(f"GPR 2D feature shape mismatch: id={sid} shape={arr.shape} expected={(H,W)}")
            return arr[None, :, :]  # (1,H,W)

        if arr.ndim == 3:
            # 若已经是 (1,H,W) 则直接返回；否则做严格检查
            if arr.shape == (1, H, W):
                return arr
            raise RuntimeError(f"GPR 3D feature shape mismatch: id={sid} shape={arr.shape} expected={(1,H,W)}")

        raise RuntimeError(f"Unsupported GPR feature ndim: id={sid} ndim={arr.ndim}")

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        sid = (r.get("id") or "").strip()
        if not sid:
            raise RuntimeError("Empty id in pairs.csv row")

        # ERT（1D）
        ert_path = ERT_FEATURE_DIR / f"{sid}.npy"
        ert_feat = np.load(ert_path).astype("float32")
        if ert_feat.ndim != 1:
            ert_feat = ert_feat.reshape(-1)

        # GPR（2D CNN 输入）
        gpr_feat = self._load_gpr_as_2d(sid)

        # label
        label_name = (r.get("label") or "").strip()
        label = self.class_to_idx.get(label_name, 0)

        # confidence（可选）
        if USE_CONFIDENCE_WEIGHT:
            try:
                confidence = float(r.get("confidence", 1.0))
            except Exception:
                confidence = 1.0
        else:
            confidence = 1.0

        return (
            torch.from_numpy(ert_feat),                    # (D_ert,)
            torch.from_numpy(gpr_feat),                    # (1,H,W)
            torch.tensor(label, dtype=torch.long),
            torch.tensor(confidence, dtype=torch.float32),
        )


# ============================================================
# 训练/验证集切分
# ============================================================

def stratified_split(indices: List[int], labels: List[int], val_ratio: float) -> Tuple[List[int], List[int]]:
    """
    按 label 分层切分。若某类样本过少，会尽量保证 train/val 都有样本。
    """
    rng = random.Random(SEED)
    cls_to_ids: Dict[int, List[int]] = defaultdict(list)
    for idx, y in zip(indices, labels):
        cls_to_ids[y].append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []

    for y, ids in cls_to_ids.items():
        rng.shuffle(ids)
        if len(ids) == 1:
            # 样本极少时直接放入 train，避免 val 为空导致统计崩溃
            train_idx.extend(ids)
            continue

        n_val = max(1, int(len(ids) * val_ratio))
        # 避免某一类 train 为空
        if n_val >= len(ids):
            n_val = len(ids) - 1

        val_idx.extend(ids[:n_val])
        train_idx.extend(ids[n_val:])

    return sorted(train_idx), sorted(val_idx)


# ============================================================
# 主训练入口
# ============================================================

def main():
    set_seed(SEED)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # 延迟导入，避免路径/依赖问题
    from fusion_net import FusionNet

    dataset = FusionPairDataset(PAIRS_CSV)

    # 收集 labels（用于分层切分；若无 label，则全为 0）
    all_indices = list(range(len(dataset)))
    all_labels = [int(dataset[i][2].item()) for i in all_indices]

    # 切分策略：优先分层；若类别数不足或 val 过小，自动回退普通切分
    if USE_STRATIFIED_SPLIT and len(set(all_labels)) > 1:
        train_idx, val_idx = stratified_split(all_indices, all_labels, VAL_RATIO)
        if len(val_idx) == 0:
            print("[WARN] stratified split produced empty val set, fallback to normal split")
            n_val = max(1, int(len(dataset) * VAL_RATIO))
            train_idx = list(range(len(dataset) - n_val))
            val_idx = list(range(len(dataset) - n_val, len(dataset)))
    else:
        n_val = max(1, int(len(dataset) * VAL_RATIO))
        train_idx = list(range(len(dataset) - n_val))
        val_idx = list(range(len(dataset) - n_val, len(dataset)))

    print(f"[INFO] split: train={len(train_idx)} val={len(val_idx)} classes={len(set(all_labels))}")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    # CNN-GPR 主线：FusionNet 不接受 gpr_dim
    model = FusionNet(
        ert_dim=dataset.ert_dim,
        num_classes=len(dataset.class_names),
        emb_dim=256,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # 分类损失：仅当类别数>1 时有意义；否则置 0
    use_ce = len(dataset.class_names) > 1

    best_val_nce = float("inf")
    history = {"train_loss": [], "val_nce": []}

    for epoch in range(1, EPOCHS + 1):
        # -------------------------
        # 训练
        # -------------------------
        model.train()
        loss_sum = 0.0

        for x1, x2, y, conf in train_loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            conf = conf.to(device, non_blocking=True)

            optimizer.zero_grad()
            _, logits, e1, e2 = model(x1, x2)

            loss_nce = info_nce_weighted(
                e1, e2,
                conf if USE_CONFIDENCE_WEIGHT else None,
                TEMPERATURE,
            )

            if use_ce:
                loss_ce = nn.CrossEntropyLoss()(logits, y)
            else:
                loss_ce = torch.zeros((), device=device)

            loss = loss_nce + (ALPHA_CE * loss_ce)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item())

        avg_train_loss = loss_sum / max(1, len(train_loader))

        # -------------------------
        # 验证：只统计 InfoNCE（与 best 选择一致）
        # -------------------------
        model.eval()
        val_nce_sum = 0.0
        with torch.no_grad():
            for x1, x2, _, conf in val_loader:
                x1 = x1.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)
                conf = conf.to(device, non_blocking=True)

                _, _, e1, e2 = model(x1, x2)
                val_nce_sum += float(
                    info_nce_weighted(
                        e1, e2,
                        conf if USE_CONFIDENCE_WEIGHT else None,
                        TEMPERATURE,
                    ).item()
                )

        val_nce = val_nce_sum / max(1, len(val_loader))

        history["train_loss"].append(avg_train_loss)
        history["val_nce"].append(val_nce)

        print(f"[Epoch {epoch:03d}] train_loss={avg_train_loss:.4f} val_nce={val_nce:.4f}")

        # -------------------------
        # 保存最优
        # -------------------------
        if val_nce < best_val_nce:
            best_val_nce = val_nce

            torch.save(model.state_dict(), SAVE_DIR / "fusion_best.pth")
            torch.save(model.gpr_branch.state_dict(), SAVE_DIR / "fused_backbone.pth")

            print("[INFO] Saved best fusion checkpoint")

    # -------------------------
    # 写 meta（复现用）
    # -------------------------
    meta = {
        "use_ert_block_dct": USE_ERT_BLOCK_DCT,
        "use_confidence_weight": USE_CONFIDENCE_WEIGHT,
        "use_stratified_split": USE_STRATIFIED_SPLIT,
        "val_ratio": VAL_RATIO,
        "gpr_dct_hw": GPR_DCT_HW,
        "pairs_csv": str(PAIRS_CSV),
        "ert_feature_dir": str(ERT_FEATURE_DIR),
        "gpr_feature_dir": str(GPR_FEATURE_DIR),
        "save_dir": str(SAVE_DIR),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "temperature": TEMPERATURE,
        "alpha_ce": ALPHA_CE,
        "seed": SEED,
        "best_val_nce": best_val_nce,
        "history": history,
        "class_names": list(getattr(dataset, "class_names", ["unknown"])),
    }

    with (SAVE_DIR / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[DONE] Fusion training completed")


if __name__ == "__main__":
    main()
