# console2_to_lwf.py
import numpy as np
import json
from pathlib import Path
import shutil


def convert_console2_result(result_path: Path, out_path: Path):
    """
    将 Console2 的 result_xxx.dat
    转换为 LWF_EFM_MODEL 格式（用于后续 DCT）
    """
    result_path = Path(result_path)
    lines = result_path.read_text().strip().splitlines()

    xs = []
    ps = []

    # 跳过表头行（NO X AM/4 p）
    for line in lines[1:]:
        cols = line.split()
        if len(cols) < 4:
            continue
        _, x, _, p = cols
        xs.append(float(x))
        ps.append(float(p))

    xs = np.array(xs)
    ps = np.array(ps)

    nx = len(xs)
    nz = 1
    dx = xs[1] - xs[0] if nx > 1 else 1.0
    dz = 1.0

    rho = ps.reshape(1, nx)  # (1, nx)

    header = {
        "source": "Console2",
        "original_file": str(result_path),
        "nx": nx,
        "nz": nz,
        "dx": dx,
        "dz": dz,
        "extent": [float(xs.min()), float(xs.max()), 0.0, 1.0],
        "comment": "仅 p 值映射，未生成伪剖面，用于 DCT"
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("LWF_EFM_MODEL\n")
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        f.write(f"{nx} {nz}\n")
        f.write(f"{dx:.6f} {dz:.6f}\n")
        f.write("0.0 0.0\n")
        for i in range(nx):
            f.write(f"{i} 0 {xs[i]:.4f} 0.0 {rho[0, i]:.6f}\n")

    print(f"[OK] 已生成 LWF ERT 数据: {out_path}")


# ============================================================
# 【新增】ERT 二维伪剖面 .npy 汇总（主线）
# ============================================================
def collect_ert_npy(npy_root: Path, out_root: Path):
    """
    将已有的二维 ERT 伪剖面 .npy
    统一复制到 data/ert/ert_npy/
    """
    npy_root = Path(npy_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(npy_root.glob("*.npy"))
    if not files:
        print(f"[WARN] 未找到任何 ERT npy: {npy_root}")
        return

    for p in files:
        dst = out_root / p.name
        shutil.copy(p, dst)

    print(f"[OK] 已汇总 {len(files)} 个 ERT 二维 npy → {out_root}")

# ======================
# 运行入口
# ======================
if __name__ == "__main__":

    # 项目根目录：LWF EFM4
    ROOT = Path(__file__).resolve().parents[1]

    # ===== 输入：生成区 =====
    summary_dir = ROOT / "BatchModels" / "summary_result"

    # ===== 输出：训练数据区 =====
    ert_lwf_dir = ROOT / "data" / "ert" / "ert_lwf"
    ert_meta_dir = ROOT / "data" / "ert" / "ert_meta"
    
    # =====【新增】 主线输出（二维 npy）=====
    #ert_npy_src = ROOT / "BatchModels" / "summary_result" / "ert_npy"
    ert_npy_src = ROOT / "BatchModels" / "summary_result" #【修改路径】
    ert_npy_dst = ROOT / "data" / "ert" / "ert_npy"
    
    ert_lwf_dir.mkdir(parents=True, exist_ok=True)
    ert_meta_dir.mkdir(parents=True, exist_ok=True)

    if not summary_dir.exists():
        raise RuntimeError(f"[ERR] 找不到 summary_result: {summary_dir}")

    for dat in sorted(summary_dir.glob("result_*.dat")):
        if dat.name.endswith("_lwf.dat"):
            continue

        stem = dat.stem.replace("result_", "")
        out_lwf = ert_lwf_dir / f"{stem}_lwf.dat"

        print(f"[INFO] 转换 {dat.name} → {out_lwf.relative_to(ROOT)}")
        convert_console2_result(dat, out_lwf)

        # ===== 同步 meta（如果存在 params_xxx.json）=====
        param_json = summary_dir / f"params_{stem}.json"
        if param_json.exists():
            shutil.copy(param_json, ert_meta_dir / f"{stem}.json")

    print("[DONE] 所有 Console2 结果已导出到 data/ert/（ERT_LWF + META）")
    
    # ============================================================
    # 【新增】ERT 二维伪剖面 npy 汇总（DCT / 融合主线）
    # ============================================================
    if ert_npy_src.exists():
        ert_npy_dst.mkdir(parents=True, exist_ok=True)

        npy_files = sorted(ert_npy_src.glob("*.npy"))
        for p in npy_files:
            dst = ert_npy_dst / p.name
            shutil.copy(p, dst)

        print(f"[OK] 已汇总 {len(npy_files)} 个 ERT 二维 npy → {ert_npy_dst}")
    else:
        print(f"[WARN] 未找到 ERT npy 目录，跳过：{ert_npy_src}")
