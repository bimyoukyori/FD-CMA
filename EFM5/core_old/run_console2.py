"""
run_console2.py
最终稳定版（批量正演 + 分批运行 + 终止标记检测 + 完整日志）
"""

import os
import subprocess
from glob import glob
import time
import stat
from pathlib import Path
import re


# ============================================================
# 工具函数
# ============================================================

def clear_result_readonly_flags(model_root):
    """清除所有 result.dat 的只读权限（防止 Fortran 覆盖失败）"""
    for f in glob(os.path.join(model_root, "model_*", "result.dat")):
        try:
            os.chmod(f, stat.S_IWRITE)
        except Exception as e:
            print(f"[WARN] 无法修改权限: {f} → {e}")


def check_file_basic(cmd_file, dat_file):
    """简单检查文件是否存在且大小不为0"""
    issues = []
    if not os.path.exists(cmd_file):
        issues.append("cmd.par 不存在")
    elif os.path.getsize(cmd_file) == 0:
        issues.append("cmd.par 是空文件")

    if not os.path.exists(dat_file):
        issues.append("Model.dat 不存在")
    elif os.path.getsize(dat_file) == 0:
        issues.append("Model.dat 是空文件")

    return issues


# ============================================================
# 主入口
# ============================================================

def run_fortran(exe_path, model_root, batch_size=300):
    """
    在模型根目录运行 Console2.exe（分批批处理模式）

    [MOD]
    - 完全去除 Press Enter 交互
    - 按 batch_size 分批运行（默认 300）
    - 通过检测 ">>>> All models completed." 判断 Fortran 正常结束
    """

    log = ""
    log += f"[INFO] Fortran 可执行文件：{exe_path}\n"
    log += f"[INFO] 模型根目录：{model_root}\n"
    log += f"[INFO] 单批模型数：{batch_size}\n"

    if not os.path.exists(exe_path):
        return f"[ERR] Console2.exe 不存在：{exe_path}"

    if not os.path.isdir(model_root):
        return f"[ERR] 模型目录不存在：{model_root}"

    project_root = os.path.dirname(os.path.normpath(model_root))
    log += f"[INFO] 推断的项目根目录：{project_root}\n"

    clear_result_readonly_flags(model_root)

    # ------------------------------------------------------------
    # 扫描模型目录
    # ------------------------------------------------------------
    model_dirs = sorted(
        [p.name for p in Path(model_root).glob("model_*") if p.is_dir()]
    )

    if not model_dirs:
        return log + "[ERR] 没有找到任何 model_XXX 目录\n"

    log += f"[INFO] 发现模型总数：{len(model_dirs)}\n"

    # ------------------------------------------------------------
    # 过滤有效模型
    # ------------------------------------------------------------
    valid_models = []
    for m in model_dirs:
        cmd_file = os.path.join(model_root, m, "cmd.par")
        dat_file = os.path.join(model_root, m, "Model.dat")
        issues = check_file_basic(cmd_file, dat_file)
        if issues:
            log += f"[WARN] {m} 文件问题：{'; '.join(issues)}，跳过\n"
        else:
            valid_models.append(m)

    if not valid_models:
        return log + "[ERR] 没有有效模型可以运行\n"

    log += f"[INFO] 有效模型数：{len(valid_models)}\n"

    # ============================================================
    # [MOD] 分批运行
    # ============================================================
    batches = [
        valid_models[i:i + batch_size]
        for i in range(0, len(valid_models), batch_size)
    ]

    log += f"[INFO] 分批数：{len(batches)}\n"

    total_success = 0

    for bi, batch in enumerate(batches, 1):
        log += "\n" + "=" * 60 + "\n"
        log += f"[RUN] 开始第 {bi}/{len(batches)} 批（{len(batch)} 个模型）\n"

        # --------------------------------------------------------
        # [MOD] 写入 model_list.txt（供 Fortran main_batch 使用）
        # --------------------------------------------------------
        list_file = os.path.join(project_root, "model_list.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for m in batch:
                f.write(m + "\n")

        log += f"[INFO] 已写入模型列表：{list_file}\n"

        # --------------------------------------------------------
        # 运行 Console2.exe（无 stdin，纯批处理）
        # --------------------------------------------------------
        proc = subprocess.Popen(
            [exe_path],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            bufsize=1
        )

        batch_finished_flag = False

        while True:
            line = proc.stdout.readline()
            if not line:
                break

            log += f"[PROC] {line}"

            # ----------------------------------------------------
            # [MOD] 检测 Fortran 终止标记
            # ----------------------------------------------------
            if ">>>> All models completed." in line:
                log += "[FINISH] Fortran 批处理已正常结束（检测到终止标记）\n"
                batch_finished_flag = True

        proc.wait()

        if not batch_finished_flag:
            log += "[WARN] 未检测到 Fortran 终止标记，请检查输出\n"

        # --------------------------------------------------------
        # 批次完成后统计 result.dat
        # --------------------------------------------------------
        success = 0
        for m in batch:
            result_file = os.path.join(model_root, m, "result.dat")
            if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
                success += 1

        log += f"[INFO] 本批完成：{success}/{len(batch)}\n"
        total_success += success

    # ============================================================
    # 总结
    # ============================================================
    log += "\n" + "=" * 60 + "\n"
    log += f"[FINISH] 所有批次完成，总成功模型数：{total_success}/{len(valid_models)}\n"

    return log
