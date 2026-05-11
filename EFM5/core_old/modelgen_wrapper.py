"""
封装旧 Console2_GUI 的 model_generator.py 中的 generate_model_files()
使其可以被新 GUI 调用。
"""

import os
from core_old import model_generator


def generate_models(
    outdir: str,
    start_id: int,
    end_id: int,
    gui_params: dict
):
    """
    新 GUI 调用入口：
    - 自动生成 model_001 ... model_NNN
    - 调用旧函数 generate_model_files()

    gui_params 示例：
    {
        "rho_background": 100,
        "Pnum": 91,
        "spacing": 2.0,
        "width": 200,
        "depth": 80,
        "anomaly_regions": [...],
        "debug_plot": False
    }
    """

    os.makedirs(outdir, exist_ok=True)

    for i in range(start_id, end_id + 1):

        folder = os.path.join(outdir, f"model_{i:03d}")
        os.makedirs(folder, exist_ok=True)

        print(f"[MODEL] 生成 {folder} ...")

        # 旧建模器核心调用
        model_generator.generate_model_files(
            path=folder,
            model_id=f"model_{i:03d}",
            rho_background=gui_params.get("rho_background", 100.0),
            Pnum=gui_params.get("Pnum", 91),
            spacing=gui_params.get("spacing", 2.0),
            width=gui_params.get("width", 200.0),
            depth=gui_params.get("depth", 80.0),
            anomaly_regions=gui_params.get("anomaly_regions", None),
            debug_plot=gui_params.get("debug_plot", False)
        )

    print("[DONE] 所有模型生成完成！")
