from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]  # .../MUL05
sys.path.insert(0, str(ROOT))

from pipelines import (  # noqa: E402
    stage0_ert_build,
    stage1_convert_ert,
    stageG_gpr_build,
    stage2_make_dct,
    stage3_train_fusion,
    stage3c_export_feature_heatmaps,
    stage3b_export_gpr_backbone,
    stage4_distill_rtdetr,
    stage5_make_init_ckpt,
    stage5b_train_rtdetr_experiments,
    stage6b_export_training_artifacts,
    stage6_generate_results_figures,
)


def main():
    print("\n========== FULL PIPELINE START ==========\n")

    # Stage0 is optional; keep disabled by default for batch pipeline
    # stage0_ert_build.run()

    stage1_convert_ert.run()
    # stageG_gpr_build.run()  # physical GPR generation (optional standalone)
    stage2_make_dct.run()
    stage3_train_fusion.run()
    stage3c_export_feature_heatmaps.run()
    stage3b_export_gpr_backbone.run()
    stage4_distill_rtdetr.run()
    stage5_make_init_ckpt.run()
    stage5b_train_rtdetr_experiments.run()
    stage6b_export_training_artifacts.run()
    stage6_generate_results_figures.run()

    print("\n========== FULL PIPELINE FINISHED ==========\n")


if __name__ == "__main__":
    # Optional single-stage debug:
    # "0", "1", "G", "2", "3", "4", "5", "6", or None for full run.
    DEBUG_STAGE = None

    if DEBUG_STAGE == "0":
        print("[DEBUG] Run Stage0 only")
        stage0_ert_build.run()
    elif DEBUG_STAGE == "1":
        print("[DEBUG] Run Stage1 only")
        stage1_convert_ert.run()
    elif DEBUG_STAGE == "G":
        print("[DEBUG] Run StageG only")
        stageG_gpr_build.run()
    elif DEBUG_STAGE == "2":
        print("[DEBUG] Run Stage2 only")
        stage2_make_dct.run()
    elif DEBUG_STAGE == "3":
        print("[DEBUG] Run Stage3 + 3c + 3b only")
        stage3_train_fusion.run()
        stage3c_export_feature_heatmaps.run()
        stage3b_export_gpr_backbone.run()
    elif DEBUG_STAGE == "4":
        print("[DEBUG] Run Stage4 only")
        stage4_distill_rtdetr.run()
    elif DEBUG_STAGE == "5":
        print("[DEBUG] Run Stage5 only")
        stage5_make_init_ckpt.run()
    elif DEBUG_STAGE == "5b":
        print("[DEBUG] Run Stage5b only")
        stage5b_train_rtdetr_experiments.run()
    elif DEBUG_STAGE == "6b":
        print("[DEBUG] Run Stage6b only")
        stage6b_export_training_artifacts.run()
    elif DEBUG_STAGE == "6":
        print("[DEBUG] Run Stage6 only")
        stage6b_export_training_artifacts.run()
        stage6_generate_results_figures.run()
    else:
        main()
