# FD-CMA

FD-CMA is a frequency-domain cross-modal alignment framework for GPR--ERT dam defect detection with decoupled single-modal deployment.

## Overview

FD-CMA uses ERT-derived low-frequency structural information as a transferable prior for GPR representation learning. During multimodal training, paired GPR and ERT samples are transformed into a unified frequency-domain representation using 2D-DCT. A sample-level physical confidence weight is used to modulate cross-modal alignment. After training, the GPR encoder is exported and used to initialize a downstream GPR-only detector.

## Repository contents

```text
FD-CMA/
├── EFM5/
│   ├── fusion/        # confidence-weighted cross-modal alignment and encoder export
│   ├── scripts/       # DCT feature construction and pair-list utilities
│   └── core_old/      # ERT model-generation utilities retained for reproducibility
├── gpr_generationV2/  # GPR forward-data generation and preprocessing utilities
├── pipelines/         # staged workflow scripts
├── tools/             # inspection and plotting utilities
├── ultralytics4/
│   └── distill/       # downstream detector initialization/distillation utilities
├── configs/           # example configuration files
├── examples/          # minimal example pair-list format
└── docs/              # workflow notes
```

## Installation

The code is implemented in Python. A typical environment can be prepared with:

```bash
pip install -r requirements.txt
```

GPU acceleration is recommended for detector training. The exact CUDA/PyTorch version should be selected according to the local hardware and driver environment.

## Minimal workflow

```bash
# 1. Prepare or generate paired GPR--ERT samples.
python pipelines/stage0_ert_build.py
python pipelines/stageG_gpr_build.py

# 2. Construct DCT features and sample pairs.
python pipelines/stage1_convert_ert.py
python pipelines/stage2_make_dct.py

# 3. Train FD-CMA and export the GPR encoder.
python pipelines/stage3_train_fusion.py
python pipelines/stage3b_export_gpr_backbone.py

# 4. Initialize and train the downstream GPR-only detector.
python pipelines/stage4_distill_rtdetr.py
python pipelines/stage5_make_init_ckpt.py
python pipelines/stage5b_train_rtdetr_experiments.py
```

The commands above show the intended workflow. Users should adjust paths in the configuration files or scripts before execution.

## Data

The full synthetic dataset and measured field profiles are not included in this repository because of file-size and field-data restrictions. Example configuration files and pair-list templates are provided to describe the expected input format. Data used in the associated manuscript are available from the corresponding author upon reasonable request, subject to applicable data-use restrictions.

## Citation

If you use this code, please cite the associated manuscript:

FD-CMA: A frequency-domain cross-modal alignment framework for GPR--ERT dam defect detection with decoupled single-modal deployment.

## License

This repository follows the license specified in the GitHub repository. Third-party packages used by this code are subject to their own licenses.
