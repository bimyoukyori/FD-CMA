# Cleaned repository manifest

This GitHub-ready package includes source scripts for FD-CMA feature construction, confidence-weighted cross-modal alignment, GPR-encoder export, and downstream detector initialization.

Excluded from this package:

- raw and generated datasets;
- `.npy`, `.npz`, `.mat`, `.dat`, `.bln`, and other large intermediate data files;
- model checkpoints and trained weights (`.pt`, `.pth`, `.onnx`, `.engine`);
- logs, outputs, runs, and temporary artifacts;
- local hardware/path configuration files;
- cache files and bytecode.

These exclusions are intentional because the manuscript data are either large, field-restricted, or generated as intermediate products. The repository is intended to provide source code and workflow scripts rather than a full data dump.
