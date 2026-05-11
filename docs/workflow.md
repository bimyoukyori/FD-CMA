# FD-CMA workflow

The repository is organized around the manuscript workflow:

1. Generate or prepare paired GPR--ERT samples.
2. Convert GPR B-scans and ERT apparent-resistivity pseudosections to a common 2D-DCT feature interface.
3. Compute sample-level physical confidence weights from effective depth and resistivity contrast.
4. Train the confidence-weighted cross-modal alignment model.
5. Export the trained GPR encoder.
6. Use the exported encoder as initialization for a downstream GPR-only detector.

The full synthetic and field datasets are not included because of file-size and field-data restrictions. The scripts provide the computational workflow and can be adapted to user-provided data following the expected feature and pair-list formats.
