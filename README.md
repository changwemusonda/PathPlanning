# uncertainty-aware-planning

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

Monte Carlo Dropout extension for CALVINConv2d that turns deep differentiable planning outputs into uncertainty-aware navigation signals without changing model architecture.

## Overview

This repository presents an uncertainty-aware planning extension to CALVINConv2d for gridworld maze navigation. The central idea is simple but consequential: the model already contains dropout layers in `aa_net`, so uncertainty estimation is obtained by keeping dropout active during evaluation rather than redesigning the network. By running stochastic forward passes and analyzing variance in `aa_logit`, the planner can identify states where it is less certain about legal actions and local structure. This matters for safer navigation research because calibration-aware planning can support better failure analysis and more cautious downstream decision policies.

## Background

This work is part of a broader replication of the CVPR 2022 deep differentiable planning pipeline, where four models were trained and evaluated on 15x15 gridworld mazes: CALVINConv2d, CALVINConv3d, VIN, and GPPN. The current repository isolates Improvement 2, Uncertainty-Aware Planning via MC Dropout, as an original extension beyond the paper’s baseline setup. It builds directly on the replication environment and checkpoints while adding a reproducible uncertainty analysis and calibration workflow.

References:
- [CVPR 2022 Paper](https://arxiv.org/abs/...)
- [Main replication repo](...)

## Key Insight

> CALVINConv2d already had dropout in its action-availability subnetwork; the contribution was recognizing that uncertainty can be extracted by leaving those layers active at evaluation time and measuring variance in action-availability logits (`aa_logit`) across stochastic passes. In practical terms, this lets the planner say "I am less sure which moves are legal here" without adding new model components.

## Method

### 1) Where uncertainty is computed

The uncertainty signal is derived from `aa_net`, the action-availability CNN inside CALVINConv2d. This subnetwork is built via `make_conv_layers`, which already includes `nn.Dropout` when configured with non-zero dropout probability.

### 2) Why this is a zero-architecture-change extension

No structural modifications are made to CALVINConv2d. During standard PyTorch evaluation, dropout is disabled; MC Dropout re-enables stochastic dropout behavior at inference time and repeats forward passes. The extension therefore changes inference protocol, not architecture.

### 3) What the tensor statistics mean

For each episode, the pipeline runs N=30 stochastic forward passes and records:
- `aa_mean`: mean of action-availability logits across passes
- `aa_var`: variance of action-availability logits across passes

High `aa_var` indicates epistemic uncertainty about local action legality and maze structure, rather than uncertainty in a reward scalar.

### 4) Calibration protocol

Calibration is evaluated with Expected Calibration Error (ECE) using 10 confidence bins over 200 evaluation episodes. Target criterion is ECE < 0.05. Additional analysis compares mean uncertainty in successful vs failed episodes, where positive Delta(fail - success) indicates uncertainty rises on harder or mispredicted trajectories.

## Notebook Pipeline

The Colab notebook is organized into six primary executable cells, with one optional diagnostic cell.

| Cell | Purpose | Output |
|---|---|---|
| 1. Setup | Installs CALVIN trainer dependencies (`overboard`, `pyglet`, and compatibility requirements) in Colab. | Ready runtime environment |
| 2. Cell A | Mounts Google Drive, enumerates checkpoints, and identifies runs trained with `--dropout > 0`. | Candidate checkpoint list |
| 3. Cell 0 | Retrains CALVINConv2d with `--dropout 0.25` from scratch (about 45 minutes on Colab GPU). | Best dropout-enabled checkpoint |
| 4. Cell B | Runs MC Dropout inference with dropout enabled at eval, N=30 passes per episode, saving `aa_mean.pt` and `aa_var.pt`. | Per-episode uncertainty tensors |
| 5. Cell C | Computes success metrics, ECE (10 bins), and uncertainty-outcome analysis. | Calibration and summary metrics |
| 6. Cell D | Generates 3x6 visualization grid with path overlays and uncertainty heatmaps; saves figure to Drive. | `CALVINConv2d_uncertainty.png` |

Optional diagnostic cell:
- Repository and dataset debug dump; useful in first-run troubleshooting, usually skippable once the environment is stable.

Run order note:
- First session: run Setup -> Cell A -> (Optional Diagnostic) -> Cell 0 -> Cell B -> Cell C -> Cell D.
- Resume session (checkpoint already available): run Setup -> Cell A -> Cell B -> Cell C -> Cell D.

## Results

Training and evaluation outputs will be inserted after the latest full run completes.

| Metric | Value |
|---|---|
| Success Rate (%) | TBA |
| Mean Uncertainty (All Episodes) | TBA |
| Mean Uncertainty (Successful Episodes) | TBA |
| Mean Uncertainty (Failed Episodes) | TBA |
| Delta(fail - success) | TBA |
| ECE (10 bins) | TBA |

Expected interpretation:
- Positive Delta(fail - success) suggests higher uncertainty on failed episodes.
- Lower ECE indicates better confidence calibration (target < 0.05).

![Uncertainty Heatmaps](figures/CALVINConv2d_uncertainty.png)

## Setup & Usage

### Run in Google Colab

1. Open the notebook in Colab and ensure Google Drive access is available (checkpoints and artifacts are read/written to Drive).
2. Run Setup and Cell A to configure dependencies and discover valid checkpoints.
3. If no suitable checkpoint exists, run Cell 0 to train CALVINConv2d with `--dropout 0.25`.
4. Run Cell B to execute MC Dropout inference (N=30 stochastic passes per episode).
5. Run Cell C and Cell D for calibration analysis and visualization export.

### Smoke test path (without retraining)

If a compatible checkpoint exists, set `FORCE_DROPOUT_P=0.25` to run a quick MC Dropout smoke test before committing to full retraining.

## Environment

- Platform: Google Colab
- Python: 3.12
- CUDA: 12.x
- PyTorch: 2.x
- Key dependencies: `overboard`, `pyglet`
- Compatibility handling in notebook:
  - NumPy legacy alias patches for older dependency assumptions
  - `torch-scatter` stubbing for non-3D model paths
  - `PYOPENGL_PLATFORM=egl` for headless rendering
  - `pyglet` headless mode configuration

Note: the original CALVIN codebase targets Python 3.7 and CUDA 11.3; this notebook includes targeted patches to run reliably in modern Colab environments.

## Citation

If you use this extension, please cite both the original CVPR 2022 work and this repository.

```bibtex
@inproceedings{ishida2022realworldddp,
  title={Towards Real-World Navigation with Deep Differentiable Planners},
  author={Ishida, ... and Henriques, ...},
  booktitle={CVPR},
  year={2022}
}
```

Repository citation for this extension: placeholder (to be added).

## License

MIT (placeholder). See `LICENSE`.
