# An Entropy-Based Uncertainty Extension of CALVIN for Interpretable Gridworld Navigation

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

**Authors:** Changwe B. Musonda, Pyson Aung, Thomas McDonnell  
**Institution:** California State Polytechnic University, Pomona  
**Course:** ECE 4990 - Final Project

An entropy-based uncertainty extension of CALVINConv2d that provides interpretable per-cell confidence signals for gridworld navigation without modifying the model architecture or training process.

## Project Overview

This project extends the CALVIN (Collision Avoidance Long-term Value Iteration Network) differentiable planner with an entropy-based uncertainty decomposition. Unlike the original CALVIN which produces only point estimates, our extension exposes two uncertainty components:

1. **Action-Availability Uncertainty** – Bernoulli entropy over predicted action legality
2. **Planner Uncertainty** – Shannon entropy over Q-value policy distribution

These are combined into a per-cell uncertainty heatmap that reveals where the planner is most uncertain about navigation decisions.

## Key Contributions

- ✅ **Entropy-based uncertainty extension** of CALVINConv2d with two interpretable decomposed components
- ✅ **Non-invasive design** – No architectural changes, no new loss terms, single forward pass
- ✅ **Trajectory validation routine** – Disambiguates rendering artifacts from true planner failures  
- ✅ **Modern environment support** – Ported to Python 3.12, CUDA 12.x, PyTorch 2.x
- ✅ **Comprehensive evaluation** – Honest assessment with 88% success rate on 100 validation episodes

## Main Results

| Metric | Value |
|--------|-------|
| **Success Rate** | 88% (100 validation episodes) |
| **Average Path Length** | 46.71 steps |
| **Mean Uncertainty (Ū)** | 0.376 |
| **Average Final Reward** | 0.870 |
| **SPL** | 0.466 |

*Note: Our results are ∼12 points below the original CALVIN paper's 99.7% – see [Discussion](#discussion) section for analysis.*

## Repository Structure

```
ECE4990/
├── README.md                    # This file
├── PROJECT_README.md            # Project notes
├── .gitignore
├── core/
│   ├── agent.py                 # Agent interaction logic
│   ├── agent_trainer.py         # Training framework
│   ├── dataset.py               # Dataset management
│   ├── env.py                   # Environment base
│   ├── experiences.py           # Experience collection
│   ├── handler.py               # Data utilities
│   ├── mdp/                     # MDP utilities
│   ├── models/                  # Neural network models
│   │   ├── calvin/              # CALVIN implementation
│   │   └── detector/            # Detection models
│   ├── domains/                 # Environment implementations
│   │   ├── gridworld/           # Gridworld navigation (primary)
│   │   ├── avd/                 # Active Vision Dataset
│   │   └── miniworld/           # MiniWorld environment
│   └── utils/                   # [TO BE ADDED]
│       └── uncertainty_utils.py # [TO BE ADDED]
└── data/                        # [TO BE CREATED]
```

## Technical Background

### CALVIN Architecture

CALVIN reformulates value iteration to explicitly model action availability and termination:

**Transition Probability:**
$$P(s' | s, a) = \begin{cases} 
1 - \hat{A}(s, a), & s' = F \\
\hat{A}(s, a) \hat{P}(s' - s | a), & s' \neq F
\end{cases}$$

**Action-Value Update:**
$$Q(s, a) = R(s, a) + \gamma \hat{A}(s, a) I_{a \neq D} \sum_{s'} \hat{P}(s' - s | a) V(s')$$

### Uncertainty Decomposition

#### Action-Availability Uncertainty
Bernoulli entropy over predicted action legality probabilities:
$$H_{aa}(s, a) = -\hat{A}(s, a) \log \hat{A}(s, a) - (1-\hat{A}(s, a)) \log(1-\hat{A}(s, a))$$

Aggregated: $H_{aa}(s) = \frac{1}{|A|} \sum_a H_{aa}(s, a)$

#### Planner Uncertainty
Shannon entropy over Q-value policy distribution:
$$\pi(a|s) = \text{softmax}_a(Q(s, a))$$
$$H_\pi(s) = -\sum_a \pi(a|s) \log \pi(a|s)$$

#### Combined Uncertainty Heatmap
$$U(s) = 0.5 \cdot H_{aa}(s) + 0.5 \cdot H_\pi(s)$$

### Design Principles

- **Non-invasive:** Uncertainty computed from existing predictions; uncertainty_penalty = 0.0 during training
- **Deterministic:** Single forward pass (unlike MC Dropout which requires multiple passes)
- **Interpretable:** Equal-weight combination allows human interpretation of uncertainty sources

## Qualitative Observations

The uncertainty heatmap reveals three interpretable patterns:

1. **Maze Junctions** – Darker red where multiple candidate actions yield similar Q-values
2. **Long Corridors** – White (low uncertainty) as planner commits strongly to a single direction
3. **Wall Boundaries** – Elevated uncertainty reflecting genuine action-legality ambiguity

## Setup & Installation

### Prerequisites

```bash
# Required
- Python 3.12+
- CUDA 12.x (for GPU)
- PyTorch 2.x
```

### Create Conda Environment

```bash
conda create -n calvin python=3.12
conda activate calvin

# (Recommended for some dependencies)
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
```

### Install Dependencies

```bash
# PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Python packages
pip install numpy matplotlib einops torch-scatter numba tensorboard
```

### Optional: MiniWorld Environment

```bash
mkdir -p third_party
cd third_party
git clone https://github.com/maximecb/gym-miniworld.git
cd gym-miniworld
pip install -e .
cd ../../
```

### Optional: Active Vision Dataset (AVD)

Download from [AVD Website](https://www.cs.unc.edu/~ammirato/active_vision_dataset_website/index.html), place under `./data/avd/src` with `train.txt` and `val.txt` manifests.

## Usage

### Generate Training Dataset

```bash
python core/domains/gridworld/dataset.py \
    --domain gridworld \
    --map-type maze \
    --grid-size 15 \
    --num-episodes 4000 \
    --output data/gridworld/
```

### Train Model

```bash
python core/domains/gridworld/trainer.py \
    --model calvin \
    --depth 60 \
    --hidden-width 150 \
    --learning-rate 0.005 \
    --epochs 30 \
    --uncertainty-penalty 0.0 \
    --checkpoint-dir checkpoints/
```

### Evaluate & Visualize

```bash
python core/scripts/eval.py \
    --model-checkpoint checkpoints/calvin_best.pt \
    --dataset data/gridworld/validation \
    --num-episodes 100 \
    --output results/

python core/scripts/visualise.py \
    --rollouts results/rollouts.pkl \
    --output results/figures/
```

### Trajectory Validation

```bash
python core/utils/trajectory_validator.py \
    --episodes results/rollouts.pkl \
    --gridmap data/gridworld/validation_gridmap.pkl
```

## Experiments

### Dataset
- **Name:** MazeMap_15x15_vr_2_4000_15_500
- **Training:** 4000 demonstrations
- **Validation:** ~1000 episodes
- **Environment:** 15×15 gridworld mazes (Wilson's algorithm)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Value iteration depth | 60 |
| Hidden width | 150 |
| Discount factor γ | 0.99 |
| Training discount | 0.25 |
| Optimizer | Adam |
| Learning rate | 0.005 |
| Gradient clipping | 0.1 |
| Uncertainty penalty | 0.0 |
| Epochs | 30 |

### Evaluation Metrics

- **Success Rate** – Fraction of episodes reaching goal
- **Path Length** – Mean, median, max trajectory steps
- **Average Uncertainty (Ū)** – Mean per-episode uncertainty
- **SPL** – Success weighted by Path Length
- **Diagnostics** – Invalid actions, repeated states, false completions

## Discussion

### Success Rate Gap (88% vs. 99.7%)

Our replication achieves 88% compared to the original CALVIN paper's 99.7% ± 0.5%. Likely contributing factors:

- **Single seed** vs. paper's three-seed average
- **Training discount schedule** (0.25) may differ from original
- **30 epochs** vs. convergence-based criterion
- **Default Adam settings** may differ

**Future Work:** Controlled ablations and multi-seed re-runs with original hyperparameters would isolate the gap.

### Trajectory Validation Finding

Rendered trajectories appeared to cross walls. Our validation confirmed that under `(row, col)` convention, all 100 episodes had **zero wall-cell hits and zero invalid actions**. Visual wall-crossings are **rendering artifacts** from drawing continuous lines between discrete cells—underlying trajectories are valid.

### Limitations

1. **Deterministic uncertainty** – Captures predictive uncertainty; confidently wrong predictions show low uncertainty
2. **Fixed weights** – 0.5/0.5 is a design choice, not learned
3. **Single dataset** – Gridworld only; multi-domain validation recommended
4. **Epistemic uncertainty** – Future work should combine with Bayesian methods (MC Dropout)

## References

[1] Ishida, S., Henriques, J. F. (2022). "Towards real-world navigation with deep differentiable planners." *CVPR 2022*.

[2] Tamar, A., et al. (2016). "Value Iteration Networks." *NeurIPS 2016*.

[3] Lee, L., et al. (2018). "Gated Path Planning Networks." *arXiv:1806.06408*.

[4] Gal, Y., Ghahramani, Z. (2016). "Dropout as a Bayesian approximation." *ICML 2016*.

[5] Lakshminarayanan, B., et al. (2017). "Simple and scalable predictive uncertainty estimation." *NeurIPS 2017*.

[6] Guo, C., et al. (2017). "On calibration of modern neural networks." *ICML 2017*.

## Contributions

- **Changwe B. Musonda** – Environment setup, codebase porting, trajectory validation, manuscript
- **Pyson Aung** – Uncertainty extension, model training, 100-episode evaluation  
- **Thomas McDonnell** – Experimental setup, dataset handling, figure preparation

## Citation

```bibtex
@inproceedings{musonda2024calvin_uncertainty,
    title={An Entropy-Based Uncertainty Extension of CALVIN for Interpretable Gridworld Navigation},
    author={Musonda, Changwe B. and Aung, Pyson and McDonnell, Thomas},
    booktitle={ECE 4990 Final Project, Cal Poly Pomona},
    year={2024}
}
```

Original CALVIN:
```bibtex
@inproceedings{ishida2022calvin,
    title={Towards real-world navigation with deep differentiable planners},
    author={Ishida, Shu and Henriques, João F.},
    booktitle={CVPR},
    year={2022}
}
```

## Resources

- 📄 **Full Project Paper** – Included in repository
- 🔗 **Original CALVIN** – https://github.com/shuishida/calvin
- 📊 **CALVIN Paper (CVPR 2022)** – https://arxiv.org/abs/2108.05713
- 🎯 **Reference Implementation** – https://github.com/tkmcdonnell/calvin-final-project.git

## License

MIT – See LICENSE file
