# CALVIN Final Project: Uncertainty Heatmap Extension

This project replicates the CALVIN gridworld navigation experiments and adds a red uncertainty heatmap overlay.

## Main additions
- Action-availability uncertainty using Bernoulli entropy over predicted action availability.
- Planner uncertainty using entropy over CALVIN Q-values.
- Red uncertainty overlay on 2D maze maps with obstacle masking.
- Trajectory visualization showing start, goal, path, and uncertainty intensity.

## Key interpretation
Darker red regions indicate higher action-availability uncertainty and lower planner confidence.

## Important files
- `core/utils/uncertainty_utils.py`
- `core/models/calvin/calvin_base.py`
- `core/utils/logger.py`
- `results/figures/`
- `data/gridworld/`
