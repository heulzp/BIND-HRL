# BIND: Bi-level Intrinsic Modulation for Hierarchical Reinforcement Learning

This repository provides the official implementation of **BIND (Bi-level Intrinsic Modulation for Stable Hierarchical Reinforcement Learning under Non-Stationary Tasks)**.

BIND is a modular intrinsic exploration framework designed for hierarchical reinforcement learning (HRL).  
It decomposes intrinsic exploration signals into task-level novelty and subgoal-level novelty, and regulates their interaction through a task-conditioned gating mechanism, enabling more stable high-level decision-making under non-stationary task distributions.

---

## 📌 Project Status

This repository currently contains the **research code used in our experimental study**, including:

- Hierarchical training pipeline
- Task Novelty Estimation (TNE)
- Subgoal Novelty Estimation (SNE)
- Task-conditioned gating and intrinsic reward modulation
- UAV navigation and DoorKey-style environments

> **Note:**  
> The codebase is provided in its original research form.  
> We are actively cleaning, refactoring, and modularizing the implementation for improved readability and usability.  
> Additional documentation, scripts, and usage examples will be released after paper acceptance.

---

## 📁 Code Structure (Current)

The current implementation integrates multiple algorithmic variants within the training scripts for controlled ablation and comparison.  
Key files include:

- `train_highlevelUAV.py`: High-level HRL training with BIND and baseline intrinsic methods
- `env.py`, `UAV.py`: Environment definitions
- `model.py`: Policy and value networks
- `replay_buffer.py`: Experience replay utilities
- `options_runner.py`: Option execution and hierarchical interaction

---

## ⚙️ Environment and Dependencies

All methods are implemented under a unified software and hardware environment to ensure fair comparison and reproducibility.

Experiments are conducted with the following configuration:

- **Operating System:** Ubuntu 20.04
- **Python:** 3.8
- **Deep Learning Framework:** PyTorch 1.11.0
- **CUDA:** 11.3
- **GPU:** NVIDIA GeForce RTX 3080

Each method is trained with **ten random seeds**, and all reported results are averaged across seeds.

Except for the high-level exploration and intrinsic motivation mechanisms, all hierarchical methods share the same HRL framework, including:
- option execution logic,
- state and action representations,
- training schedules,
- logging and evaluation protocols.

### Required Python Packages

The core implementation relies on commonly used scientific and reinforcement learning libraries, including:

- `numpy`
- `scipy`
- `matplotlib`
- `torch`
- `gym`
- `opencv-python` (for visualization, if enabled)
- `tqdm`

> **Note:**  
> The current repository reflects the original research code used for experiments.  
> A fully packaged `requirements.txt` and environment setup script will be provided in a future update after paper acceptance.



## 🔬 Reproducibility

All experiments in the paper are conducted using this codebase under a unified HRL framework.  
Differences between methods are strictly limited to high-level exploration and intrinsic motivation mechanisms, while low-level controllers, environments, and training budgets remain identical.

Random seeds and evaluation protocols follow the settings described in the paper.

---

## 📄 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{bind2026,
  title   = {BIND: Bi-level Intrinsic Modulation for Stable Hierarchical Reinforcement Learning under Non-Stationary Tasks},
  author  = {Li, Zepei and Mo, Hongwei},
  year    = {2026}
}
