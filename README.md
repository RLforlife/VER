# Revisiting Prioritized Experience Replay: A Value Perspective (VER)

Official code release for the AAAI 2025 paper **Revisiting Prioritized Experience Replay: A Value Perspective**.

This repository studies **Prioritized Experience Replay (PER)** from a value-based perspective and proposes **Valuable Experience Replay (VER)**, which prioritizes transitions using both TD error and the policy probability of the taken action.

All experiments are implemented in **Python 3**, primarily as Jupyter notebooks (`.ipynb`). Atari experiments also provide a standalone script (`.py`).

---

## Overview

| Component | Description |
|-----------|-------------|
| **Problem** | Standard PER ranks samples by `\|TD error\|`, which can miss transitions that are valuable from a value / policy perspective. |
| **Method (VER)** | On top of Soft DQN, replay priorities are updated with **`π(a\|s) × TD error`** instead of `\|TD error\|` alone. |
| **Baselines** | Tabular Q-learning / Soft Q-learning, DQN / Soft DQN, and DisCor on Atari. |

---

## Repository Structure

```
VER/
├── README.md
└── code/
    ├── maze_q_learning.ipynb          # Figure 3 & 4 — tabular Q-learning (Maze)
    ├── maze_soft_q_learning.ipynb     # Figure 3 & 4 — tabular Soft Q-learning (Maze)
    ├── cartpole_dqn.ipynb             # Figure 4 — DQN (CartPole)
    ├── cartpole_soft_dqn.ipynb        # Figure 4 — Soft DQN (CartPole)
    ├── SoftDQN_VER_Seaquest.ipynb     # Figure 6 — VER on Atari Seaquest
    ├── SoftDQN_VER_Seaquest.py        # Script version of the VER Atari experiment
    ├── SoftDQN_DisCor_Seaquest.ipynb  # DisCor baseline on Atari Seaquest
    ├── atari_wrappers.py              # Atari environment wrappers
    ├── logx.py                        # Experiment logging (Spinning Up style)
    ├── run_utils.py                   # Logger / run utilities
    ├── mpi_tools.py                   # MPI helpers (optional)
    └── serialization_utils.py         # JSON serialization helpers
```

---

## Requirements

Core dependencies used across experiments:

- Python 3.7+
- [PyTorch](https://pytorch.org/)
- [OpenAI Gym](https://github.com/openai/gym) (classic control & Atari)
- NumPy, SciPy, Matplotlib, pandas
- [TensorBoardX](https://github.com/lanpa/tensorboardX)
- [OpenCV](https://opencv.org/) (`cv2`, for Atari preprocessing)
- [gym-maze](https://github.com/MattChanTK/gym-maze) (maze experiments)
- [OpenAI Baselines](https://github.com/openai/baselines) (CartPole notebooks only)

Optional / utility dependencies:

- `tqdm`, `cloudpickle`, `psutil`, `joblib` (logging & run utilities)
- `mpi4py` (only if using MPI features in `mpi_tools.py`)
- CUDA-capable GPU (recommended for Atari experiments)

Example installation:

```bash
pip install torch numpy scipy matplotlib pandas gym tensorboardX opencv-python gym-maze tqdm cloudpickle psutil joblib

# Atari ROM support (required for Seaquest experiments)
pip install "gym[atari,accept-rom-license]"

# CartPole notebooks only
pip install git+https://github.com/openai/baselines.git
```

---

## Setup

### 1. Working directory

Run notebooks and scripts from the `code/` directory so local imports resolve correctly:

```bash
cd code
jupyter notebook
```

### 2. Logger configuration (`user_config.py`)

Atari experiments import `run_utils.py`, which expects a `user_config.py` module in the same directory. Create `code/user_config.py` with:

```python
import os.path as osp

# Directory where experiment logs and checkpoints are saved
DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(__file__)), 'data')

# Spinning Up-style logger options
FORCE_DATESTAMP = False
DEFAULT_SHORTHAND = False
WAIT_BEFORE_LAUNCH = 3
```

### 3. GPU settings (Atari)

The Atari scripts/notebooks assume CUDA is available. If you run on CPU or a different GPU index, edit the device settings in:

- `SoftDQN_VER_Seaquest.py`
- `SoftDQN_VER_Seaquest.ipynb`
- `SoftDQN_DisCor_Seaquest.ipynb`

For example, remove or change `torch.cuda.set_device(2)` and set `USE_CUDA` accordingly.

---

## Experiments

### Figure 3 & 4 — Maze (Q-learning vs Soft Q-learning)

Analyzes the relationship between TD error and value-relevant quantities in a tabular maze setting.

| Notebook | Algorithm |
|----------|-----------|
| [maze_q_learning.ipynb](./code/maze_q_learning.ipynb) | Q-learning |
| [maze_soft_q_learning.ipynb](./code/maze_soft_q_learning.ipynb) | Soft Q-learning |

**Environment:** `gym_maze`  
**Outputs:** scatter plots and saved pickle results (`td`, `evb`, `piv`, `eiv`, episode stats)

---

### Figure 4 — CartPole (DQN vs Soft DQN)

Compares standard DQN and Soft DQN on `CartPole-v0`.

| Notebook | Algorithm |
|----------|-----------|
| [cartpole_dqn.ipynb](./code/cartpole_dqn.ipynb) | DQN |
| [cartpole_soft_dqn.ipynb](./code/cartpole_soft_dqn.ipynb) | Soft DQN |

**Environment:** `CartPole-v0`  
**Outputs:** training curves and TensorBoard logs

---

### Figure 6 — Atari Seaquest (VER)

Implements **Soft DQN + Valuable Experience Replay (VER)** on `SeaquestNoFrameskip-v4`.

| File | Description |
|------|-------------|
| [SoftDQN_VER_Seaquest.ipynb](./code/SoftDQN_VER_Seaquest.ipynb) | Interactive notebook |
| [SoftDQN_VER_Seaquest.py](./code/SoftDQN_VER_Seaquest.py) | Standalone training script |

**Key VER update rule** (priority for PER):

```python
td_error = predicted_qvalues_for_actions - target_qvalues_for_actions
priority = action_prob * td_error   # VER: policy-weighted TD error
```

where `action_prob = softmax(Q(s, ·)/α)[a]`.

**Default hyperparameters (Seaquest):**

| Parameter | Value |
|-----------|-------|
| `env_id` | `SeaquestNoFrameskip-v4` |
| `learning_rate` | `1e-4` |
| `batch_size` | `32` |
| `max_buff` | `1,000,000` |
| `prio_a` | `0.6` |
| `prio_beta` | `0.4` |
| `gamma` | `0.99` |
| `update_tar_interval` | `10,000` |
| `learning_start` | `50,000` |

Run the script:

```bash
cd code
python SoftDQN_VER_Seaquest.py
```

Logs and checkpoints are written under `code/data/` (or `DEFAULT_DATA_DIR` if configured).

---

### Baseline — DisCor on Seaquest

[SoftDQN_DisCor_Seaquest.ipynb](./code/SoftDQN_DisCor_Seaquest.ipynb) implements the **DisCor** baseline for comparison on the same Atari environment.

---

## Logging & Outputs

- **Maze / CartPole:** plots are rendered inline in notebooks; maze runs also export pickle files.
- **Atari:** uses `EpochLogger` (`logx.py`) to write `progress.txt`, model checkpoints, and optional evaluation videos under the experiment directory.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{ver2025,
  title     = {Revisiting Prioritized Experience Replay: A Value Perspective},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025}
}
```

*(Update author names and paper URL/DOI when available.)*

---

## License

Please refer to the paper authors / repository maintainers for licensing terms.
