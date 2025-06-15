Ось повністю оновлений `README.md`, з інтегрованими графіками та конфігом із **trial 13**, готовий до викладки на GitHub:

---

````markdown
# DQN Trading Signal Optimization

This repository contains a baseline Deep Q-Network (DQN) implementation for trading signal generation with hyperparameter optimization powered by Optuna and experiment tracking via Weights & Biases (W&B).

---

## Project Overview

We train a DQN agent to learn trading strategies on financial time series data using reinforcement learning. Optuna is used for efficient hyperparameter tuning, while W&B logs training metrics and model artifacts.

### Key Features

- Modular PyTorch-based DQN training pipeline
- Hyperparameter search with Optuna (supports pruning)
- GPU memory monitoring during training
- Saving best models and configs automatically
- W&B integration for detailed experiment tracking

---

## Setup

### Prerequisites

- Python 3.8+ (recommended 3.10+)
- PyTorch (with CUDA support if GPU available)
- PostgreSQL database for Optuna storage (optional but recommended for persistent trials)
- W&B account and API key for experiment logging

### Installation

```bash
pip install -r requirements.txt
````

Create a `.env` file with the following variables:

```
OPTUNA_DB=postgresql://username:password@localhost:5432/optuna_db
STUDY_NAME=dqn_study
WANDB_API_KEY=your_wandb_api_key
```

Make sure to replace `username`, `password`, and other placeholders with your actual credentials.

---

## Usage

Run hyperparameter optimization with:

```bash
python optuna_train.py
```

This will start Optuna trials, log progress to W\&B, and save the best models and configs in the `optuna_models/` directory.

---

## Configuration

Main hyperparameters tuned by Optuna:

| Parameter   | Description                                | Range / Default |
| ----------- | ------------------------------------------ | --------------- |
| lr          | Learning rate for optimizer                | 1e-5 to 1e-3    |
| gamma       | Discount factor for future rewards         | 0.90 to 0.999   |
| eps\_start  | Starting epsilon for epsilon-greedy policy | 0.8 to 1.0      |
| eps\_end    | Minimum epsilon value                      | 0.01 to 0.1     |
| eps\_decay  | Epsilon decay rate (steps)                 | 500 to 3000     |
| tau         | Target network update rate                 | 0.001 to 0.02   |
| batch\_size | Mini-batch size                            | 64, 128, 256    |

---

## Data

Place your financial time series CSV files (e.g. `EURUSD30.csv`) in the project root or a specified `/data/` folder.
Modify `train.py` or data loading scripts if needed to match your dataset.

---

## ✅ Sample Trial — Run #13

The following metrics were logged during one of the best runs (trial 13):

### Average Reward Over Time

![Average Reward](docs/assets/wandb_chart_1.png)

### Epsilon and Loss Metrics

![Epsilon and Loss](docs/assets/wandb_chart_2.png)

### Trial 13 Configuration

```yaml
batch_size: 128
early_stop_patience: 20
episodes: 55
eps_decay: 841
eps_end: 0.03708619719247948
eps_start: 0.8431333228793039
gamma: 0.9054965016193754
lr: 0.0006181456255736161
memory_size: 10000
model_dir: optuna_models/trial_13
reward_scale: 1.0
run_name: trial_13
tau: 0.015858218049624916
wandb: true
```

---

## Notes

* Ensure your PostgreSQL database is running and accessible via the connection string in `OPTUNA_DB`.
* W\&B will log metrics under the project named `dqn-trading-optuna`.
* You can adjust training parameters and hyperparameter ranges in `optuna_train.py` and `train.py`.
* To stop early trials, Optuna uses pruning — speeding up the search process.

---

## License

MIT License — feel free to use and modify.

---

## Contact

For questions or contributions, reach out to [druilsenctr@gmail.com](mailto:druilsenctr@gmail.com).

````

---

### 📌 Додаткові кроки

1. 📁 **Розмісти файли:**

   - `docs/assets/wandb_chart_1.png`
   - `docs/assets/wandb_chart_2.png`
   - `optuna_models/trial_13/config.yaml`

2. 🔧 **Git add + commit:**

```bash
git add README.md docs/assets/*.png optuna_models/trial_13/config.yaml
git commit -m "Add Trial 13 W&B charts and config to README"
git push
````

Хочеш — можу автоматизувати це як `docs/gen_readme.py`, який підтягує з `optuna_models/` активну конфігурацію та графіки.
