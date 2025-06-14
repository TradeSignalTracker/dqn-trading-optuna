````
# DQN Trading Signal Optimization

This repository contains a baseline Deep Q-Network (DQN) implementation for trading signal generation with hyperparameter optimization powered by Optuna and experiment tracking via Weights & Biases (W&B).

---

## Project Overview

We train a DQN agent to learn trading strategies on financial time series data using reinforcement learning. Optuna is used for efficient hyperparameter tuning, while W&B logs training metrics and model artifacts.

Key features:
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

Place your financial time series CSV files (e.g. `EURUSD30.csv`) in the `/` folder.
Modify `train.py` or data loading scripts if needed to match your dataset.

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

For questions or contributions, reach out to \[[druilsenctr@gmail.com](mailto:druilsenctr@gmail.com)].

```

---
