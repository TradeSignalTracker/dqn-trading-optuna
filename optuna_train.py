# optuna_train.py
import os
import time
import yaml
import torch
import wandb
import optuna
import random
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from optuna.pruners import MedianPruner
from train import train, load_and_align_data  # must return avg_reward, model, episodes_ran, avg_duration

# 🔁 Load .env
load_dotenv()
storage = os.getenv("OPTUNA_DB")
study_name = os.getenv("STUDY_NAME")

if not storage or not study_name:
    raise ValueError("❌ Check your .env: OPTUNA_DB and STUDY_NAME must be defined!")

print(f"✅ Optuna DB: {storage}")
print(f"✅ Study name: {study_name}")

# 📥 Load data ONCE for the entire run
raw_data_30, raw_data_h4 = load_and_align_data("EURUSD30.csv", "EURUSD240.csv")

# 🔺 Global best reward tracker
best_reward_so_far = float("-inf")

def objective(trial):
    global best_reward_so_far

    # 🎲 Set seed
    seed = trial.number + 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ⚙️ Trial hyperparameters
    model_dir = f"optuna_models/trial_{trial.number}"
    config = {
        "lr": trial.suggest_float("lr", 5e-5, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.90, 0.999),
        "eps_start": trial.suggest_float("eps_start", 0.8, 1.0),
        "eps_end": trial.suggest_float("eps_end", 0.01, 0.1),
        "eps_decay": trial.suggest_int("eps_decay", 300, 2000),
        "tau": trial.suggest_float("tau", 0.001, 0.02),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
        "episodes": 55,
        "memory_size": 10000,
        "reward_scale": 1.0,
        "wandb": True,
        "early_stop_patience": 30,
        "model_dir": model_dir,
        "run_name": f"trial_{trial.number}",
    }

    start_time = time.time()
    avg_reward, model, episodes_ran, avg_duration = train(config, raw_data_30, raw_data_h4, use_wandb=True)
    duration_sec = time.time() - start_time

    # 📝 Trial user_attrs — must be serializable
    trial.set_user_attr("seed", int(seed))
    trial.set_user_attr("episodes_ran", int(episodes_ran))
    trial.set_user_attr("avg_duration", float(avg_duration))
    trial.set_user_attr("reward", float(avg_reward))

    # 💾 Save model if it's the best so far
    if float(avg_reward) > best_reward_so_far:
        best_reward_so_far = float(avg_reward)
        os.makedirs(model_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
        with open(os.path.join(model_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f)

        print(f"✅ New best model saved: Trial {trial.number}, Reward: {avg_reward:.5f}")

    # 📄 Append to results CSV
    results_path = "optuna_results.csv"
    row = {
        "trial": trial.number,
        "avg_reward": float(avg_reward),
        "episodes_ran": int(episodes_ran),
        "avg_duration": float(avg_duration),
        "seed": int(seed),
        **{k: config[k] for k in ["lr", "gamma", "eps_start", "eps_end", "eps_decay", "tau", "batch_size"]}
    }
    pd.DataFrame([row]).to_csv(results_path, mode='a', header=not os.path.exists(results_path), index=False)

    return float(avg_reward)

if __name__ == "__main__":
    pruner = MedianPruner(n_startup_trials=2, n_warmup_steps=5)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        pruner=pruner
    )

    study.optimize(objective, n_trials=55)

    # ✅ After optimization completes
    best_trial = study.best_trial

    # 📁 Create best_model directory
    best_model_dir = "best_model"
    os.makedirs(best_model_dir, exist_ok=True)

    # 💾 Save best config
    best_config = {
        "seed": best_trial.user_attrs.get("seed"),
        "episodes_ran": best_trial.user_attrs.get("episodes_ran"),
        "avg_duration": best_trial.user_attrs.get("avg_duration"),
        "reward": best_trial.user_attrs.get("reward"),
        **best_trial.params,
        "batch_size": best_trial.params["batch_size"],  # explicit for clarity
    }
    with open(os.path.join(best_model_dir, "best_config.yaml"), "w") as f:
        yaml.dump(best_config, f)

    # 📄 Create summary.json
    summary = {
        "trial_number": best_trial.number,
        "reward": best_trial.value,
        "params": best_trial.params,
        "user_attrs": best_trial.user_attrs,
        "datetime_start": str(best_trial.datetime_start),
        "datetime_complete": str(best_trial.datetime_complete),
    }
    with open(os.path.join(best_model_dir, "summary.json"), "w") as f:
        import json
        json.dump(summary, f, indent=4)

    # 📦 Copy the best model weights
    import shutil
    best_trial_model_path = f"optuna_models/trial_{best_trial.number}/model.pt"
    if os.path.exists(best_trial_model_path):
        shutil.copy(best_trial_model_path, os.path.join(best_model_dir, "model.pt"))
        print(f"✅ Best model copied to {best_model_dir}/model.pt")
    else:
        print(f"⚠️ Model file not found at {best_trial_model_path}")

    print("\n🏆 Best trial:")
    print(study.best_trial)
    print("Params:", study.best_params)
    print(f"\n📄 Best config saved to {os.path.abspath(best_model_dir)}")
    print("🏆 Best Reward:", best_trial.value)
