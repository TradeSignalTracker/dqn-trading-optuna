# optuna_train.py
import os
import torch
import wandb
import optuna
from dotenv import load_dotenv
from optuna.pruners import MedianPruner
from train import train  # should return avg_reward and model

# 🔁 Load variables from .env
load_dotenv()
storage = os.getenv("OPTUNA_DB")
study_name = os.getenv("STUDY_NAME")

if not storage or not study_name:
    raise ValueError("❌ Check .env: OPTUNA_DB and STUDY_NAME must be defined!")

print(f"✅ Optuna DB connected: {storage}")
print(f"✅ Study name: {study_name}")

# 🔺 Global best score for saving top model
best_reward_so_far = float("-inf")

def objective(trial):
    global best_reward_so_far

    config = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.90, 0.999),
        "eps_start": trial.suggest_float("eps_start", 0.8, 1.0),
        "eps_end": trial.suggest_float("eps_end", 0.01, 0.1),
        "eps_decay": trial.suggest_int("eps_decay", 500, 3000),
        "tau": trial.suggest_float("tau", 0.001, 0.02),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "episodes": 100,
        "memory_size": 10000,
        "reward_scale": 1.0,
        "wandb": True,
        "early_stop_patience": 20
    }

    wandb_mode = "online" if config["wandb"] else "disabled"
    wandb.init(
        project="dqn-trading-optuna",
        config=config,
        reinit=True,
        mode=wandb_mode
    )

    torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated() / 1024**2

    # 🎯 Training
    avg_reward, model = train(config)

    mem_after = torch.cuda.memory_allocated() / 1024**2
    mem_delta = mem_after - mem_before

    wandb.log({
        "avg_reward": avg_reward,
        "gpu_mem_MB_before": mem_before,
        "gpu_mem_MB_after": mem_after,
        "gpu_mem_MB_delta": mem_delta,
    })

    # ✅ Save best model so far
    if avg_reward > best_reward_so_far:
        best_reward_so_far = avg_reward
        model_dir = "optuna_models"
        os.makedirs(model_dir, exist_ok=True)

        model_path = f"{model_dir}/model_trial_{trial.number}.pt"
        config_path = f"{model_dir}/config_trial_{trial.number}.yaml"

        torch.save(model.state_dict(), model_path)

        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        print(f"✅ New best model saved: Trial {trial.number}, Reward: {avg_reward:.5f}")

    wandb.finish()
    return avg_reward


if __name__ == "__main__":
    pruner = MedianPruner(n_startup_trials=2, n_warmup_steps=5)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        pruner=pruner
    )

    study.optimize(objective, n_trials=10)

    print("\n🏆 Best trial:")
    print(study.best_trial)
    print("Params:", study.best_params)
