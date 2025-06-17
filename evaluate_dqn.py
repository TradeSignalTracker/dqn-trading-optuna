# utils/evaluate_dqn.py
import torch
import pandas as pd
import numpy as np
from trading_env import TradingEnv
from dqn_model import DQN
import matplotlib.pyplot as plt
import argparse

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='model.pt')
parser.add_argument('--csv_m30', type=str, default='EURUSD30.csv')
parser.add_argument('--csv_h4', type=str, default='EURUSD240.csv')
parser.add_argument('--episodes', type=int, default=1)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

# --- Seed ---
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# --- Data ---
df_base = pd.read_csv(args.csv_m30, sep="\t", header=None).drop(columns=[0])
df_h4 = pd.read_csv(args.csv_h4, sep="\t", header=None).drop(columns=[0])

length = 10000
df_base = df_base.iloc[10000:10000+length]
df_h4 = df_h4.iloc[1250:1250+length]
raw_data = df_base.values.astype(np.float32)
raw_data_h4 = df_h4.values.astype(np.float32)

# --- Env ---
env = TradingEnv(raw_data, raw_data_h4=raw_data_h4)
n_actions = 3
n_observations = env.observation_spec().shape[0]
action_mapping = {0: -1, 1: 0, 2: 1}

# --- Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(n_observations, n_actions).to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()

# --- Evaluation ---
equity = [1.0]
positions = []

with torch.no_grad():
    for ep in range(args.episodes):
        timestep = env.reset()
        state = torch.tensor(timestep.observation, dtype=torch.float32, device=device).unsqueeze(0)

        done = False
        entry_price = None
        position = 0

        while not done:
            if timestep.last() or env.current_step >= len(env.raw_data):
                break

            step = env.current_step
            action_idx = model(state).max(1).indices.view(1, 1)
            action = action_mapping[action_idx.item()]
            timestep = env.step(action)

            if env.current_step >= len(env.raw_data):
                break

            price = env.raw_data[env.current_step, 3]
            reward = timestep.reward

            if position == 0 and action != 0:
                position = action
                entry_price = price
                entry_step = step
                print(f"  Enter position {position} at price {entry_price:.5f} (step {entry_step})")
            elif position != 0:
                if action == 0:
                    pnl = price - entry_price if position == 1 else entry_price - price
                    equity.append(equity[-1] + pnl)
                    positions.append({
                        "entry": entry_price,
                        "exit": price,
                        "side": position,
                        "pnl": pnl,
                        "entry_step": entry_step,
                        "exit_step": env.current_step
                    })
                    print(f"  Close position {position} at price {price:.5f} (step {env.current_step}), PnL={pnl:.5f}")
                    position = 0
                    entry_price = None
                else:
                    equity.append(equity[-1])
            else:
                equity.append(equity[-1])

            state = torch.tensor(timestep.observation, dtype=torch.float32, device=device).unsqueeze(0)

# --- Metrics ---
equity = np.array(equity)
returns = np.diff(equity) / equity[:-1]
sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
calmar = np.mean(returns) / (np.max(np.maximum.accumulate(equity) - equity + 1e-8))
total_pnl = equity[-1] - equity[0]
max_drawdown = np.max(np.maximum.accumulate(equity) - equity)

print("\n=== Evaluation Results ===")
print(f"Trades executed: {len(positions)}")
print(f"Total PnL: {total_pnl:.6f}")
print(f"Sharpe Ratio: {sharpe:.4f}")
print(f"Calmar Ratio: {calmar:.4f}")
print(f"Max Drawdown: {max_drawdown:.4f}")

# --- Trade Analysis ---
trades_df = pd.DataFrame(positions)
trades_df["holding_time"] = trades_df["exit_step"] - trades_df["entry_step"]

longs = trades_df[trades_df["side"] == 1]
shorts = trades_df[trades_df["side"] == -1]
winners = trades_df[trades_df["pnl"] > 0]

avg_pnl_long = longs["pnl"].mean() if not longs.empty else 0
avg_pnl_short = shorts["pnl"].mean() if not shorts.empty else 0
avg_holding_time = trades_df["holding_time"].mean()
win_rate = len(winners) / len(trades_df) if not trades_df.empty else 0
max_win = trades_df["pnl"].max()
max_loss = trades_df["pnl"].min()

print("\n=== Trade Analysis ===")
print(f"Avg PnL (Long): {avg_pnl_long:.5f}")
print(f"Avg PnL (Short): {avg_pnl_short:.5f}")
print(f"Avg Holding Time: {avg_holding_time:.2f} steps")
print(f"Win Rate: {win_rate * 100:.2f}%")
print(f"Max Win: {max_win:.5f}")
print(f"Max Loss: {max_loss:.5f}")

# --- Visualization ---
plt.figure(figsize=(12,6))
plt.plot(equity, label='Equity Curve')
plt.title('Equity Curve')
plt.xlabel('Step')
plt.ylabel('Balance')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()