# train.py
import yaml
import wandb
import random
import math
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd

from dqn_model import DQN
from trading_env import TradingEnv  


def train(config):
    wandb.init(project="dqn-trading", config=config)

    # === Load and preprocess M30 data ===
    df_30 = pd.read_csv('EURUSD30.csv', sep='\t', header=None)
    df_30 = df_30.drop(columns=[0]).iloc[:15000]
    raw_data_30 = df_30.values.astype(np.float32)

    # === Initialize trading environment ===
    env = TradingEnv(raw_data_30)

    # === Select device ===
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    # === Transition tuple for replay memory ===
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    # === Replay buffer ===
    class ReplayMemory:
        def __init__(self, capacity):
            self.memory = deque([], maxlen=capacity)
        def push(self, *args):
            self.memory.append(Transition(*args))
        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)
        def __len__(self):
            return len(self.memory)

    # === Initialize Q-networks ===
    n_actions = 3  # short, hold, long
    n_observations = env.observation_spec().shape[0]

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=config['lr'], amsgrad=True)
    memory = ReplayMemory(config['memory_size'])

    steps_done = 0
    action_mapping = {0: -1, 1: 0, 2: 1}  # Map discrete actions to env actions

    # === Epsilon-greedy action selection ===
    def select_action(state):
        nonlocal steps_done
        sample = random.random()
        eps_threshold = config['eps_end'] + (config['eps_start'] - config['eps_end']) * math.exp(-1. * steps_done / config['eps_decay'])
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                q_values = policy_net(state)
                action_idx = q_values.max(1).indices.view(1, 1)
                return action_idx
        else:
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

    # === Optimize Q-network ===
    def optimize_model():
        if len(memory) < config['batch_size']:
            return
        transitions = memory.sample(config['batch_size'])
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device, dtype=torch.bool
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(config['batch_size'], device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * config['gamma']) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    # === Main training loop ===
    num_episodes = config['episodes']
    episode_durations = []
    episode_rewards = []

    early_stop_patience = config.get('early_stop_patience', 20)
    best_avg_reward = float('-inf')
    no_improve_counter = 0

    for i_episode in range(num_episodes):
        timestep = env.reset()
        state = torch.tensor(timestep.observation, dtype=torch.float32, device=device).unsqueeze(0)
        ep_reward = 0.0

        for t in count():
            action_idx = select_action(state)
            real_action = action_mapping[action_idx.item()]
            timestep = env.step(real_action)

            reward = timestep.reward * config['reward_scale']
            ep_reward += reward
            reward_t = torch.tensor([reward], device=device)

            done = timestep.last()
            next_state = None if done else torch.tensor(
                timestep.observation, dtype=torch.float32, device=device
            ).unsqueeze(0)

            memory.push(state, action_idx, next_state, reward_t)
            state = next_state

            optimize_model()

            # === Soft update of target network ===
            if t % 10 == 0:
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = (
                        policy_net_state_dict[key] * config['tau']
                        + target_net_state_dict[key] * (1 - config['tau'])
                    )
                target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                episode_rewards.append(ep_reward)
                avg_duration = np.mean(episode_durations)
                avg_reward = np.mean(episode_rewards)

                print(f"Episode {i_episode+1}/{num_episodes} finished after {t+1} steps. "
                      f"Avg duration: {avg_duration:.2f}, "
                      f"Episode reward: {ep_reward:.2f}, "
                      f"Avg reward: {avg_reward:.2f}")

                wandb.log({
                    "episode_reward": ep_reward,
                    "episode_length": t + 1,
                    "avg_reward": avg_reward,
                    "avg_length": avg_duration,
                    "episode": i_episode + 1
                })

                # === Early stopping if no improvement ===
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1

                if no_improve_counter >= early_stop_patience:
                    print(f"⏹ Early stopping at episode {i_episode+1} due to no improvement")
                    break

                break

    print('Training complete')
    avg_last_rewards = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
    return avg_last_rewards, policy_net


if __name__ == "__main__":
    with open("wandb/wandb_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train(config)
