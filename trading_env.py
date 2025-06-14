# trading_env.py
import dm_env
import numpy as np
from dm_env import specs
from typing import Optional

class TradingEnv(dm_env.Environment):
    def __init__(self, raw_data: np.ndarray, raw_data_h4: Optional[np.ndarray] = None):
        self.raw_data = raw_data
        self.raw_data_h4 = raw_data_h4

        # RSI from M30
        self.rsi = self._compute_rsi(self.raw_data[:, 3], window=14)

        # RSI from H4 (optional, disabled for now)
        # if self.raw_data_h4 is not None:
        #     self.rsi_h4 = self._compute_rsi(self.raw_data_h4[:, 3], window=14)
        #     self.rsi_h4_interpolated = np.repeat(self.rsi_h4, 8)[:len(self.raw_data)]
        # else:
        #     self.rsi_h4_interpolated = np.zeros(len(self.raw_data))

        # Combine base data with indicators
        self.data = np.hstack([
            self.raw_data,                          # OHLCV (5 columns)
            self.rsi.reshape(-1, 1),                # M30 RSI
            # self.rsi_h4_interpolated.reshape(-1, 1) # H4 RSI
        ])

        # Initial internal state
        self.current_step = 14
        self.position = 0
        self.done = False
        
        self.trailing_gap = 0.002
        self.entry_price = None
        self.trailing_stop = None

        self._obs_spec = specs.Array(shape=(self.data.shape[1] + 1,), dtype=np.float32, name='observation')
        self._action_spec = specs.BoundedArray(shape=(), dtype=np.int32, minimum=-1, maximum=1, name='action')

        self._last_price = self.raw_data[self.current_step, 3]

    def reset(self) -> dm_env.TimeStep:
        self.current_step = 14
        self.position = 0
        self.done = False
        self._last_price = self.raw_data[self.current_step, 3]
        self.entry_price = None
        self.trailing_stop = None
        obs = self._get_obs()
        return dm_env.restart(obs)

    def step(self, action: int) -> dm_env.TimeStep:
        if self.done:
            return self.reset()

        prev_price = self.raw_data[self.current_step, 3]
        self.current_step += 1

        if self.current_step >= len(self.raw_data):
            self.done = True
            obs = np.append(self.data[self.current_step - 1], self.position).astype(np.float32)
            reward = 0.0
            return dm_env.termination(reward=reward, observation=obs)

        current_price = self.raw_data[self.current_step, 3]
        reward = 0.0

        # === TRAILING STOP LOGIC ===
        if self.position != 0:
            if self.trailing_stop is not None:
                if self.position == 1:
                    self.trailing_stop = max(self.trailing_stop, current_price - self.trailing_gap)
                    if current_price < self.trailing_stop:
                        reward = current_price - self.entry_price
                        self.position = 0
                        self.entry_price = None
                        self.trailing_stop = None
                elif self.position == -1:
                    self.trailing_stop = min(self.trailing_stop, current_price + self.trailing_gap)
                    if current_price > self.trailing_stop:
                        reward = self.entry_price - current_price
                        self.position = 0
                        self.entry_price = None
                        self.trailing_stop = None

        # === AGENT ACTION ===
        if self.position == 0 and action != 0:
            self.position = action
            self.entry_price = current_price
            if action == 1:
                self.trailing_stop = current_price - self.trailing_gap
            elif action == -1:
                self.trailing_stop = current_price + self.trailing_gap

        obs = self._get_obs()
        return dm_env.transition(reward=reward, observation=obs)

    def _get_obs(self):
        features = self.data[self.current_step]
        return np.append(features, self.position).astype(np.float32)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def _compute_rsi(self, prices, window=14):
        deltas = np.diff(prices)
        seed = deltas[:window]
        up = seed[seed >= 0].sum() / window
        down = -seed[seed < 0].sum() / window
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:window] = 100. - 100. / (1. + rs)

        for i in range(window, len(prices)):
            delta = deltas[i - 1]
            upval = max(delta, 0)
            downval = -min(delta, 0)
            up = (up * (window - 1) + upval) / window
            down = (down * (window - 1) + downval) / window
            rs = up / down if down != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)
        return rsi

    def _compute_ema(self, prices, window=14):
        ema = np.zeros_like(prices)
        alpha = 2 / (window + 1)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        return ema

    def _align_h4_to_base(self, base_data, h4_data, h4_feature):
        base_timestamps = base_data[:, 0].astype(np.int64)
        h4_timestamps = h4_data[:, 0].astype(np.int64)

        h4_map = dict(zip(h4_timestamps, h4_feature))
        aligned = np.zeros(len(base_data), dtype=np.float32)

        last_val = 50.0
        for i, ts in enumerate(base_timestamps):
            past_ts = h4_timestamps[h4_timestamps <= ts]
            if len(past_ts) > 0:
                last_val = h4_map[past_ts[-1]]
            aligned[i] = last_val
        return aligned
