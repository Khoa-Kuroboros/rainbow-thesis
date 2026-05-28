"""
Atari Wrappers — chuẩn hoá môi trường theo đúng paper DQN/Rainbow.
Áp dụng: grayscale, resize 84x84, frame stack 4, reward clipping, v.v.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2


class NoopResetEnv(gym.Wrapper):
    """Thực hiện N random no-op lúc reset để khởi đầu đa dạng."""
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """Lặp action 4 lần (frame skip), max pooling 2 frame cuối."""
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


class EpisodicLifeEnv(gym.Wrapper):
    """Coi mất mạng = terminal (chỉ trong training, không phải eval)."""
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True  # mất mạng = terminal
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    """Nhấn FIRE lúc reset cho những game cần (Breakout, v.v.)."""
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class WarpFrame(gym.ObservationWrapper):
    """Convert RGB (210,160,3) → grayscale (84,84,1)."""
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(1, self.height, self.width),
            dtype=np.uint8,
        )

    def observation(self, obs):
        frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        return frame[np.newaxis, :, :]  # (1, 84, 84)


class ClipRewardEnv(gym.RewardWrapper):
    """Clip reward về {-1, 0, +1}."""
    def reward(self, reward):
        return np.sign(reward)


class FrameStack(gym.Wrapper):
    """
    Stack 4 frames liên tiếp thành state (4, 84, 84).
    Dùng LazyFrames để tiết kiệm RAM — không copy array cho đến khi cần.
    """
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self._frames = []
        low = np.repeat(env.observation_space.low, n_frames, axis=0)
        high = np.repeat(env.observation_space.high, n_frames, axis=0)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._frames = [obs] * self.n_frames
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.pop(0)
        self._frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return LazyFrames(list(self._frames))


class LazyFrames:
    """
    Tối ưu RAM: lưu list frames, chỉ convert sang np.array khi gọi.
    Replay buffer 1M transitions: 27GB (float32) → 6.7GB (uint8 LazyFrames).
    """
    __slots__ = ["_frames", "_out"]

    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    @property
    def shape(self):
        return self._force().shape


def make_atari_env(game_name: str, clip_rewards: bool = True,
                   episodic_life: bool = True, seed: int = 0):
    """
    Tạo Atari environment đã được wrap đầy đủ theo chuẩn DQN/Rainbow.

    Args:
        game_name: ví dụ 'PongNoFrameskip-v4', 'BreakoutNoFrameskip-v4'
        clip_rewards: True khi training, False khi eval
        episodic_life: True khi training, False khi eval
        seed: random seed

    Returns:
        env đã wrap, observation shape = (4, 84, 84), uint8
    """
    import ale_py
    gym.register_envs(ale_py)

    env = gym.make(game_name, render_mode=None)
    env.reset(seed=seed)

    assert "NoFrameskip" in game_name, \
        "Dùng NoFrameskip variant để tự điều khiển frame skip!"

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)

    if episodic_life:
        env = EpisodicLifeEnv(env)

    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    env = WarpFrame(env)

    if clip_rewards:
        env = ClipRewardEnv(env)

    env = FrameStack(env, n_frames=4)
    return env
