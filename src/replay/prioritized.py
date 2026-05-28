"""
Replay Buffer cho Rainbow DQN — kết hợp 2 cải tiến:
  1. Prioritized Experience Replay (Schaul et al. 2015)
     - SumTree cho O(log n) sample/update
     - Importance Sampling correction với β annealing
     - Priority = KL loss (không phải TD error như DQN gốc)

  2. N-step Return (Sutton 1988)
     - Tích lũy R_t^(n) = Σ γ^k * R_{t+k+1} trong n bước
     - Bootstrap từ state S_{t+n} thay vì S_{t+1}
"""

import numpy as np
from collections import deque
from typing import Tuple, Optional


# ─────────────────────────────────────────────
# 1. SumTree — cấu trúc dữ liệu cho priority
# ─────────────────────────────────────────────

class SumTree:
    """
    Binary tree lưu priorities — O(log n) cho update và sample.

    Cấu trúc:
        - Leaf nodes [n_leaves:] lưu priority của từng transition
        - Internal nodes lưu tổng priority của subtree con
        - Root = tổng tất cả priorities
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity, dtype=np.float64)
        self.data_ptr = 0   # con trỏ ghi vòng
        self.size     = 0   # số transitions hiện có

    def update(self, idx: int, priority: float):
        """Cập nhật priority tại leaf idx và propagate lên root."""
        tree_idx = idx + self.capacity
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        # Propagate
        tree_idx //= 2
        while tree_idx >= 1:
            self.tree[tree_idx] += delta
            tree_idx //= 2

    def add(self, priority: float) -> int:
        """Thêm transition mới với priority, trả về idx."""
        idx = self.data_ptr
        self.update(idx, priority)
        self.data_ptr = (self.data_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return idx

    def sample(self, value: float) -> int:
        """Tìm leaf idx tương ứng với cumulative sum = value."""
        node = 1  # bắt đầu từ root
        while node < self.capacity:
            left = 2 * node
            if value <= self.tree[left]:
                node = left
            else:
                value -= self.tree[left]
                node = left + 1
        return node - self.capacity  # convert về data index

    @property
    def total_priority(self) -> float:
        return self.tree[1]  # root

    @property
    def max_priority(self) -> float:
        if self.size == 0:
            return 1.0
        return self.tree[self.capacity:self.capacity + self.size].max()

    @property
    def min_priority(self) -> float:
        return self.tree[self.capacity:self.capacity + self.size].min()


# ─────────────────────────────────────────────
# 2. N-step Buffer — tích lũy return nhiều bước
# ─────────────────────────────────────────────

class NStepBuffer:
    """
    Tích lũy n transitions, trả về:
        (state_t, action_t, R_t^(n), gamma^n, state_{t+n}, done_{t+n})

    R_t^(n) = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γ^{n-1}*r_{t+n-1}
    """
    def __init__(self, n_step: int, gamma: float):
        self.n_step = n_step
        self.gamma  = gamma
        self.buffer = deque(maxlen=n_step)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def is_ready(self) -> bool:
        return len(self.buffer) == self.n_step

    def get(self):
        """
        Trả về n-step transition từ oldest entry trong buffer.
        """
        state, action = self.buffer[0][0], self.buffer[0][1]

        # Tính n-step return
        R = 0.0
        gamma_n = 1.0
        for i, (_, _, r, _, done) in enumerate(self.buffer):
            R += gamma_n * r
            gamma_n *= self.gamma
            if done:
                break

        # State và done tại bước n (hoặc bước terminal gần nhất)
        next_state = self.buffer[-1][3]
        done_n     = self.buffer[-1][4]

        return state, action, R, gamma_n, next_state, done_n

    def clear(self):
        self.buffer.clear()


# ─────────────────────────────────────────────
# 3. Prioritized Replay Buffer — chính
# ─────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """
    Replay buffer với Prioritized Experience Replay + N-step return.

    Workflow:
        1. add() transition → NStepBuffer tích lũy n bước
        2. Sau n bước → lưu (s, a, R^n, γ^n, s_n, done) vào SumTree
        3. sample() → lấy batch theo priority + tính IS weights
        4. update_priorities() → cập nhật sau khi tính KL loss

    Lưu observations dưới dạng uint8 (không float32) → tiết kiệm 4× RAM.
    """

    def __init__(
        self,
        capacity:     int   = 1_000_000,
        n_step:       int   = 3,
        gamma:        float = 0.99,
        alpha:        float = 0.5,    # priority exponent ω trong paper
        beta_start:   float = 0.4,    # IS exponent ban đầu
        beta_end:     float = 1.0,    # IS exponent cuối training
        total_frames: int   = 10_000_000,
        obs_shape:    tuple = (4, 84, 84),
    ):
        self.capacity     = capacity
        self.n_step       = n_step
        self.gamma        = gamma
        self.alpha        = alpha
        self.beta_start   = beta_start
        self.beta_end     = beta_end
        self.total_frames = total_frames

        # SumTree cho priorities
        self.tree = SumTree(capacity)

        # Pre-allocate arrays — dùng uint8 để tiết kiệm RAM
        self.states      = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions     = np.zeros(capacity, dtype=np.int64)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.gammas      = np.zeros(capacity, dtype=np.float32)  # γ^n
        self.dones       = np.zeros(capacity, dtype=np.float32)

        # N-step buffer
        self.nstep_buf = NStepBuffer(n_step, gamma)

        # Epsilon nhỏ để tránh priority = 0
        self._eps = 1e-6

    def _beta(self, current_frame: int) -> float:
        """Linear annealing β từ beta_start → beta_end."""
        frac = min(current_frame / self.total_frames, 1.0)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def add(self, state, action: int, reward: float,
            next_state, done: bool):
        """
        Thêm 1 transition vào n-step buffer.
        Khi n-step buffer đủ n bước → flush vào replay buffer chính.
        """
        self.nstep_buf.add(
            np.array(state, dtype=np.uint8),
            action, reward,
            np.array(next_state, dtype=np.uint8),
            done
        )

        if self.nstep_buf.is_ready():
            self._flush_nstep()

        # Episode kết thúc → flush những transitions còn lại
        if done:
            while len(self.nstep_buf.buffer) > 0:
                self._flush_nstep()
                self.nstep_buf.buffer.popleft()

    def _flush_nstep(self):
        """Lưu n-step transition vào SumTree với max priority."""
        state, action, R, gamma_n, next_state, done = self.nstep_buf.get()

        # Priority mới = max priority hiện tại (để đảm bảo được sample ít nhất 1 lần)
        priority = max(self.tree.max_priority, self._eps) ** self.alpha

        idx = self.tree.add(priority)
        self.states[idx]      = state
        self.next_states[idx] = next_state
        self.actions[idx]     = action
        self.rewards[idx]     = R
        self.gammas[idx]      = gamma_n
        self.dones[idx]       = float(done)

    def sample(self, batch_size: int, current_frame: int) -> dict:
        """
        Sample batch_size transitions theo priority.

        Returns dict với keys:
            states, next_states, actions, rewards, gammas, dones,
            weights (IS weights), indices (để update priorities sau)
        """
        assert self.tree.size >= batch_size, \
            f"Buffer chỉ có {self.tree.size} transitions, cần {batch_size}"

        beta = self._beta(current_frame)
        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        # Chia total priority thành batch_size phân đoạn đều nhau
        segment = self.tree.total_priority / batch_size
        for i in range(batch_size):
            low  = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            idx = self.tree.sample(value)
            indices[i]   = idx
            priorities[i] = self.tree.tree[idx + self.tree.capacity]

        # IS weights = (N * p_i)^{-β} / max_weight
        N = self.tree.size
        probs   = priorities / self.tree.total_priority
        weights = (N * probs) ** (-beta)
        weights /= weights.max()  # normalize

        return {
            "states":      self.states[indices].copy(),
            "next_states": self.next_states[indices].copy(),
            "actions":     self.actions[indices].copy(),
            "rewards":     self.rewards[indices].copy(),
            "gammas":      self.gammas[indices].copy(),
            "dones":       self.dones[indices].copy(),
            "weights":     weights.astype(np.float32),
            "indices":     indices,
        }

    def update_priorities(self, indices: np.ndarray, kl_losses: np.ndarray):
        """
        Cập nhật priorities sau khi tính KL loss.
        Trong Rainbow: priority = KL_loss^α (không phải |TD error|^α)
        """
        for idx, kl in zip(indices, kl_losses):
            priority = (float(kl) + self._eps) ** self.alpha
            self.tree.update(int(idx), priority)

    @property
    def size(self) -> int:
        return self.tree.size

    def __len__(self) -> int:
        return self.size


if __name__ == "__main__":
    # Unit test
    buf = PrioritizedReplayBuffer(
        capacity=1000, n_step=3, gamma=0.99,
        obs_shape=(4, 84, 84), total_frames=100_000
    )

    # Thêm dummy transitions
    dummy_obs = np.zeros((4, 84, 84), dtype=np.uint8)
    for i in range(200):
        done = (i % 20 == 19)
        buf.add(dummy_obs, action=0, reward=1.0,
                next_state=dummy_obs, done=done)

    print(f"Buffer size: {buf.size}")
    assert buf.size > 0, "Buffer rỗng!"

    batch = buf.sample(32, current_frame=1000)
    print(f"states shape:  {batch['states'].shape}")    # (32, 4, 84, 84)
    print(f"rewards shape: {batch['rewards'].shape}")   # (32,)
    print(f"weights range: [{batch['weights'].min():.3f}, {batch['weights'].max():.3f}]")

    # Update priorities
    fake_kl = np.random.rand(32).astype(np.float32)
    buf.update_priorities(batch["indices"], fake_kl)

    print("✓ PrioritizedReplayBuffer OK")
