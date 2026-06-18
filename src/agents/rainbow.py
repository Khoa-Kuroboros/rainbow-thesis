"""
Rainbow Agent — kết hợp tất cả 6 cải tiến:
  1. Double Q-learning       → bootstrap action từ online net, evaluate bằng target net
  2. Prioritized Replay      → sample theo KL priority
  3. Dueling Network         → trong RainbowNet (value + advantage streams)
  4. Multi-step learning     → trong PrioritizedReplayBuffer (n-step return)
  5. Distributional RL (C51) → KL divergence loss + Bellman projection
  6. Noisy Nets              → trong RainbowNet (NoisyLinear layers)
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Optional

from src.networks.rainbow_net import RainbowNet, DQNNet, build_network
from src.replay.prioritized import PrioritizedReplayBuffer


class RainbowAgent:
    """
    Rainbow DQN Agent.

    Hyperparameters mặc định theo Table 1 của paper:
        lr=6.25e-5, adam_eps=1.5e-4, n_atoms=51, n_step=3
        target_update=32000, batch_size=32, min_history=80000
    """

    def __init__(
        self,
        n_actions:        int,
        device:           torch.device,
        # Distributional
        n_atoms:          int   = 51,
        v_min:            float = -10.0,
        v_max:            float = 10.0,
        # Noisy
        sigma_0:          float = 0.5,
        # Training
        lr:               float = 6.25e-5,
        adam_eps:         float = 1.5e-4,
        batch_size:       int   = 32,
        target_update:    int   = 32_000,
        discount:         float = 0.99,
        min_history:      int   = 80_000,
        # Replay
        buffer_capacity:  int   = 1_000_000,
        n_step:           int   = 3,
        priority_omega:   float = 0.5,
        beta_start:       float = 0.4,
        beta_end:         float = 1.0,
        total_frames:     int   = 10_000_000,
        # Ablation
        ablation:         Optional[str] = None,
        # Epsilon-greedy (chỉ dùng khi ablation = no_noisy)
        eps_start:        float = 1.0,
        eps_end:          float = 0.01,
        eps_decay_frames: int   = 250_000,
    ):
        self.n_actions     = n_actions
        self.device        = device
        self.n_atoms       = n_atoms
        self.v_min         = v_min
        self.v_max         = v_max
        self.batch_size    = batch_size
        self.target_update = target_update
        self.discount      = discount
        self.min_history   = min_history
        self.ablation      = ablation
        self.total_frames  = total_frames

        # Support z (fixed atoms)
        self.support = torch.linspace(v_min, v_max, n_atoms).to(device)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # Epsilon-greedy (cho no_noisy ablation)
        self.eps_start        = eps_start
        self.eps_end          = eps_end
        self.eps_decay_frames = eps_decay_frames

        # Distributional flag
        self.use_distributional = (ablation != "no_distributional")
        self.use_double         = (ablation != "no_double")
        self.use_noisy          = (ablation != "no_noisy")

        # ── Networks ──────────────────────────────────────
        self.online_net = build_network(
            n_actions, ablation, n_atoms, v_min, v_max, sigma_0
        ).to(device)

        self.target_net = build_network(
            n_actions, ablation, n_atoms, v_min, v_max, sigma_0
        ).to(device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # target net không train

        # ── Optimizer ─────────────────────────────────────
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=lr, eps=adam_eps
        )

        # ── Replay Buffer ──────────────────────────────────
        self.replay = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            n_step=n_step,
            gamma=discount,
            alpha=priority_omega,
            beta_start=beta_start,
            beta_end=beta_end,
            total_frames=total_frames,
        )

        # ── Counters ───────────────────────────────────────
        self.learn_steps  = 0   # số lần gọi learn()
        self.frame_count  = 0   # tổng frames

    # ─────────────────────────────────────────────
    # Action selection
    # ─────────────────────────────────────────────

    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        Chọn action.
        - Training + Noisy Nets: greedy theo Q (noise built-in)
        - Training + no_noisy:   epsilon-greedy
        - Eval mode:             greedy, no noise
        """
        # Epsilon-greedy cho no_noisy ablation
        if not eval_mode and not self.use_noisy:
            eps = max(
                self.eps_end,
                self.eps_start - self.frame_count / self.eps_decay_frames
                * (self.eps_start - self.eps_end)
            )
            if np.random.random() < eps:
                return np.random.randint(self.n_actions)

        state_t = torch.tensor(
            np.array(state, dtype=np.float32)[None], device=self.device
        )  # (1, 4, 84, 84) float32

        if eval_mode:
            self.online_net.eval()

        with torch.no_grad():
            if self.use_distributional:
                _, q_values = self.online_net(state_t)
            else:
                q_values = self.online_net(state_t)

        if eval_mode:
            self.online_net.train()
            self.online_net.reset_noise()

        return q_values.argmax(dim=1).item()

    # ─────────────────────────────────────────────
    # Store transition
    # ─────────────────────────────────────────────

    def store(self, state, action: int, reward: float,
              next_state, done: bool):
        self.frame_count += 1
        self.replay.add(state, action, reward, next_state, done)

    # ─────────────────────────────────────────────
    # Learning step
    # ─────────────────────────────────────────────

    def learn(self) -> Optional[float]:
        """
        1 gradient step.
        Returns: loss value (float) hoặc None nếu chưa đủ data.
        """
        if self.replay.size < self.min_history:
            return None

        # ── 1. Sample batch ──────────────────────────────
        batch = self.replay.sample(self.batch_size, self.frame_count)

        states      = torch.tensor(batch["states"],      dtype=torch.uint8 ).to(self.device)
        next_states = torch.tensor(batch["next_states"], dtype=torch.uint8 ).to(self.device)
        actions     = torch.tensor(batch["actions"],     dtype=torch.long  ).to(self.device)
        rewards     = torch.tensor(batch["rewards"],     dtype=torch.float32).to(self.device)
        gammas      = torch.tensor(batch["gammas"],      dtype=torch.float32).to(self.device)
        dones       = torch.tensor(batch["dones"],       dtype=torch.float32).to(self.device)
        weights     = torch.tensor(batch["weights"],     dtype=torch.float32).to(self.device)
        indices     = batch["indices"]

        # Reset noise mỗi gradient step
        self.online_net.reset_noise()
        self.target_net.reset_noise()

        # ── 2. Compute loss ──────────────────────────────
        if self.use_distributional:
            loss, kl_losses = self._distributional_loss(
                states, next_states, actions, rewards, gammas, dones
            )
        else:
            loss, kl_losses = self._td_loss(
                states, next_states, actions, rewards, gammas, dones
            )

        # ── 3. Weighted loss (IS correction) ─────────────
        weighted_loss = (weights * kl_losses).mean()

        # ── 4. Gradient step ─────────────────────────────
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        # ── 5. Update priorities ─────────────────────────
        self.replay.update_priorities(
            indices,
            kl_losses.detach().cpu().numpy()
        )

        # ── 6. Periodic target network update ────────────
        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return weighted_loss.item()

    # ─────────────────────────────────────────────
    # Distributional loss (C51) — Bellman projection + KL divergence
    # ─────────────────────────────────────────────

    def _distributional_loss(self, states, next_states, actions,
                              rewards, gammas, dones):
        """
        KL divergence loss với Bellman projection.

        Target distribution:
            d'_t = (R_t^n + γ^n * z, p_θ'(s_{t+n}, a*))
        dimana a* = argmax_a Q_online(s_{t+n}, a)  ← Double Q-learning

        Projection Φz: chiếu target distribution về fixed support z.
        Loss = KL(Φz(d'_t) || d_t)
        """
        batch_size = states.size(0)

        with torch.no_grad():
            # ── Target distribution ──────────────────────
            # Double Q: chọn action bằng online net
            if self.use_double:
                _, q_online_next = self.online_net(next_states)
                best_actions = q_online_next.argmax(dim=1)  # (batch,)
            else:
                _, q_target_next = self.target_net(next_states)
                best_actions = q_target_next.argmax(dim=1)

            # Evaluate bằng target net
            probs_next, _ = self.target_net(next_states)  # (batch, n_actions, n_atoms)
            probs_next = probs_next[range(batch_size), best_actions]  # (batch, n_atoms)

            # ── Bellman projection ───────────────────────
            # T_z = clip(R + γ^n * z, v_min, v_max)
            T_z = rewards.unsqueeze(1) + \
                  (1 - dones.unsqueeze(1)) * gammas.unsqueeze(1) * \
                  self.support.unsqueeze(0)
            T_z = T_z.clamp(self.v_min, self.v_max)  # (batch, n_atoms)

            # b = (T_z - v_min) / delta_z — vị trí liên tục trên support
            b   = (T_z - self.v_min) / self.delta_z   # (batch, n_atoms)
            lo  = b.floor().long().clamp(0, self.n_atoms - 1)
            hi  = b.ceil().long().clamp(0, self.n_atoms - 1)

            # Phân phối target m — phân bổ probability về 2 atoms lân cận
            m = torch.zeros(batch_size, self.n_atoms, device=self.device)
            offset = torch.arange(batch_size, device=self.device) \
                         .unsqueeze(1) * self.n_atoms  # (batch, 1)

            # Fix: khi lo==hi (b là số nguyên), cả 2 weight = 0 → mất mass.
            # Phải gán full mass về 1 phía khi lo==hi.
            lo_weight = hi.float() - b
            hi_weight = b - lo.float()
            eq_mask = (lo == hi)
            lo_weight = torch.where(eq_mask, torch.ones_like(lo_weight), lo_weight)
            hi_weight = torch.where(eq_mask, torch.zeros_like(hi_weight), hi_weight)

            m.view(-1).index_add_(
                0, (lo + offset).view(-1),
                (probs_next * lo_weight).view(-1)
            )
            m.view(-1).index_add_(
                0, (hi + offset).view(-1),
                (probs_next * hi_weight).view(-1)
            )
            # m: (batch, n_atoms) — target distribution, sum = 1

        # ── Online network prediction ────────────────────
        probs_online, _ = self.online_net(states)   # (batch, n_actions, n_atoms)
        probs_online = probs_online[range(batch_size), actions]  # (batch, n_atoms)

        # ── KL divergence = -Σ m * log(p) ───────────────
        log_p = torch.log(probs_online.clamp(min=1e-8))
        kl_losses = -(m * log_p).sum(dim=1)          # (batch,)

        # Tổng loss trước khi weight (để return cho priority update)
        loss = kl_losses.mean()
        return loss, kl_losses

    # ─────────────────────────────────────────────
    # Standard TD loss (cho no_distributional ablation)
    # ─────────────────────────────────────────────

    def _td_loss(self, states, next_states, actions,
                 rewards, gammas, dones):
        """MSE TD loss cho DQN chuẩn."""
        q_online = self.online_net(states)  # (batch, n_actions)
        q_pred   = q_online[range(len(actions)), actions]

        with torch.no_grad():
            if self.use_double:
                best_actions = self.online_net(next_states).argmax(dim=1)
                q_target_next = self.target_net(next_states)
            else:
                q_target_next = self.target_net(next_states)
                best_actions  = q_target_next.argmax(dim=1)

            q_next = q_target_next[range(len(actions)), best_actions]
            q_target = rewards + (1 - dones) * gammas * q_next

        td_errors = F.smooth_l1_loss(q_pred, q_target, reduction="none")
        loss = td_errors.mean()
        return loss, td_errors

    # ─────────────────────────────────────────────
    # Checkpoint save / load
    # ─────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "online_net":   self.online_net.state_dict(),
            "target_net":   self.target_net.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "learn_steps":  self.learn_steps,
            "frame_count":  self.frame_count,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.learn_steps = ckpt["learn_steps"]
        self.frame_count = ckpt.get("frame_count", 0)
        print(f"✓ Loaded checkpoint: {self.learn_steps} steps, "
              f"{self.frame_count} frames")
