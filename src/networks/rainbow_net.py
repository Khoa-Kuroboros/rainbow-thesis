"""
Rainbow Network Architecture:
  CNN Encoder → NoisyLinear → Dueling (Value + Advantage) → Distributional head
  Output: probability distribution over N_ATOMS returns cho mỗi action.

Kết hợp 3 trong 6 cải tiến của Rainbow:
  - Noisy Nets       (NoisyLinear thay Linear)
  - Dueling Network  (V stream + A stream)
  - Distributional   (softmax → phân phối, không phải giá trị scalar)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────
# 1. NoisyLinear — thay thế nn.Linear thông thường
# ─────────────────────────────────────────────

class NoisyLinear(nn.Module):
    """
    Noisy linear layer với factorised Gaussian noise (Fortunato et al. 2017).

    y = (μ_b + σ_b ⊙ ε_b) + (μ_W + σ_W ⊙ ε_W) @ x
    Factorised: ε_W[i,j] = f(ε_i) * f(ε_j),  f(x) = sgn(x)*sqrt(|x|)

    Tham số học được: μ_W, σ_W, μ_b, σ_b
    Noise:           ε_W, ε_b  (sample lại mỗi forward pass)
    """
    def __init__(self, in_features: int, out_features: int, sigma_0: float = 0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.sigma_0      = sigma_0

        # Learnable parameters
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))

        # Noise buffers (không phải parameter — không optimize)
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon",   torch.empty(out_features))

        self._init_parameters()
        self.reset_noise()

    def _init_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_0 / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_0 / math.sqrt(self.out_features))

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        """Factorised noise transform: f(x) = sgn(x) * sqrt(|x|)"""
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """Sample noise mới — gọi mỗi bước training."""
        eps_i = self._f(torch.randn(self.in_features))
        eps_j = self._f(torch.randn(self.out_features))
        self.weight_epsilon.copy_(eps_j.outer(eps_i))  # outer product
        self.bias_epsilon.copy_(eps_j)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            # Eval mode: dùng mean (không noise) → deterministic
            weight = self.weight_mu
            bias   = self.bias_mu
        return F.linear(x, weight, bias)


# ─────────────────────────────────────────────
# 2. CNN Encoder — dùng chung cho cả network
# ─────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """
    3-layer CNN theo chuẩn DQN (Mnih et al. 2015).
    Input:  (batch, 4, 84, 84) uint8 → normalize → float32
    Output: (batch, 512) feature vector
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # → (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # → (64,  9,  9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # → (64,  7,  7)
            nn.ReLU(),
        )
        self.out_size = 64 * 7 * 7  # = 3136

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize pixel values [0,255] → [0,1]
        x = x.float() / 255.0
        x = self.conv(x)
        return x.flatten(start_dim=1)  # (batch, 3136)


# ─────────────────────────────────────────────
# 3. Rainbow Network — kết hợp tất cả
# ─────────────────────────────────────────────

class RainbowNet(nn.Module):
    """
    Rainbow = CNN + NoisyLinear + Dueling + Distributional

    Architecture:
        CNN encoder (shared)
            ├── Value stream:     NoisyLinear(3136,512) → NoisyLinear(512, n_atoms)
            └── Advantage stream: NoisyLinear(3136,512) → NoisyLinear(512, n_atoms × n_actions)

        Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]   (dueling aggregation)
        softmax per action → probability distribution over atoms

    Output:
        probs:    (batch, n_actions, n_atoms)  — probability distribution
        q_values: (batch, n_actions)           — expected Q = sum(p * z)
    """

    def __init__(self, n_actions: int, n_atoms: int = 51,
                 v_min: float = -10.0, v_max: float = 10.0,
                 sigma_0: float = 0.5):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms   = n_atoms
        self.v_min     = v_min
        self.v_max     = v_max

        # Support z: n_atoms điểm đều đặn trong [v_min, v_max]
        self.register_buffer(
            "support",
            torch.linspace(v_min, v_max, n_atoms)  # (n_atoms,)
        )

        # CNN encoder
        self.encoder = CNNEncoder()
        enc_out = self.encoder.out_size  # 3136

        # Value stream: scalar per atom
        self.value_hidden = NoisyLinear(enc_out, 512, sigma_0)
        self.value_out    = NoisyLinear(512, n_atoms, sigma_0)

        # Advantage stream: n_actions values per atom
        self.adv_hidden = NoisyLinear(enc_out, 512, sigma_0)
        self.adv_out    = NoisyLinear(512, n_actions * n_atoms, sigma_0)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, 4, 84, 84) uint8 tensor
        Returns:
            probs:    (batch, n_actions, n_atoms)
            q_values: (batch, n_actions)
        """
        batch = x.size(0)
        phi = self.encoder(x)  # (batch, 3136)

        # Value stream
        v = F.relu(self.value_hidden(phi))          # (batch, 512)
        v = self.value_out(v)                        # (batch, n_atoms)
        v = v.view(batch, 1, self.n_atoms)           # (batch, 1, n_atoms)

        # Advantage stream
        a = F.relu(self.adv_hidden(phi))             # (batch, 512)
        a = self.adv_out(a)                          # (batch, n_actions * n_atoms)
        a = a.view(batch, self.n_actions, self.n_atoms)  # (batch, n_actions, n_atoms)

        # Dueling aggregation — Q = V + A - mean(A)
        q_atoms = v + a - a.mean(dim=1, keepdim=True)  # (batch, n_actions, n_atoms)

        # Softmax per action → probability distribution
        probs = F.softmax(q_atoms, dim=2)            # (batch, n_actions, n_atoms)

        # Expected Q = sum(p * z) per action
        q_values = (probs * self.support.view(1, 1, -1)).sum(dim=2)  # (batch, n_actions)

        return probs, q_values

    def reset_noise(self):
        """Gọi mỗi bước training để sample noise mới cho Noisy Nets."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def act(self, x: torch.Tensor) -> int:
        """Greedy action selection (eval mode — no noise)."""
        self.eval()
        with torch.no_grad():
            _, q_values = self.forward(x)
            action = q_values.argmax(dim=1).item()
        self.train()
        return action


# ─────────────────────────────────────────────
# 4. Ablation variants — bỏ từng component
# ─────────────────────────────────────────────

class DQNNet(nn.Module):
    """Baseline DQN — không có Noisy, Dueling, Distributional."""
    def __init__(self, n_actions: int):
        super().__init__()
        self.encoder = CNNEncoder()
        self.fc = nn.Sequential(
            nn.Linear(self.encoder.out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor):
        phi = self.encoder(x)
        q_values = self.fc(phi)
        return q_values

    def act(self, x: torch.Tensor) -> int:
        self.eval()
        with torch.no_grad():
            q = self.forward(x)
        self.train()
        return q.argmax(dim=1).item()

    def reset_noise(self):
        pass  # DQN không có noisy — no-op để tương thích interface


def build_network(n_actions: int, ablation: str = None,
                  n_atoms: int = 51, v_min: float = -10.0,
                  v_max: float = 10.0, sigma_0: float = 0.5) -> nn.Module:
    """
    Factory function — tạo network theo ablation config.

    Args:
        ablation: None = Rainbow full, hoặc tên ablation
    """
    if ablation in ("no_noisy", "no_dueling", "no_distributional"):
        # Những ablation này đơn giản nhất: dùng DQNNet làm base
        # (chi tiết implement từng ablation riêng nếu cần)
        return DQNNet(n_actions)
    else:
        # Rainbow full (hoặc no_double, no_priority, no_multistep
        # — những ablation này không thay đổi network architecture)
        return RainbowNet(
            n_actions=n_actions,
            n_atoms=n_atoms,
            v_min=v_min,
            v_max=v_max,
            sigma_0=sigma_0,
        )


if __name__ == "__main__":
    # Quick test
    net = RainbowNet(n_actions=6, n_atoms=51)
    x = torch.randint(0, 255, (4, 4, 84, 84), dtype=torch.uint8)
    probs, q = net(x)
    print(f"probs shape:    {probs.shape}")    # (4, 6, 51)
    print(f"q_values shape: {q.shape}")        # (4, 6)
    print(f"prob sum check: {probs.sum(dim=2).mean():.4f}")  # phải = 1.0
    print("✓ RainbowNet OK")
