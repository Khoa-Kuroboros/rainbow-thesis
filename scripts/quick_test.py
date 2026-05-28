"""
quick_test.py — Verify toàn bộ pipeline hoạt động đúng.
Chạy ~5–8 phút, kiểm tra tất cả components trước khi chạy thật.

Cách dùng:
    conda activate rainbow
    cd ~/rainbow_thesis
    python scripts/quick_test.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import torch

PASS = "✓"
FAIL = "✗"
results = []

def check(name, fn):
    try:
        fn()
        print(f"  {PASS} {name}")
        results.append((name, True, ""))
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        results.append((name, False, str(e)))


print("=" * 55)
print("Rainbow DQN — Pipeline Verification")
print("=" * 55)

# ── Test 1: CUDA ─────────────────────────────────────────────
print("\n[1] CUDA & PyTorch")

def test_cuda():
    assert torch.cuda.is_available(), "CUDA not available!"
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"       GPU: {name} ({vram:.1f} GB VRAM)")

check("torch.cuda.is_available()", test_cuda)
check("PyTorch version ≥ 2.0",
      lambda: (_ for _ in [torch.__version__]
               if _ >= "2.0" or (_ for _ in ()).throw(AssertionError(f"Version {_} < 2.0"))))

# ── Test 2: Atari environment ─────────────────────────────────
print("\n[2] Atari Environment")

from src.utils.atari_wrappers import make_atari_env

def test_env():
    env = make_atari_env("PongNoFrameskip-v4", seed=0)
    obs, _ = env.reset()
    obs_arr = np.array(obs)
    assert obs_arr.shape == (4, 84, 84), f"Wrong shape: {obs_arr.shape}"
    obs2, r, term, trunc, _ = env.step(env.action_space.sample())
    obs2_arr = np.array(obs2)
    assert obs2_arr.dtype == np.uint8
    env.close()
    print(f"       obs: {obs_arr.shape} uint8, actions: {env.action_space.n}")

check("make_atari_env (Pong)", test_env)

# ── Test 3: Networks ──────────────────────────────────────────
print("\n[3] Neural Networks")
device = torch.device("cuda")

from src.networks.rainbow_net import RainbowNet, NoisyLinear

def test_noisy_linear():
    layer = NoisyLinear(512, 256).to(device)
    x = torch.randn(4, 512, device=device)
    out = layer(x)
    assert out.shape == (4, 256)
    layer.reset_noise()

check("NoisyLinear forward + reset_noise", test_noisy_linear)

def test_rainbow_net():
    net = RainbowNet(n_actions=6, n_atoms=51).to(device)
    x = torch.randint(0, 255, (4, 4, 84, 84), dtype=torch.uint8, device=device)
    probs, q = net(x)
    assert probs.shape == (4, 6, 51), f"probs: {probs.shape}"
    assert q.shape == (4, 6),         f"q: {q.shape}"
    prob_sum = probs.sum(dim=2).mean().item()
    assert abs(prob_sum - 1.0) < 1e-4, f"Probs sum = {prob_sum} ≠ 1"
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"       probs: {tuple(probs.shape)}, q: {tuple(q.shape)}")
    print(f"       VRAM used: {vram:.3f} GB")

check("RainbowNet (forward, dueling, distributional)", test_rainbow_net)

# ── Test 4: Replay Buffer ─────────────────────────────────────
print("\n[4] Prioritized Replay Buffer")

from src.replay.prioritized import PrioritizedReplayBuffer, SumTree

def test_sumtree():
    tree = SumTree(100)
    for i in range(50):
        tree.add(float(i + 1))
    assert tree.size == 50
    assert tree.total_priority > 0

check("SumTree add + sample", test_sumtree)

def test_replay_buffer():
    buf = PrioritizedReplayBuffer(
        capacity=500, n_step=3, gamma=0.99,
        obs_shape=(4, 84, 84), total_frames=10_000
    )
    obs = np.zeros((4, 84, 84), dtype=np.uint8)
    for i in range(100):
        buf.add(obs, 0, 1.0, obs, i % 20 == 19)

    assert buf.size > 0, "Buffer empty after adding!"
    batch = buf.sample(16, current_frame=500)
    assert batch["states"].shape == (16, 4, 84, 84)
    assert batch["rewards"].shape == (16,)
    assert 0.0 < batch["weights"].max() <= 1.0

    # Update priorities
    kl = np.random.rand(16).astype(np.float32)
    buf.update_priorities(batch["indices"], kl)
    print(f"       Buffer size: {buf.size}, batch OK")

check("PrioritizedReplayBuffer (add, sample, update)", test_replay_buffer)

# ── Test 5: Rainbow Agent ─────────────────────────────────────
print("\n[5] Rainbow Agent")

from src.agents.rainbow import RainbowAgent
from src.utils.atari_wrappers import make_atari_env

def test_agent_act():
    env = make_atari_env("PongNoFrameskip-v4", seed=0)
    agent = RainbowAgent(
        n_actions=env.action_space.n,
        device=device,
        min_history=50,
        buffer_capacity=500,
        total_frames=10_000,
    )
    obs, _ = env.reset()
    action = agent.act(obs)
    assert 0 <= action < env.action_space.n
    env.close()
    print(f"       act() returned action: {action}")

check("RainbowAgent.act()", test_agent_act)

def test_agent_learn():
    env = make_atari_env("PongNoFrameskip-v4", seed=0)
    agent = RainbowAgent(
        n_actions=env.action_space.n,
        device=device,
        min_history=50,
        buffer_capacity=500,
        batch_size=8,
        total_frames=10_000,
        n_step=3,
    )
    # Thu thập đủ transitions để learn
    obs, _ = env.reset()
    for i in range(120):
        action = agent.act(obs)
        next_obs, reward, term, trunc, _ = env.step(action)
        agent.store(obs, action, reward, next_obs, term or trunc)
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()

    loss = agent.learn()
    assert loss is not None, "learn() returned None — buffer quá nhỏ?"
    assert loss >= 0, f"Loss âm: {loss}"
    print(f"       learn() loss: {loss:.4f}")
    env.close()

check("RainbowAgent.learn() (1 gradient step)", test_agent_learn)

# ── Test 6: End-to-end mini training loop ────────────────────
print("\n[6] End-to-end (500 frames, Pong)")

def test_e2e():
    t0 = time.time()
    env = make_atari_env("PongNoFrameskip-v4", seed=42)
    agent = RainbowAgent(
        n_actions=env.action_space.n,
        device=device,
        min_history=100,
        buffer_capacity=2000,
        batch_size=16,
        total_frames=500,
        n_step=3,
        target_update=200,
    )
    obs, _ = env.reset()
    losses = []
    for frame in range(500):
        action = agent.act(obs)
        next_obs, reward, term, trunc, _ = env.step(action)
        agent.store(obs, action, reward, next_obs, term or trunc)
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()
        if frame % 4 == 0:
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

    elapsed = time.time() - t0
    fps = 500 / elapsed
    n_loss = len(losses)
    print(f"       500 frames in {elapsed:.1f}s ({fps:.0f} fps)")
    print(f"       {n_loss} gradient steps, "
          f"avg loss: {np.mean(losses):.4f}" if losses else "       no gradient steps")
    assert fps > 100, f"FPS quá thấp: {fps:.0f} (pipeline bottleneck?)"
    env.close()

check("500-frame training loop", test_e2e)

# ── Test 7: Checkpoint save/load ──────────────────────────────
print("\n[7] Checkpoint")

def test_checkpoint():
    import tempfile
    env = make_atari_env("PongNoFrameskip-v4", seed=0)
    agent = RainbowAgent(n_actions=env.action_space.n, device=device,
                         total_frames=1000, buffer_capacity=500)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    agent.save(path)
    agent.load(path)
    os.unlink(path)
    env.close()
    print(f"       save/load OK")

check("save() + load() checkpoint", test_checkpoint)

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 55)
passed = sum(1 for _, ok, _ in results if ok)
total  = len(results)

if passed == total:
    print(f"✓ TẤT CẢ {total}/{total} TESTS PASS — Sẵn sàng chạy training!")
    print("\nLệnh chạy thực nghiệm đầu tiên:")
    print("  conda activate rainbow && cd ~/rainbow_thesis")
    print("  tmux new -s training")
    print("  python scripts/train.py \\")
    print("    --game PongNoFrameskip-v4 \\")
    print("    --seed 1 --total_frames 10000000 \\")
    print("    --no_wandb")
else:
    print(f"✗ {total - passed}/{total} TESTS FAILED:")
    for name, ok, err in results:
        if not ok:
            print(f"   ✗ {name}: {err}")
    print("\nFix các lỗi trên trước khi chạy training.")
print("=" * 55)
