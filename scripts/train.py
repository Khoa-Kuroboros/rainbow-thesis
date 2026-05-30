"""
train.py — Entry point cho tất cả experiments.

Cách dùng:
    # Rainbow full trên Pong
    python scripts/train.py --game PongNoFrameskip-v4 --agent rainbow --seed 1

    # Ablation: bỏ priority
    python scripts/train.py --game BreakoutNoFrameskip-v4 --agent rainbow \
        --ablation no_priority --seed 2

    # Resume từ checkpoint
    python scripts/train.py --game PongNoFrameskip-v4 --agent rainbow --seed 1 \
        --resume checkpoints/Pong_rainbow_s1_5000000.pt
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import json
import numpy as np
import torch
from collections import deque

import wandb

from src.utils.atari_wrappers import make_atari_env
from src.agents.rainbow import RainbowAgent


# ─────────────────────────────────────────────
# Human / Random baseline scores (để tính human-normalized score)
# Nguồn: Table 3, Mnih et al. 2015 + Wang et al. 2016
# ─────────────────────────────────────────────
HUMAN_SCORES = {
    "PongNoFrameskip-v4":         9.3,
    "BreakoutNoFrameskip-v4":   30.5,
    "SpaceInvadersNoFrameskip-v4": 1669.0,
    "SeaquestNoFrameskip-v4":    42054.7,
    "MontezumaRevengeNoFrameskip-v4": 4753.3,
}
RANDOM_SCORES = {
    "PongNoFrameskip-v4":        -20.7,
    "BreakoutNoFrameskip-v4":      1.7,
    "SpaceInvadersNoFrameskip-v4": 148.0,
    "SeaquestNoFrameskip-v4":      68.4,
    "MontezumaRevengeNoFrameskip-v4": 0.0,
}


def human_normalized_score(game: str, score: float) -> float:
    """(score - random) / (human - random) * 100 (%)"""
    h = HUMAN_SCORES.get(game, 1.0)
    r = RANDOM_SCORES.get(game, 0.0)
    if h == r:
        return 0.0
    return (score - r) / (h - r) * 100.0


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate(agent: RainbowAgent, game: str, n_episodes: int = 10,
             seed: int = 42) -> dict:
    """Chạy n_episodes greedy, trả về dict kết quả."""
    env = make_atari_env(game, clip_rewards=False,
                         episodic_life=False, seed=seed + 1000)
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.act(obs, eval_mode=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
    env.close()

    return {
        "mean":   np.mean(rewards),
        "std":    np.std(rewards),
        "min":    np.min(rewards),
        "max":    np.max(rewards),
        "human_norm": human_normalized_score(game, np.mean(rewards)),
    }


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

def train(args):
    # ── Setup ────────────────────────────────────────────
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True  # tăng tốc convolution

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Run name ─────────────────────────────────────────
    game_short = args.game.replace("NoFrameskip-v4", "")
    ablation_tag = args.ablation or "full"
    run_name = f"{game_short}_{ablation_tag}_s{args.seed}"

    # ── Wandb ─────────────────────────────────────────────
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
        tags=[game_short, ablation_tag, f"seed{args.seed}"],
        mode="online" if not args.no_wandb else "disabled",
    )

    # ── Checkpoint dir ────────────────────────────────────
    ckpt_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # ── Environment ───────────────────────────────────────
    env = make_atari_env(args.game, clip_rewards=True,
                         episodic_life=True, seed=args.seed)
    n_actions = env.action_space.n
    print(f"Game: {args.game} | Actions: {n_actions} | Ablation: {ablation_tag}")

    # ── Agent ─────────────────────────────────────────────
    agent = RainbowAgent(
        n_actions       = n_actions,
        device          = device,
        n_atoms         = args.n_atoms,
        v_min           = args.v_min,
        v_max           = args.v_max,
        sigma_0         = args.sigma_0,
        lr              = args.lr,
        adam_eps        = args.adam_eps,
        batch_size      = args.batch_size,
        target_update   = args.target_update,
        discount        = args.discount,
        min_history     = args.min_history,
        buffer_capacity = args.buffer_size,
        n_step          = args.n_step,
        priority_omega  = args.priority_omega,
        beta_start      = args.beta_start,
        beta_end        = args.beta_end,
        total_frames    = args.total_frames,
        ablation        = args.ablation,
    )

    # ── Resume từ checkpoint ──────────────────────────────
    start_frame = 0
    if args.resume and os.path.exists(args.resume):
        agent.load(args.resume)
        start_frame = agent.frame_count
        print(f"Resumed from frame {start_frame:,}")

    # ── Training variables ────────────────────────────────
    obs, _ = env.reset()
    ep_reward  = 0.0
    ep_count   = 0
    ep_start   = time.time()
    recent_rewards = deque(maxlen=100)

    loss_sum    = 0.0
    loss_count  = 0
    fps_timer   = time.time()
    fps_frames  = 0

    best_eval_score = -float("inf")

    log_path = os.path.join(args.log_dir, f"{run_name}.jsonl")

    print(f"\n{'='*60}")
    print(f"Training: {run_name}")
    print(f"Total frames: {args.total_frames:,} | "
          f"Batch: {args.batch_size} | n_step: {args.n_step}")
    print(f"{'='*60}\n")

    # ── Main loop ─────────────────────────────────────────
    for frame in range(start_frame, args.total_frames):

        # 1. Act
        action = agent.act(obs)

        # 2. Step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward

        # 3. Store
        agent.store(obs, action, reward, next_obs, done)
        obs = next_obs

        # 4. Learn (every train_freq steps)
        if frame % args.train_freq == 0:
            loss = agent.learn()
            if loss is not None:
                loss_sum   += loss
                loss_count += 1

        # 5. Episode done
        if done:
            ep_count += 1
            recent_rewards.append(ep_reward)
            ep_dur = time.time() - ep_start

            # FPS
            fps_frames += 1
            if fps_frames % 50 == 0:
                fps = fps_frames / (time.time() - fps_timer + 1e-8)
                fps_frames = 0
                fps_timer  = time.time()
            else:
                fps = 0

            # Log episode
            avg_loss = loss_sum / max(loss_count, 1)
            mean100  = np.mean(recent_rewards) if recent_rewards else 0.0
            hn_score = human_normalized_score(args.game, ep_reward)

            wandb.log({
                "episode/reward":       ep_reward,
                "episode/reward_100":   mean100,
                "episode/human_norm":   hn_score,
                "episode/length":       ep_dur,
                "train/loss":           avg_loss,
                "train/buffer_size":    agent.replay.size,
                "train/learn_steps":    agent.learn_steps,
                "frame": frame,
            })

            # Console print mỗi 50 episodes
            if ep_count % 50 == 0:
                gpu_mem = torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else 0
                print(
                    f"[{frame:>9,}] "
                    f"Ep {ep_count:>5} | "
                    f"R: {ep_reward:>7.1f} | "
                    f"Avg100: {mean100:>7.1f} | "
                    f"HN: {hn_score:>6.1f}% | "
                    f"Loss: {avg_loss:.4f} | "
                    f"VRAM: {gpu_mem:.1f}GB"
                )

            # Reset episode
            loss_sum   = 0.0
            loss_count = 0
            ep_reward  = 0.0
            ep_start   = time.time()
            obs, _     = env.reset()

        # 6. Evaluation
        if frame > 0 and frame % args.eval_freq == 0:
            print(f"\n[Eval @ {frame:,} frames]")
            eval_result = evaluate(agent, args.game,
                                   args.eval_episodes, args.seed)

            print(f"  Mean: {eval_result['mean']:.1f} ± {eval_result['std']:.1f} | "
                  f"HN: {eval_result['human_norm']:.1f}%")

            wandb.log({
                "eval/mean_reward": eval_result["mean"],
                "eval/std_reward":  eval_result["std"],
                "eval/human_norm":  eval_result["human_norm"],
                "frame": frame,
            })

            # Save best model
            if eval_result["mean"] > best_eval_score:
                best_eval_score = eval_result["mean"]
                best_path = os.path.join(ckpt_dir, "best.pt")
                agent.save(best_path)
                print(f"  ✓ New best ({best_eval_score:.1f}) saved to {best_path}")

            # Log to file
            with open(log_path, "a") as f:
                json.dump({"frame": frame, **eval_result}, f)
                f.write("\n")

        # 7. Checkpoint
        if frame > 0 and frame % args.checkpoint_freq == 0:
            ckpt_path = os.path.join(ckpt_dir, f"frame_{frame}.pt")
            agent.save(ckpt_path)
            print(f"[Checkpoint] Saved: {ckpt_path}")

    # ── Done ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training complete! Best eval score: {best_eval_score:.1f}")
    print(f"{'='*60}")

    # Final eval
    print("\n[Final Evaluation — 30 episodes]")
    final = evaluate(agent, args.game, n_episodes=30, seed=args.seed + 9999)
    print(f"Final score: {final['mean']:.1f} ± {final['std']:.1f} | "
          f"HN: {final['human_norm']:.1f}%")

    wandb.log({
        "final/mean_reward": final["mean"],
        "final/human_norm":  final["human_norm"],
    })

    # Save final
    agent.save(os.path.join(ckpt_dir, "final.pt"))
    env.close()
    wandb.finish()


# ─────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Rainbow DQN on Atari")

    # Environment
    p.add_argument("--game",    default="PongNoFrameskip-v4",
                   help="Atari game (NoFrameskip-v4 variant)")
    p.add_argument("--seed",    type=int, default=1)

    # Agent variant
    p.add_argument("--agent",   default="rainbow", choices=["rainbow", "dqn"])
    p.add_argument("--ablation", default=None,
                   choices=[None, "no_double", "no_priority", "no_dueling",
                             "no_multistep", "no_distributional", "no_noisy"],
                   help="Tên ablation — bỏ đúng component đó")

    # Hyperparameters (defaults = Table 1 của paper)
    p.add_argument("--total_frames",   type=int,   default=10_000_000)
    p.add_argument("--n_atoms",        type=int,   default=51)
    p.add_argument("--v_min",          type=float, default=-10.0)
    p.add_argument("--v_max",          type=float, default=10.0)
    p.add_argument("--sigma_0",        type=float, default=0.5)
    p.add_argument("--lr",             type=float, default=6.25e-5)
    p.add_argument("--adam_eps",       type=float, default=1.5e-4)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--target_update",  type=int,   default=32_000)
    p.add_argument("--discount",       type=float, default=0.99)
    p.add_argument("--min_history",    type=int,   default=80_000)
    p.add_argument("--buffer_size",    type=int,   default=250_000)
    p.add_argument("--n_step",         type=int,   default=3)
    p.add_argument("--priority_omega", type=float, default=0.5)
    p.add_argument("--beta_start",     type=float, default=0.4)
    p.add_argument("--beta_end",       type=float, default=1.0)
    p.add_argument("--train_freq",     type=int,   default=4)

    # Training ops
    p.add_argument("--eval_freq",       type=int,  default=250_000)
    p.add_argument("--eval_episodes",   type=int,  default=10)
    p.add_argument("--checkpoint_freq", type=int,  default=500_000)
    p.add_argument("--checkpoint_dir",  default="checkpoints")
    p.add_argument("--log_dir",         default="logs")
    p.add_argument("--resume",          default=None,
                   help="Path tới .pt file để resume")

    # Logging
    p.add_argument("--wandb_project", default="rainbow-thesis")
    p.add_argument("--no_wandb",      action="store_true",
                   help="Tắt wandb (chạy offline/debug)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
