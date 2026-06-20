import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
import imageio
import gymnasium as gym
import ale_py
from src.agents.rainbow import RainbowAgent
from src.utils.atari_wrappers import (
    NoopResetEnv, MaxAndSkipEnv, WarpFrame, FrameStack
)


def record_episode(agent, game, seed=0, render_size=(420, 320), max_steps=3000,
                   mp4_path=None, fps=30):
    """
    Quay 1 episode, ghi MP4 streaming (không giữ frames trong RAM).
    Fix: tự bấm FIRE sau khi mất mạng (agent chưa học việc này khi train
    vì EpisodicLifeEnv + FireResetEnv tự làm sẵn lúc training).
    """
    np.random.seed(seed)
    gym.register_envs(ale_py)
    env = gym.make(game, render_mode="rgb_array")
    env.reset(seed=seed)
    env_w = FrameStack(WarpFrame(MaxAndSkipEnv(NoopResetEnv(env, 30), 4)), 4)
    obs, _ = env_w.reset()

    has_fire = "FIRE" in env.unwrapped.get_action_meanings()
    fire_action = env.unwrapped.get_action_meanings().index("FIRE") if has_fire else None
    try:
        lives = env.unwrapped.ale.lives()
    except Exception:
        lives = None

    total = 0.0
    done = False
    step = 0

    writer = imageio.get_writer(mp4_path, fps=fps) if mp4_path else None
    gif_frames = []

    while not done and step < max_steps:
        step += 1
        raw = env.render()
        if raw is not None:
            try:
                from PIL import Image
                raw = np.array(Image.fromarray(raw).resize(render_size, Image.NEAREST))
            except Exception:
                pass
            if writer is not None:
                writer.append_data(raw)
            if step % 5 == 0:
                gif_frames.append(raw)

        action = agent.act(obs, eval_mode=True)
        obs, r, term, trunc, _ = env_w.step(action)
        total += r
        done = term or trunc

        # Fix: phát hiện mất mạng -> tự bấm FIRE để launch lại bóng
        if has_fire and lives is not None and not done:
            try:
                new_lives = env.unwrapped.ale.lives()
            except Exception:
                new_lives = lives
            if new_lives < lives and new_lives > 0:
                obs, r2, term2, trunc2, _ = env_w.step(fire_action)
                total += r2
                done = term2 or trunc2
                step += 1
                raw2 = env.render()
                if raw2 is not None and writer is not None:
                    try:
                        from PIL import Image
                        raw2 = np.array(Image.fromarray(raw2).resize(render_size, Image.NEAREST))
                    except Exception:
                        pass
                    writer.append_data(raw2)
            lives = new_lives

    if writer is not None:
        writer.close()
    env_w.close()
    env.close()
    return gif_frames, total, step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game",       required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--agent_name", default="Rainbow")
    p.add_argument("--episodes",   type=int, default=2)
    p.add_argument("--output",     default="results/videos")
    p.add_argument("--n_step",     type=int, default=10)
    p.add_argument("--ablation",   default=None)
    p.add_argument("--max_steps",  type=int, default=5000)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game_short = args.game.replace("NoFrameskip-v4", "")
    os.makedirs(args.output, exist_ok=True)

    gym.register_envs(ale_py)
    env_tmp   = gym.make(args.game)
    n_actions = env_tmp.action_space.n
    env_tmp.close()

    agent = RainbowAgent(n_actions=n_actions, device=device,
                         total_frames=5_000_000, buffer_capacity=10,
                         n_step=args.n_step, ablation=args.ablation)
    agent.load(args.checkpoint)
    agent.online_net.eval()

    all_rewards = []
    best_frames, best_reward = None, -1e9

    for ep in range(args.episodes):
        print(f"Recording episode {ep+1}/{args.episodes}...")
        mp4 = os.path.join(args.output, f"{game_short}_{args.agent_name}_ep{ep+1}.mp4")
        gif_frames, reward, n_steps = record_episode(
            agent, args.game, seed=42+ep,
            max_steps=args.max_steps, mp4_path=mp4
        )
        all_rewards.append(reward)
        print(f"  Reward: {reward:.0f} | Steps: {n_steps}")
        print(f"  Saved: {mp4}")

        if reward > best_reward:
            best_reward = reward
            best_frames = gif_frames
        else:
            del gif_frames

    all_frames = best_frames

    gif = os.path.join(args.output, f"{game_short}_{args.agent_name}_demo.gif")
    try:
        from PIL import Image
        resized = [np.array(Image.fromarray(f).resize(
            (f.shape[1]//2, f.shape[0]//2))) for f in all_frames]
    except Exception:
        resized = all_frames
    imageio.mimsave(gif, resized, fps=10, loop=0)
    print(f"\nGIF saved: {gif} (best episode, reward={best_reward:.0f})")
    print(f"Mean reward: {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")


if __name__ == "__main__":
    main()
