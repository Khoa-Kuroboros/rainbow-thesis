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

def record_episode(agent, game, seed=0, render_size=(420, 320), max_steps=1500):
    gym.register_envs(ale_py)
    env = gym.make(game, render_mode="rgb_array")
    env.reset(seed=seed)
    env_w = FrameStack(WarpFrame(MaxAndSkipEnv(NoopResetEnv(env, 30), 4)), 4)
    obs, _ = env_w.reset()
    frames, total = [], 0.0
    done = False
    step = 0
    saved_frames = []  # chỉ giữ subsample cho GIF, không giữ full cho MP4
    while not done and step < max_steps:
        step += 1
        raw = env.render()
        if raw is not None:
            try:
                from PIL import Image
                raw = np.array(Image.fromarray(raw).resize(render_size, Image.NEAREST))
            except:
                pass
            frames.append(raw)
        action = agent.act(obs, eval_mode=True)
        obs, r, term, trunc, _ = env_w.step(action)
        total += r
        done = term or trunc
    env_w.close(); env.close()
    return frames, total

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game",       required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--agent_name", default="Rainbow")
    p.add_argument("--episodes",   type=int, default=2)
    p.add_argument("--output",     default="results/videos")
    p.add_argument("--n_step",     type=int, default=10)
    p.add_argument("--ablation",   default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game_short = args.game.replace("NoFrameskip-v4","")
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

    all_rewards, all_frames = [], []
    for ep in range(args.episodes):
        print(f"Recording episode {ep+1}/{args.episodes}...")
        frames, reward = record_episode(agent, args.game, seed=42+ep)
        all_rewards.append(reward)
        all_frames.extend(frames)
        print(f"  Reward: {reward:.0f} | Frames: {len(frames)}")

        mp4 = os.path.join(args.output, f"{game_short}_{args.agent_name}_ep{ep+1}.mp4")
        imageio.mimsave(mp4, frames, fps=30)
        print(f"  Saved: {mp4}")

    n_total = len(all_frames)
    if n_total > 200:
        idx = np.linspace(0, n_total - 1, 200, dtype=int)
        gif_frames = [all_frames[i] for i in idx]
    else:
        gif_frames = all_frames
    try:
        from PIL import Image
        gif_frames = [np.array(Image.fromarray(f).resize(
            (f.shape[1]//2, f.shape[0]//2))) for f in gif_frames]
    except:
        pass
    gif = os.path.join(args.output, f"{game_short}_{args.agent_name}_demo.gif")
    imageio.mimsave(gif, gif_frames, fps=15, loop=0)
    print(f"\nGIF saved: {gif}")
    print(f"Mean reward: {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")

if __name__ == "__main__":
    main()
