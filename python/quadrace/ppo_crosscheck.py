"""Cross-check the hand-written PPO against Stable-Baselines3 (SB3) on CartPole and Pendulum.

Handoff section 9: before touching the drone, check the PPO on two classic problems and compare
it with a trusted implementation. If both learn about equally well with the same settings, a later
failure on the drone is the task or the reward, not a bug in `ppo.py`.

Both implementations get the same settings (the presets below), the same number of environments,
the same seeds, and the same amount of experience. The comparison is of the learning curve: the
mean return of the last 100 finished episodes, against environment steps.

    python -m quadrace.ppo_crosscheck --env CartPole-v1 --seeds 1 2 3
    python -m quadrace.ppo_crosscheck --env Pendulum-v1 --seeds 1 2 3 --only ours

gymnasium and stable-baselines3 are approved for this cross-check only (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from quadrace import ppo
from quadrace.vec_env import GymVecEnv

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Preset:
    """The settings both implementations use for one environment."""

    env_id: str
    num_envs: int
    rollout_len: int
    num_minibatches: int
    update_epochs: int
    gamma: float
    gae_lambda: float
    clip_coef: float
    learning_rate: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    hidden_sizes: tuple[int, ...]
    init_log_std: float
    normalize_obs: bool
    normalize_reward: bool
    total_timesteps: int
    solved_return: float  # a mean return at or above this counts as "learned"


PRESETS: dict[str, Preset] = {
    # CartPole: CleanRL's PPO settings for CartPole-v1. A pole on a cart; the best return is 500.
    "CartPole-v1": Preset(
        env_id="CartPole-v1", num_envs=4, rollout_len=128, num_minibatches=4, update_epochs=4,
        gamma=0.99, gae_lambda=0.95, clip_coef=0.2, learning_rate=2.5e-4, ent_coef=0.01, vf_coef=0.5,
        max_grad_norm=0.5, hidden_sizes=(64, 64), init_log_std=-0.5, normalize_obs=False,
        normalize_reward=False, total_timesteps=500_000, solved_return=475.0,
    ),
    # Pendulum: a rod on a pivot with a weak motor, swung up and balanced. Returns are negative;
    # closer to zero is better. Settings follow the SB3 zoo's Pendulum recipe (short horizon,
    # observation and reward normalization), without its state-dependent exploration.
    "Pendulum-v1": Preset(
        env_id="Pendulum-v1", num_envs=8, rollout_len=256, num_minibatches=8, update_epochs=10,
        gamma=0.9, gae_lambda=0.95, clip_coef=0.2, learning_rate=1e-3, ent_coef=0.0, vf_coef=0.5,
        max_grad_norm=0.5, hidden_sizes=(64, 64), init_log_std=-0.5, normalize_obs=True,
        normalize_reward=True, total_timesteps=1_000_000, solved_return=-250.0,
    ),
}


def run_ours(preset: Preset, seed: int) -> dict:
    config = ppo.PPOConfig(
        total_timesteps=preset.total_timesteps, rollout_len=preset.rollout_len,
        num_minibatches=preset.num_minibatches, update_epochs=preset.update_epochs, gamma=preset.gamma,
        gae_lambda=preset.gae_lambda, clip_coef=preset.clip_coef, learning_rate=preset.learning_rate,
        anneal_lr=True, ent_coef=preset.ent_coef, vf_coef=preset.vf_coef, max_grad_norm=preset.max_grad_norm,
        hidden_sizes=preset.hidden_sizes, init_log_std=preset.init_log_std,
        normalize_obs=preset.normalize_obs, normalize_reward=preset.normalize_reward,
        seed=seed, device="cpu",
    )
    env = GymVecEnv(preset.env_id, preset.num_envs)
    start = time.time()
    result = ppo.train(env, config)
    seconds = time.time() - start
    curve = [(m["global_step"], m["ep_return_mean"]) for m in result.history]
    eval_env = GymVecEnv(preset.env_id, 10)
    final_eval = ppo.evaluate(result.policy, eval_env, episodes=50, deterministic=True, seed=10_000 + seed)
    return {"curve": curve, "seconds": seconds, "eval_return_mean": float(final_eval.mean())}


def run_sb3(preset: Preset, seed: int) -> dict:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    class CurveCallback(BaseCallback):
        """Records the mean return of the last 100 episodes at the end of every rollout."""

        def __init__(self) -> None:
            super().__init__()
            self.curve: list[tuple[int, float]] = []

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            if len(self.model.ep_info_buffer) > 0:
                mean = float(np.mean([episode["r"] for episode in self.model.ep_info_buffer]))
                self.curve.append((self.model.num_timesteps, mean))

    venv = make_vec_env(preset.env_id, n_envs=preset.num_envs, seed=seed)
    if preset.normalize_obs or preset.normalize_reward:
        venv = VecNormalize(
            venv, norm_obs=preset.normalize_obs, norm_reward=preset.normalize_reward,
            gamma=preset.gamma, clip_obs=10.0, clip_reward=10.0,
        )
    base_lr = preset.learning_rate
    model = PPO(
        "MlpPolicy", venv, n_steps=preset.rollout_len,
        batch_size=preset.num_envs * preset.rollout_len // preset.num_minibatches,
        n_epochs=preset.update_epochs, gamma=preset.gamma, gae_lambda=preset.gae_lambda,
        clip_range=preset.clip_coef, learning_rate=lambda progress_remaining: base_lr * progress_remaining,
        ent_coef=preset.ent_coef, vf_coef=preset.vf_coef, max_grad_norm=preset.max_grad_norm,
        normalize_advantage=True,
        policy_kwargs={
            "net_arch": {"pi": list(preset.hidden_sizes), "vf": list(preset.hidden_sizes)},
            "log_std_init": preset.init_log_std,
        },
        seed=seed, device="cpu", verbose=0,
    )
    callback = CurveCallback()
    start = time.time()
    model.learn(total_timesteps=preset.total_timesteps, callback=callback)
    seconds = time.time() - start

    # Deterministic evaluation on fresh episodes, using the training observation statistics.
    returns = []
    env = gym.make(preset.env_id)
    for episode in range(50):
        obs, _ = env.reset(seed=10_000 + seed * 100 + episode)
        done, total = False, 0.0
        while not done:
            model_obs = venv.normalize_obs(obs[None]) if isinstance(venv, VecNormalize) and preset.normalize_obs else obs[None]
            action, _ = model.predict(model_obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action[0])
            total += reward
            done = terminated or truncated
        returns.append(total)
    return {"curve": callback.curve, "seconds": seconds, "eval_return_mean": float(np.mean(returns))}


def curve_at(curve: list[tuple[int, float]], step: int) -> float:
    """The curve's value at the last recorded point at or before `step`."""
    value = float("nan")
    for recorded_step, recorded_value in curve:
        if recorded_step <= step:
            value = recorded_value
    return value


def first_step_reaching(curve: list[tuple[int, float]], threshold: float) -> float:
    """The first recorded step at which the mean return is at or above `threshold`, else NaN."""
    for recorded_step, recorded_value in curve:
        if recorded_value >= threshold:
            return float(recorded_step)
    return float("nan")


def summarize(preset: Preset, runs: list[dict]) -> dict:
    fractions = [0.25, 0.5, 0.75, 1.0]
    summary = {}
    for fraction in fractions:
        step = int(preset.total_timesteps * fraction)
        values = [curve_at(run["curve"], step) for run in runs]
        summary[f"return_at_{int(fraction * 100)}pct"] = (float(np.nanmean(values)), float(np.nanstd(values)))
    reached = [first_step_reaching(run["curve"], preset.solved_return) for run in runs]
    summary["runs_reaching_threshold"] = int(np.sum(~np.isnan(reached)))
    summary["steps_to_threshold_mean"] = float(np.nanmean(reached)) if not np.all(np.isnan(reached)) else float("nan")
    evals = [run["eval_return_mean"] for run in runs]
    summary["deterministic_eval_return"] = (float(np.mean(evals)), float(np.std(evals)))
    summary["seconds_per_run"] = float(np.mean([run["seconds"] for run in runs]))
    return summary


def print_table(preset: Preset, results: dict[str, dict]) -> None:
    print(f"\n{preset.env_id}: {preset.total_timesteps:,} steps, threshold {preset.solved_return}")
    header = f"{'':28s}" + "".join(f"{name:>22s}" for name in results)
    print(header)
    rows = [
        ("return at 25% of budget", "return_at_25pct"),
        ("return at 50%", "return_at_50pct"),
        ("return at 75%", "return_at_75pct"),
        ("return at 100%", "return_at_100pct"),
        ("deterministic eval return", "deterministic_eval_return"),
    ]
    for label, key in rows:
        cells = "".join(f"{results[name]['summary'][key][0]:>14.1f} +- {results[name]['summary'][key][1]:<5.1f}" for name in results)
        print(f"{label:28s}{cells}")
    reached = "".join(f"{results[name]['summary']['runs_reaching_threshold']:>11d} of {len(results[name]['runs'])}   " for name in results)
    print(f"{'runs reaching threshold':28s}{reached}")
    steps = "".join(f"{results[name]['summary']['steps_to_threshold_mean']:>22,.0f}" for name in results)
    print(f"{'mean steps to threshold':28s}{steps}")
    secs = "".join(f"{results[name]['summary']['seconds_per_run']:>22.0f}" for name in results)
    print(f"{'seconds per run (CPU)':28s}{secs}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-check ppo.py against Stable-Baselines3")
    parser.add_argument("--env", choices=sorted(PRESETS), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--only", choices=["ours", "sb3"], default=None)
    parser.add_argument("--timesteps", type=int, default=None, help="override the preset's training budget")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "crosscheck")
    args = parser.parse_args()

    preset = PRESETS[args.env]
    if args.timesteps is not None:
        preset = replace(preset, total_timesteps=args.timesteps)

    runners = {"ours": run_ours, "sb3": run_sb3}
    if args.only:
        runners = {args.only: runners[args.only]}
    results: dict[str, dict] = {}
    for name, runner in runners.items():
        runs = []
        for seed in args.seeds:
            print(f"[{name}] {preset.env_id} seed {seed} ...", flush=True)
            run = runner(preset, seed)
            print(f"[{name}]   final eval return {run['eval_return_mean']:.1f}, {run['seconds']:.0f} s", flush=True)
            runs.append(run)
        results[name] = {"runs": runs, "summary": summarize(preset, runs)}

    print_table(preset, results)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{preset.env_id}.json"
    path.write_text(json.dumps({"preset": preset.__dict__, "seeds": args.seeds, "results": results}, indent=1, default=str), encoding="utf-8")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
