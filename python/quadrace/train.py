"""Train a PPO policy on a task of the Rust simulator (milestone 3: the hover task).

    python -m quadrace.train --config configs/ppo.yaml --task configs/task_hover.yaml --run-name hover1

Everything a run produces goes to `runs/<run-name>/` (not committed): `meta.json` (settings, git
commit, seed), `metrics.jsonl` (one line per update), TensorBoard logs in `tb/`, and checkpoints.
Watch it live with  `tensorboard --logdir runs`.

A run of more than an hour should be agreed with the owner first (CLAUDE.md): the printed speed
and the total-timesteps setting tell you in advance how long it will take.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import yaml

from quadrace import ppo
from quadrace.vec_env import QuadVecEnv

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_ppo_config(section: dict, overrides: dict) -> ppo.PPOConfig:
    """A PPOConfig from the `ppo:` section of the YAML, with command-line overrides applied.
    Unknown keys are an error, so a typo in the YAML cannot silently change nothing."""
    settings = {**section, **{k: v for k, v in overrides.items() if v is not None}}
    known = {f.name for f in dataclasses.fields(ppo.PPOConfig)}
    unknown = sorted(set(settings) - known)
    if unknown:
        raise ValueError(f"unknown PPO setting(s) in the config: {unknown}. Known: {sorted(known)}")
    if "hidden_sizes" in settings:
        settings["hidden_sizes"] = tuple(settings["hidden_sizes"])
    return ppo.PPOConfig(**settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on a task of the Rust simulator")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "ppo.yaml")
    parser.add_argument("--drone", type=Path, default=REPO_ROOT / "configs" / "drone.yaml")
    parser.add_argument("--task", type=Path, default=REPO_ROOT / "configs" / "task_hover.yaml")
    parser.add_argument("--run-name", default=None, help="folder name under runs/ (default: task, seed and time)")
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto, cpu or cuda")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--save-every", type=int, default=None, help="checkpoint every this many updates")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    config = build_ppo_config(
        cfg["ppo"],
        {
            "total_timesteps": args.total_timesteps,
            "seed": args.seed,
            "device": args.device,
            "learning_rate": args.learning_rate,
            "ent_coef": args.ent_coef,
            "save_every_updates": args.save_every,
        },
    )
    num_envs = args.num_envs or cfg["env"]["num_envs"]
    drone_cfg, track_cfg = load_yaml(args.drone), load_yaml(args.task)

    run_name = args.run_name or f"{track_cfg['task']}-seed{config.seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = args.runs_dir / run_name

    env = QuadVecEnv(num_envs, drone_cfg, track_cfg, seed=config.seed)
    batch = num_envs * config.rollout_len
    print(f"task {track_cfg['task']}, {num_envs} environments, batch {batch:,}, "
          f"{config.total_timesteps:,} steps = {config.total_timesteps // batch} updates, device {ppo.resolve_device(config.device)}")
    print(f"run folder: {run_dir}")

    start = time.time()
    result = ppo.train(env, config, run_dir=run_dir, verbose=True)
    seconds = time.time() - start

    last = result.history[-1]
    summary = {
        "run_name": run_name,
        "seconds": seconds,
        "updates": len(result.history),
        "global_step": last["global_step"],
        "steps_per_second": last["global_step"] / seconds,
        "final_ep_return_mean": last["ep_return_mean"],
        "final_ep_length_mean": last["ep_length_mean"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    checkpoints = sorted(run_dir.glob("checkpoint_*.pt"))
    print(f"\nfinished in {seconds / 60:.1f} min ({summary['steps_per_second']:.0f} steps/s); "
          f"final mean return {last['ep_return_mean']:.1f}, mean episode length {last['ep_length_mean']:.0f}")
    print(f"last checkpoint: {checkpoints[-1]}")
    print(f"evaluate it with:  python -m quadrace.evaluate --checkpoint {checkpoints[-1]}")


if __name__ == "__main__":
    main()
