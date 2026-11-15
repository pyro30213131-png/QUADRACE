"""PPO (Proximal Policy Optimization), written by hand in PyTorch (milestone 3).

Read top to bottom: the pieces come in the order a training run uses them.

The algorithm, in one picture. One *update* does:

1. Collect: run the current policy in `num_envs` environments for `rollout_len` steps each.
2. Score: for every step, the advantage says how much better the action was than the critic
   expected (computed with GAE, below).
3. Improve: several passes (epochs) over that data in minibatches. The actor is nudged to make
   good actions likelier and bad ones less likely, but the "clip" stops any single action's
   probability from changing by more than about `clip_coef` (20%) in one update. The critic is
   trained to predict the return.
4. Throw the data away and collect fresh data with the improved policy (PPO is "on-policy").

Choices that follow the handoff (section 9) and the CleanRL / Stable-Baselines3 conventions:

- Separate actor and critic MLPs, tanh hidden units, orthogonal initial weights.
- Continuous actions are Gaussian with a state-independent learned log standard deviation.
  Actions are NOT squashed: the sampled value is what the log-probability is computed for, and
  the environment clips it to its limits. Discrete actions use a categorical distribution.
- Observations are normalized with a running mean and variance; the statistics are saved in
  every checkpoint. Rewards can optionally be scaled the same way (off by default).
- On a time-limit cut-off (`truncated`) the value is bootstrapped from `final_obs`; on a real
  ending (`terminated`) it is not.
- Advantages are normalized inside each minibatch. The value loss is a plain mean squared error.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal

from quadrace.vec_env import VecEnv

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


@dataclass
class PPOConfig:
    """Every PPO setting. Defaults are the handoff's starting values for the drone (section 9)."""

    total_timesteps: int  # environment steps to train for, summed over all environments
    rollout_len: int = 64  # steps per environment per update
    num_minibatches: int = 8
    update_epochs: int = 5
    gamma: float = 0.99  # discount
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    learning_rate: float = 3e-4
    anneal_lr: bool = True  # linear decay to zero over the run
    ent_coef: float = 0.001  # entropy bonus
    vf_coef: float = 0.5  # weight of the value loss
    max_grad_norm: float = 1.0
    hidden_sizes: tuple[int, ...] = (256, 256)
    init_log_std: float = -0.5  # continuous actions only
    normalize_obs: bool = True
    normalize_reward: bool = False
    clip_obs: float = 10.0
    clip_reward: float = 10.0
    normalize_advantage: bool = True
    adam_eps: float = 1e-5
    seed: int = 1
    device: str = "auto"  # "auto" uses the GPU if there is one
    save_every_updates: int = 50  # checkpoint cadence; the last update is always saved

    def validate(self, num_envs: int) -> None:
        """Raise ValueError if these settings cannot be run with `num_envs` environments."""
        batch = num_envs * self.rollout_len
        if self.num_minibatches < 1 or batch % self.num_minibatches != 0:
            raise ValueError(
                f"num_envs * rollout_len = {batch} must be divisible by num_minibatches = {self.num_minibatches}"
            )
        if self.total_timesteps < batch:
            raise ValueError(f"total_timesteps = {self.total_timesteps} is less than one batch of {batch}")
        for name in ("rollout_len", "update_epochs", "save_every_updates"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be between 0 and 1")


# --------------------------------------------------------------------------
# Running statistics (observation and reward normalization)
# --------------------------------------------------------------------------


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    """A float64 numpy copy of a tensor, wherever the tensor lives."""
    return torch.as_tensor(x).detach().cpu().numpy().astype(np.float64)


class RunningMeanStd:
    """Mean and variance of everything seen so far, updated batch by batch (float64)."""

    def __init__(self, shape: tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4  # a tiny prior so the first update does not divide by zero

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        self._merge(batch.mean(axis=0), batch.var(axis=0), batch.shape[0])

    def _merge(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        # The standard way to combine the statistics of two groups of numbers.
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total
        self.mean, self.var, self.count = new_mean, m2 / total, total

    def normalize(self, x: np.ndarray, clip: float) -> np.ndarray:
        """(x - mean) / std, clipped to [-clip, clip], as float32."""
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip).astype(np.float32)

    def state(self) -> dict:
        return {
            "mean": torch.as_tensor(self.mean),
            "var": torch.as_tensor(self.var),
            "count": float(self.count),
        }

    def load_state(self, state: dict) -> None:
        self.mean = _to_numpy(state["mean"])
        self.var = _to_numpy(state["var"])
        self.count = float(state["count"])


class RewardScaler:
    """Divides rewards by the running standard deviation of the discounted return.

    This keeps the value-function targets near unit size when raw rewards are large (Pendulum's
    are down to -16 per step). It is the same scheme as Stable-Baselines3's VecNormalize.
    """

    def __init__(self, num_envs: int, gamma: float, clip: float) -> None:
        self.returns = np.zeros(num_envs, dtype=np.float64)
        self.rms = RunningMeanStd(())
        self.gamma = gamma
        self.clip = clip

    def __call__(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        self.returns = self.returns * self.gamma + rewards
        self.rms.update(self.returns)
        scaled = np.clip(rewards / np.sqrt(self.rms.var + 1e-8), -self.clip, self.clip)
        self.returns[dones] = 0.0  # a new episode starts with a fresh return
        return scaled

    def state(self) -> dict:
        return {"returns": torch.as_tensor(self.returns), "rms": self.rms.state()}

    def load_state(self, state: dict) -> None:
        self.returns = _to_numpy(state["returns"])
        self.rms.load_state(state["rms"])


# --------------------------------------------------------------------------
# The actor and the critic
# --------------------------------------------------------------------------


def _layer(layer: nn.Linear, gain: float) -> nn.Linear:
    """Orthogonal initial weights (scaled by `gain`) and zero biases."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


def _mlp(in_dim: int, hidden_sizes: tuple[int, ...], out_dim: int, out_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for size in hidden_sizes:
        layers += [_layer(nn.Linear(last, size), math.sqrt(2.0)), nn.Tanh()]
        last = size
    layers.append(_layer(nn.Linear(last, out_dim), out_gain))
    return nn.Sequential(*layers)


class Agent(nn.Module):
    """Two separate networks: the actor (what to do) and the critic (how good is this state).

    The actor's last layer starts with a tiny gain (0.01), so the first actions are close to
    "the mean" and the policy starts out nearly neutral. The critic outputs one number.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_actions: int | None,
        hidden_sizes: tuple[int, ...],
        init_log_std: float,
    ) -> None:
        super().__init__()
        self.spec = {
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "n_actions": n_actions,
            "hidden_sizes": list(hidden_sizes),
            "init_log_std": init_log_std,
        }
        self.discrete = n_actions is not None
        actor_out = n_actions if n_actions is not None else act_dim
        self.actor = _mlp(obs_dim, hidden_sizes, actor_out, 0.01)
        self.critic = _mlp(obs_dim, hidden_sizes, 1, 1.0)
        # Continuous actions: one learned log standard deviation per action value, the same
        # for every state. Discrete actions have no such parameter.
        self.log_std = None if self.discrete else nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """The critic's estimate of the discounted return from `obs`, shape [batch]."""
        return self.critic(obs).squeeze(-1)

    def _distribution(self, obs: torch.Tensor) -> Categorical | Normal:
        out = self.actor(obs)
        if self.discrete:
            return Categorical(logits=out)
        return Normal(out, self.log_std.exp().expand_as(out))

    @staticmethod
    def _log_prob(dist: Categorical | Normal, action: torch.Tensor) -> torch.Tensor:
        log_prob = dist.log_prob(action)
        return log_prob if isinstance(dist, Categorical) else log_prob.sum(-1)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Choose actions for a batch of observations: (action, log_prob, value).

        With `deterministic=True` the noise is switched off: the most likely discrete action, or
        the mean of the Gaussian. That is how a trained policy is evaluated and flown.
        """
        dist = self._distribution(obs)
        if not deterministic:
            action = dist.sample()
        elif self.discrete:
            action = dist.logits.argmax(dim=-1)
        else:
            action = dist.mean
        return action, self._log_prob(dist, action), self.value(obs)

    def evaluate_actions(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Under the CURRENT networks: (log_prob of `action`, entropy, value) for a batch."""
        dist = self._distribution(obs)
        entropy = dist.entropy() if self.discrete else dist.entropy().sum(-1)
        return self._log_prob(dist, action), entropy, self.value(obs)


# --------------------------------------------------------------------------
# The maths that turns rewards into learning signals (pure functions, tested on their own)
# --------------------------------------------------------------------------


def bootstrap_truncated_rewards(
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    final_values: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Add `gamma * V(final_obs)` to the reward of every step that ended by TRUNCATION.

    A time-limit cut-off says nothing about how good the final state was, so the value target
    must still count the future the episode would have had. A termination (a crash) really
    does end the future, so nothing is added there.
    """
    cut_off = truncated & ~terminated
    return rewards + gamma * final_values * cut_off


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation. Shapes: rewards, values, dones are [T, N].

    `next_value` [N] is the critic's value of the observation AFTER the last step. `dones[t]`
    is 1 if the episode ended at step t, which cuts the link to the next step (the next
    observation belongs to a new episode).

    For each step the TD error is  delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)  and the
    advantage is the discounted sum  A_t = delta_t + (gamma * lambda) * delta_{t+1} + ...
    lambda = 0 gives the one-step error (low variance, more bias); lambda = 1 gives the full
    Monte Carlo return minus the baseline (no bias, more variance).

    Returns (advantages, returns) where returns = advantages + values is the critic's target.
    """
    steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(next_value)
    for t in reversed(range(steps)):
        not_done = 1.0 - dones[t]
        following_value = next_value if t == steps - 1 else values[t + 1]
        delta = rewards[t] + gamma * following_value * not_done - values[t]
        last_advantage = delta + gamma * gae_lambda * not_done * last_advantage
        advantages[t] = last_advantage
    return advantages, advantages + values


def minibatch_indices(batch_size: int, num_minibatches: int, generator: torch.Generator) -> list[torch.Tensor]:
    """A random split of range(batch_size) into `num_minibatches` equal parts: every sample once."""
    permutation = torch.randperm(batch_size, generator=generator)
    return list(permutation.chunk(num_minibatches))


def ppo_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    new_value: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    config: PPOConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """The PPO loss for one minibatch. Returns (loss to minimize, numbers to log).

    - ratio = pi_new(a|s) / pi_old(a|s): how much likelier the new policy makes the action
      that was actually taken. 1 means unchanged.
    - Policy loss: -advantage * ratio, but with the ratio clipped to [1 - clip, 1 + clip], and
      the worse (larger) of the two taken. So a good action (positive advantage) earns no
      extra credit for raising its probability beyond +clip, and a bad one is not punished
      extra beyond -clip. That limit is what keeps updates small and training stable.
    - Value loss: mean squared error between the critic and the return target.
    - Entropy bonus: subtracted, so the loss is lower for a more random policy. It keeps the
      policy from becoming deterministic too early.
    """
    log_ratio = new_log_prob - old_log_prob
    ratio = log_ratio.exp()

    if config.normalize_advantage:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    loss_plain = -advantage * ratio
    loss_clipped = -advantage * torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef)
    policy_loss = torch.max(loss_plain, loss_clipped).mean()
    value_loss = ((new_value - returns) ** 2).mean()
    entropy_mean = entropy.mean()
    loss = policy_loss - config.ent_coef * entropy_mean + config.vf_coef * value_loss

    with torch.no_grad():
        logged = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_mean.item(),
            # A cheap estimate of how far the policy moved, and the share of ratios that were clipped.
            "approx_kl": ((ratio - 1.0) - log_ratio).mean().item(),
            "clip_fraction": ((ratio - 1.0).abs() > config.clip_coef).float().mean().item(),
        }
    return loss, logged


def apply_update(agent: Agent, optimizer: torch.optim.Optimizer, loss: torch.Tensor, max_grad_norm: float) -> float:
    """One gradient step: compute gradients of `loss`, rescale them if their combined size
    exceeds `max_grad_norm`, and step the optimizer. Returns the gradient size before rescaling.

    The rescaling stops one unusually large gradient (a rare, extreme batch) from throwing the
    networks far off course.
    """
    optimizer.zero_grad()
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
    optimizer.step()
    return float(grad_norm)


def explained_variance(predicted: np.ndarray, target: np.ndarray) -> float:
    """1 means the critic predicts the returns perfectly, 0 no better than a constant, below 0 worse."""
    target_var = float(np.var(target))
    if target_var == 0.0:
        return float("nan")
    return 1.0 - float(np.var(target - predicted)) / target_var


# --------------------------------------------------------------------------
# Using a trained policy, and checkpoints
# --------------------------------------------------------------------------


class Policy:
    """A trained agent plus the observation statistics it was trained with.

    Raw observations go in, actions come out. This is what evaluation and replay use.
    """

    def __init__(self, agent: Agent, obs_rms: RunningMeanStd | None, clip_obs: float, device: torch.device) -> None:
        self.agent = agent
        self.obs_rms = obs_rms
        self.clip_obs = clip_obs
        self.device = device

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Actions for raw observations `obs` [N, obs_dim], as numpy."""
        if self.obs_rms is not None:
            obs = self.obs_rms.normalize(obs, self.clip_obs)
        with torch.no_grad():
            action, _, _ = self.agent.act(torch.as_tensor(obs, dtype=torch.float32, device=self.device), deterministic)
        return action.cpu().numpy()


def _git_state() -> dict:
    """The commit this code is at, and whether files have uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        return {"commit": commit, "uncommitted_changes": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "uncommitted_changes": None}


def save_checkpoint(
    path: Path,
    agent: Agent,
    optimizer: torch.optim.Optimizer,
    obs_rms: RunningMeanStd | None,
    reward_scaler: RewardScaler | None,
    config: PPOConfig,
    update: int,
    global_step: int,
) -> None:
    """Write everything needed to fly the policy or resume training.

    Handoff section 9: the observation statistics are saved with every checkpoint, and each
    checkpoint carries the config, the git commit and the seed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "agent": agent.state_dict(),
            "agent_spec": agent.spec,
            "optimizer": optimizer.state_dict(),
            "obs_rms": None if obs_rms is None else obs_rms.state(),
            "reward_scaler": None if reward_scaler is None else reward_scaler.state(),
            "config_json": json.dumps(asdict(config)),
            "update": update,
            "global_step": global_step,
            "seed": config.seed,
            "git": _git_state(),
        },
        path,
    )


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[Policy, dict]:
    """Load a checkpoint written by `save_checkpoint`. Returns (policy, info dict).

    Loaded with `weights_only=True`, so a checkpoint file cannot run code when opened.
    """
    torch_device = torch.device(device)
    data = torch.load(path, map_location=torch_device, weights_only=True)
    spec = data["agent_spec"]
    agent = Agent(
        spec["obs_dim"], spec["act_dim"], spec["n_actions"], tuple(spec["hidden_sizes"]), spec["init_log_std"]
    ).to(torch_device)
    agent.load_state_dict(data["agent"])
    config = json.loads(data["config_json"])
    obs_rms = None
    if data["obs_rms"] is not None:
        obs_rms = RunningMeanStd((spec["obs_dim"],))
        obs_rms.load_state(data["obs_rms"])
    info = {
        "config": config,
        "update": data["update"],
        "global_step": data["global_step"],
        "seed": data["seed"],
        "git": data["git"],
    }
    return Policy(agent, obs_rms, config["clip_obs"], torch_device), info


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


@dataclass
class TrainResult:
    policy: Policy
    history: list[dict] = field(default_factory=list)  # one dict of metrics per update
    run_dir: Path | None = None


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train(env: VecEnv, config: PPOConfig, run_dir: Path | None = None, verbose: bool = False) -> TrainResult:
    """Train a PPO agent on `env` for `config.total_timesteps` environment steps.

    If `run_dir` is given, the run writes there: `meta.json` (config, git commit, seed),
    `metrics.jsonl` (one line per update), TensorBoard logs in `tb/`, and checkpoints.
    """
    config.validate(env.num_envs)
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    shuffle_generator = torch.Generator().manual_seed(config.seed)

    num_envs, steps = env.num_envs, config.rollout_len
    batch_size = num_envs * steps
    minibatch_size = batch_size // config.num_minibatches
    num_updates = config.total_timesteps // batch_size
    discrete = env.n_actions is not None

    agent = Agent(env.obs_dim, env.act_dim, env.n_actions, config.hidden_sizes, config.init_log_std).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=config.adam_eps)
    obs_rms = RunningMeanStd((env.obs_dim,)) if config.normalize_obs else None
    reward_scaler = RewardScaler(num_envs, config.gamma, config.clip_reward) if config.normalize_reward else None

    def normalized(obs: np.ndarray) -> np.ndarray:
        return obs if obs_rms is None else obs_rms.normalize(obs, config.clip_obs)

    writer = None
    metrics_file = None
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "config": asdict(config),
            "num_envs": num_envs,
            "seed": config.seed,
            "git": _git_state(),
            "device": str(device),
            "torch": torch.__version__,
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        metrics_file = open(run_dir / "metrics.jsonl", "w", encoding="utf-8")
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(run_dir / "tb"))

    # Rollout storage, on the training device.
    obs_buf = torch.zeros((steps, num_envs, env.obs_dim), device=device)
    if discrete:
        action_buf = torch.zeros((steps, num_envs), dtype=torch.long, device=device)
    else:
        action_buf = torch.zeros((steps, num_envs, env.act_dim), device=device)
    logprob_buf = torch.zeros((steps, num_envs), device=device)
    reward_buf = torch.zeros((steps, num_envs), device=device)
    done_buf = torch.zeros((steps, num_envs), device=device)
    value_buf = torch.zeros((steps, num_envs), device=device)

    episode_return = np.zeros(num_envs)
    episode_length = np.zeros(num_envs, dtype=np.int64)
    recent_returns: deque[float] = deque(maxlen=100)
    recent_lengths: deque[float] = deque(maxlen=100)

    obs = env.reset(seed=config.seed)
    if obs_rms is not None:
        obs_rms.update(obs)

    history: list[dict] = []
    global_step = 0
    start_time = time.time()

    for update in range(1, num_updates + 1):
        if config.anneal_lr:
            for group in optimizer.param_groups:
                group["lr"] = config.learning_rate * (1.0 - (update - 1.0) / num_updates)

        # ---- 1. collect experience ----
        for t in range(steps):
            obs_tensor = torch.as_tensor(normalized(obs), device=device)
            with torch.no_grad():
                action, log_prob, value = agent.act(obs_tensor)
            obs_buf[t] = obs_tensor
            action_buf[t] = action
            logprob_buf[t] = log_prob
            value_buf[t] = value

            obs, reward, terminated, truncated, final_obs = env.step(action.cpu().numpy())
            done = terminated | truncated
            global_step += num_envs
            if obs_rms is not None:
                obs_rms.update(obs)

            # Episode statistics use the raw reward.
            episode_return += reward
            episode_length += 1
            for i in np.flatnonzero(done):
                recent_returns.append(float(episode_return[i]))
                recent_lengths.append(float(episode_length[i]))
                episode_return[i] = 0.0
                episode_length[i] = 0

            # The reward the agent learns from: optionally scaled, plus the bootstrap term
            # for episodes that were cut off by a time limit.
            learn_reward = reward_scaler(reward, done) if reward_scaler is not None else reward.copy()
            cut_off = truncated & ~terminated
            if cut_off.any():
                indices = np.flatnonzero(cut_off)
                final_tensor = torch.as_tensor(normalized(final_obs[indices]), device=device)
                with torch.no_grad():
                    final_values = agent.value(final_tensor).cpu().numpy()
                learn_reward[indices] = bootstrap_truncated_rewards(
                    learn_reward[indices], terminated[indices], truncated[indices], final_values, config.gamma
                )
            reward_buf[t] = torch.as_tensor(learn_reward, dtype=torch.float32, device=device)
            done_buf[t] = torch.as_tensor(done, dtype=torch.float32, device=device)

        # ---- 2. score it ----
        with torch.no_grad():
            next_value = agent.value(torch.as_tensor(normalized(obs), device=device))
        advantages, returns = compute_gae(reward_buf, value_buf, done_buf, next_value, config.gamma, config.gae_lambda)

        flat_obs = obs_buf.reshape(batch_size, env.obs_dim)
        flat_actions = action_buf.reshape(batch_size) if discrete else action_buf.reshape(batch_size, env.act_dim)
        flat_logprobs = logprob_buf.reshape(batch_size)
        flat_advantages = advantages.reshape(batch_size)
        flat_returns = returns.reshape(batch_size)
        flat_values = value_buf.reshape(batch_size)

        # ---- 3. improve the policy and the critic ----
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_fraction": []}
        for _ in range(config.update_epochs):
            for chosen in minibatch_indices(batch_size, config.num_minibatches, shuffle_generator):
                idx = chosen.to(device)
                new_log_prob, entropy, new_value = agent.evaluate_actions(flat_obs[idx], flat_actions[idx])
                loss, logged = ppo_loss(
                    new_log_prob, flat_logprobs[idx], flat_advantages[idx], new_value, flat_returns[idx], entropy, config
                )

                apply_update(agent, optimizer, loss, config.max_grad_norm)

                for name, value in logged.items():
                    stats[name].append(value)

        # ---- 4. record ----
        metrics = {
            "update": update,
            "global_step": global_step,
            "ep_return_mean": float(np.mean(recent_returns)) if recent_returns else float("nan"),
            "ep_length_mean": float(np.mean(recent_lengths)) if recent_lengths else float("nan"),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "explained_variance": explained_variance(flat_values.cpu().numpy(), flat_returns.cpu().numpy()),
            "steps_per_second": global_step / max(time.time() - start_time, 1e-9),
        }
        for name, values in stats.items():
            metrics[name] = float(np.mean(values))
        if agent.log_std is not None:
            metrics["action_std"] = float(agent.log_std.exp().mean().item())
        history.append(metrics)

        if metrics_file is not None:
            metrics_file.write(json.dumps(metrics) + "\n")
            metrics_file.flush()
        if writer is not None:
            for name, value in metrics.items():
                if name not in ("update", "global_step") and not math.isnan(value):
                    writer.add_scalar(name, value, global_step)
        if verbose and (update % 10 == 0 or update == num_updates):
            print(
                f"update {update}/{num_updates}  step {global_step}  return {metrics['ep_return_mean']:.1f}  "
                f"{metrics['steps_per_second']:.0f} steps/s"
            )
        if run_dir is not None and (update % config.save_every_updates == 0 or update == num_updates):
            save_checkpoint(
                run_dir / f"checkpoint_{update:06d}.pt",
                agent, optimizer, obs_rms, reward_scaler, config, update, global_step,
            )

    if writer is not None:
        writer.close()
    if metrics_file is not None:
        metrics_file.close()
    return TrainResult(Policy(agent, obs_rms, config.clip_obs, device), history, run_dir)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def evaluate(policy: Policy, env: VecEnv, episodes: int, deterministic: bool = True, seed: int | None = None) -> np.ndarray:
    """Return of each of (at least) `episodes` episodes, with the policy acting deterministically.

    Only the first `ceil(episodes / num_envs)` episodes of each environment count. Counting
    whatever finishes first would favour short episodes.
    """
    per_env = math.ceil(episodes / env.num_envs)
    obs = env.reset(seed=seed)
    running = np.zeros(env.num_envs)
    finished_count = np.zeros(env.num_envs, dtype=np.int64)
    returns: list[float] = []
    while finished_count.min() < per_env:
        obs, reward, terminated, truncated, _ = env.step(policy.act(obs, deterministic))
        running += reward
        for i in np.flatnonzero(terminated | truncated):
            if finished_count[i] < per_env:
                returns.append(float(running[i]))
                finished_count[i] += 1
            running[i] = 0.0
    return np.asarray(returns)
