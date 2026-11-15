"""Milestone 3: tests for the hand-written PPO (python/quadrace/ppo.py).

The maths pieces (GAE, truncation bootstrapping, running statistics) are checked on their own
against independent calculations. The training loop is checked on tiny made-up environments
whose right answer can be worked out by hand, and then on real CartPole.

One behaviour per test, named after what it checks.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from quadrace import ppo
from quadrace.vec_env import GymVecEnv


# --------------------------------------------------------------------------
# A made-up environment with a known answer
# --------------------------------------------------------------------------


class ConstantEnv:
    """`num_envs` copies of a one-state environment: the observation is always 0 and the reward
    is always 1. Every episode lasts `length` steps and then ends either by `terminated` (a real
    ending) or by `truncated` (a time limit), as chosen by `ending`.

    Because the state never changes, the right value of that state can be worked out by hand,
    which makes this a precise test of how PPO treats the two kinds of ending.
    """

    obs_dim = 1
    act_dim = 1
    n_actions = None

    def __init__(self, num_envs: int, length: int, ending: str, obs_value: float = 0.0) -> None:
        assert ending in ("terminated", "truncated")
        self.num_envs = num_envs
        self.length = length
        self.ending = ending
        self.obs_value = obs_value
        self.time = np.zeros(num_envs, dtype=np.int64)

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.time[:] = 0
        return np.full((self.num_envs, 1), self.obs_value, dtype=np.float32)

    def step(self, actions: np.ndarray):
        assert actions.shape == (self.num_envs, 1)
        self.time += 1
        done = self.time == self.length
        self.time[done] = 0
        none = np.zeros(self.num_envs, dtype=bool)
        obs = np.full((self.num_envs, 1), self.obs_value, dtype=np.float32)
        reward = np.ones(self.num_envs)
        terminated = done if self.ending == "terminated" else none
        truncated = done if self.ending == "truncated" else none
        return obs, reward, terminated, truncated, obs.copy()


def tiny_config(**overrides) -> ppo.PPOConfig:
    """A small, fast config for the made-up environment: gamma 0.9, no normalization."""
    settings = dict(
        total_timesteps=4 * 32 * 300, rollout_len=32, num_minibatches=1, update_epochs=4,
        gamma=0.9, gae_lambda=0.95, learning_rate=3e-3, anneal_lr=False, ent_coef=0.0,
        hidden_sizes=(16, 16), normalize_obs=False, normalize_reward=False, seed=3, device="cpu",
    )
    settings.update(overrides)
    return ppo.PPOConfig(**settings)


def learned_value(result: ppo.TrainResult) -> float:
    with torch.no_grad():
        return float(result.policy.agent.value(torch.zeros(1, 1)).item())


# --------------------------------------------------------------------------
# GAE
# --------------------------------------------------------------------------


def gae_by_definition(rewards, values, dones, next_value, gamma, lam):
    """GAE written as the plain forward sum A_t = sum_k (gamma lam)^k delta_{t+k}, with loops.
    It shares nothing with the backward recursion in ppo.compute_gae, so it is an independent check."""
    steps, envs = rewards.shape
    advantages = np.zeros((steps, envs))
    for n in range(envs):
        for t in range(steps):
            total, weight = 0.0, 1.0
            for k in range(t, steps):
                following = next_value[n] if k == steps - 1 else values[k + 1, n]
                not_done = 1.0 - dones[k, n]
                delta = rewards[k, n] + gamma * following * not_done - values[k, n]
                total += weight * delta
                weight *= gamma * lam * not_done
            advantages[t, n] = total
    return advantages


def test_gae_matches_the_definition_on_random_data() -> None:
    rng = np.random.default_rng(0)
    steps, envs = 12, 5
    rewards = rng.normal(size=(steps, envs))
    values = rng.normal(size=(steps, envs))
    dones = (rng.random((steps, envs)) < 0.2).astype(np.float64)
    next_value = rng.normal(size=envs)

    advantages, returns = ppo.compute_gae(
        torch.as_tensor(rewards), torch.as_tensor(values), torch.as_tensor(dones),
        torch.as_tensor(next_value), gamma=0.97, gae_lambda=0.9,
    )

    expected = gae_by_definition(rewards, values, dones, next_value, 0.97, 0.9)
    np.testing.assert_allclose(advantages.numpy(), expected, atol=1e-12)
    np.testing.assert_allclose(returns.numpy(), expected + values, atol=1e-12)


def test_gae_with_lambda_one_and_gamma_one_is_the_summed_rewards_plus_the_final_value() -> None:
    rewards = torch.ones(3, 1)
    advantages, returns = ppo.compute_gae(
        rewards, torch.zeros(3, 1), torch.zeros(3, 1), torch.tensor([5.0]), gamma=1.0, gae_lambda=1.0
    )
    # Step 0 sees rewards 1 + 1 + 1 and then the final value 5, and so on.
    np.testing.assert_allclose(advantages[:, 0].numpy(), [8.0, 7.0, 6.0])
    np.testing.assert_allclose(returns[:, 0].numpy(), [8.0, 7.0, 6.0])


def test_gae_with_lambda_zero_is_the_one_step_td_error() -> None:
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [1.5], [2.5]])
    advantages, _ = ppo.compute_gae(
        rewards, values, torch.zeros(3, 1), torch.tensor([4.0]), gamma=0.9, gae_lambda=0.0
    )
    expected = [1.0 + 0.9 * 1.5 - 0.5, 2.0 + 0.9 * 2.5 - 1.5, 3.0 + 0.9 * 4.0 - 2.5]
    np.testing.assert_allclose(advantages[:, 0].numpy(), expected, rtol=1e-6)


def test_gae_does_not_look_past_the_end_of_an_episode() -> None:
    # Step 1 ends its episode. Its advantage is just its own reward minus its value, with no
    # value of a next state, and step 0 only sees as far as step 1.
    rewards = torch.ones(3, 1)
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    advantages, _ = ppo.compute_gae(
        rewards, torch.zeros(3, 1), dones, torch.tensor([5.0]), gamma=1.0, gae_lambda=1.0
    )
    np.testing.assert_allclose(advantages[:, 0].numpy(), [2.0, 1.0, 6.0])


# --------------------------------------------------------------------------
# Truncation versus termination
# --------------------------------------------------------------------------


def test_truncated_steps_get_gamma_times_the_final_value_added_to_their_reward() -> None:
    rewards = np.array([1.0, 1.0, 1.0, 1.0])
    terminated = np.array([False, True, False, False])
    truncated = np.array([False, False, True, False])
    final_values = np.array([9.0, 9.0, 9.0, 9.0])

    result = ppo.bootstrap_truncated_rewards(rewards, terminated, truncated, final_values, gamma=0.5)

    # Only the truncated step (index 2) gets 0.5 * 9; the terminated and running steps do not.
    np.testing.assert_allclose(result, [1.0, 1.0, 5.5, 1.0])


def test_a_time_limit_cut_off_is_bootstrapped_so_the_critic_learns_the_true_infinite_horizon_value() -> None:
    # Reward 1 forever, cut off by a time limit every 5 steps. The state never changes, so the
    # true value is 1 / (1 - gamma) = 10 for gamma = 0.9. Without bootstrapping from the final
    # observation the critic would learn about 3 (the average of the short truncated returns).
    result = ppo.train(ConstantEnv(4, length=5, ending="truncated"), tiny_config())
    assert learned_value(result) == pytest.approx(10.0, abs=0.5)


def test_a_real_ending_is_not_bootstrapped_so_the_critic_learns_the_short_return() -> None:
    # Same environment, but each episode really ends after 5 steps. Now the future after the
    # last step is worth nothing. A critic that cannot see the time must learn the average
    # remaining return, about 2.7 for gamma = 0.9, lambda = 0.95 (worked out by hand), not 10.
    result = ppo.train(ConstantEnv(4, length=5, ending="terminated"), tiny_config())
    assert learned_value(result) == pytest.approx(2.7, abs=0.5)


# --------------------------------------------------------------------------
# The PPO loss
# --------------------------------------------------------------------------


def loss_for(*, ratio: float, advantage: float, entropy: float = 0.0, new_value: float = 0.0,
             target: float = 0.0, **config_overrides) -> tuple[float, dict]:
    """The PPO loss for a single made-up sample whose new/old probability ratio is `ratio`."""
    settings = dict(normalize_advantage=False, ent_coef=0.0, vf_coef=0.5, clip_coef=0.2)
    settings.update(config_overrides)
    config = tiny_config(**settings)
    loss, logged = ppo.ppo_loss(
        new_log_prob=torch.tensor([np.log(ratio)]), old_log_prob=torch.zeros(1),
        advantage=torch.tensor([advantage]), new_value=torch.tensor([new_value]),
        returns=torch.tensor([target]), entropy=torch.tensor([entropy]), config=config,
    )
    return float(loss), logged


def test_a_good_action_gets_no_extra_credit_once_its_ratio_passes_one_plus_clip() -> None:
    # Positive advantage, ratio 1.5: the clipped term (-1.2) is the larger loss, so it is used.
    loss, logged = loss_for(ratio=1.5, advantage=1.0)
    assert loss == pytest.approx(-1.2, abs=1e-6)
    assert logged["clip_fraction"] == 1.0


def test_a_bad_action_is_not_punished_extra_once_its_ratio_falls_below_one_minus_clip() -> None:
    # Negative advantage, ratio 0.5: the clipped term (+0.8) is the larger loss, so it is used.
    loss, _ = loss_for(ratio=0.5, advantage=-1.0)
    assert loss == pytest.approx(0.8, abs=1e-6)


def test_a_bad_action_whose_probability_rose_a_lot_is_still_punished_in_full() -> None:
    # Negative advantage, ratio 1.5: the unclipped term (+1.5) is the larger loss, so it is used.
    # Clipping must only ever remove credit, never remove a penalty.
    loss, _ = loss_for(ratio=1.5, advantage=-1.0)
    assert loss == pytest.approx(1.5, abs=1e-6)


def test_inside_the_trust_region_the_loss_is_just_minus_advantage_times_ratio() -> None:
    loss, logged = loss_for(ratio=1.1, advantage=2.0)
    assert loss == pytest.approx(-2.2, abs=1e-6)
    assert logged["clip_fraction"] == 0.0


def test_a_higher_entropy_lowers_the_loss_by_ent_coef_times_the_entropy() -> None:
    base, _ = loss_for(ratio=1.0, advantage=0.0, entropy=0.0)
    bonus, _ = loss_for(ratio=1.0, advantage=0.0, entropy=2.0, ent_coef=0.1)
    assert bonus - base == pytest.approx(-0.2, abs=1e-6)


def test_value_loss_is_vf_coef_times_the_squared_error() -> None:
    loss, logged = loss_for(ratio=1.0, advantage=0.0, new_value=3.0, target=1.0)  # error 2, squared 4
    assert logged["value_loss"] == pytest.approx(4.0)
    assert loss == pytest.approx(0.5 * 4.0, abs=1e-6)  # vf_coef 0.5


def test_advantages_are_normalized_inside_the_minibatch_when_asked() -> None:
    advantage = torch.tensor([1.0, 2.0, 3.0])
    args = dict(
        new_log_prob=torch.zeros(3), old_log_prob=torch.zeros(3), advantage=advantage,
        new_value=torch.zeros(3), returns=torch.zeros(3), entropy=torch.zeros(3),
    )
    normalized, _ = ppo.ppo_loss(config=tiny_config(normalize_advantage=True, ent_coef=0.0), **args)
    raw, _ = ppo.ppo_loss(config=tiny_config(normalize_advantage=False, ent_coef=0.0), **args)
    assert float(normalized) == pytest.approx(0.0, abs=1e-6)  # ratio 1: -mean(normalized advantages)
    assert float(raw) == pytest.approx(-2.0, abs=1e-6)


def test_the_loss_pushes_a_good_actions_log_prob_up() -> None:
    new_log_prob = torch.zeros(1, requires_grad=True)
    loss, _ = ppo.ppo_loss(
        new_log_prob, torch.zeros(1), torch.tensor([1.0]), torch.zeros(1), torch.zeros(1), torch.zeros(1),
        tiny_config(normalize_advantage=False),
    )
    loss.backward()
    assert new_log_prob.grad.item() < 0  # minimizing the loss raises new_log_prob


# --------------------------------------------------------------------------
# Running statistics
# --------------------------------------------------------------------------


def test_running_mean_std_matches_numpy_on_the_same_data() -> None:
    rng = np.random.default_rng(1)
    data = rng.normal(loc=3.0, scale=2.0, size=(1000, 4))
    rms = ppo.RunningMeanStd((4,))
    rms.update(data)
    # The tiny prior count (1e-4) makes this match to about 1e-6, not exactly.
    np.testing.assert_allclose(rms.mean, data.mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(rms.var, data.var(axis=0), rtol=1e-5)


def test_running_mean_std_gives_the_same_answer_in_pieces_as_all_at_once() -> None:
    rng = np.random.default_rng(2)
    data = rng.normal(size=(900, 3))
    whole = ppo.RunningMeanStd((3,))
    whole.update(data)
    pieces = ppo.RunningMeanStd((3,))
    for chunk in (data[:100], data[100:450], data[450:]):
        pieces.update(chunk)
    np.testing.assert_allclose(pieces.mean, whole.mean, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(pieces.var, whole.var, rtol=1e-9)


def test_normalize_gives_zero_mean_and_unit_variance_and_clips() -> None:
    rng = np.random.default_rng(3)
    data = rng.normal(loc=10.0, scale=5.0, size=(5000, 2))
    rms = ppo.RunningMeanStd((2,))
    rms.update(data)
    normalized = rms.normalize(data, clip=10.0)
    np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-2)
    np.testing.assert_allclose(normalized.std(axis=0), 1.0, atol=1e-2)
    assert rms.normalize(np.array([[1e9, -1e9]]), clip=10.0).tolist() == [[10.0, -10.0]]


def test_reward_scaler_result_does_not_depend_on_the_size_of_the_rewards() -> None:
    rng = np.random.default_rng(4)
    rewards = rng.normal(size=(200, 3))
    small, large = ppo.RewardScaler(3, 0.99, 10.0), ppo.RewardScaler(3, 0.99, 10.0)
    no_done = np.zeros(3, dtype=bool)
    for r in rewards:
        a = small(r, no_done)
        b = large(1000.0 * r, no_done)
    np.testing.assert_allclose(a, b, rtol=1e-4)


def test_reward_scaler_clips_to_the_given_limit() -> None:
    scaler = ppo.RewardScaler(2, 0.99, clip=3.0)
    for _ in range(50):
        scaler(np.array([0.001, 0.001]), np.zeros(2, dtype=bool))
    assert np.all(np.abs(scaler(np.array([1e6, -1e6]), np.zeros(2, dtype=bool))) <= 3.0)


# --------------------------------------------------------------------------
# Minibatches, config, and the networks
# --------------------------------------------------------------------------


def test_every_sample_is_in_exactly_one_minibatch_per_epoch() -> None:
    chunks = ppo.minibatch_indices(96, 8, torch.Generator().manual_seed(0))
    assert len(chunks) == 8 and all(len(c) == 12 for c in chunks)
    assert sorted(torch.cat(chunks).tolist()) == list(range(96))


def test_config_rejects_a_batch_that_does_not_divide_into_minibatches() -> None:
    with pytest.raises(ValueError, match="divisible"):
        tiny_config(num_minibatches=7).validate(num_envs=4)  # 4 * 32 = 128 is not divisible by 7


def test_config_rejects_a_budget_smaller_than_one_batch() -> None:
    with pytest.raises(ValueError, match="total_timesteps"):
        tiny_config(total_timesteps=10).validate(num_envs=4)


def test_continuous_agent_samples_a_vector_and_deterministic_action_is_the_mean() -> None:
    agent = ppo.Agent(3, 2, None, (8,), -0.5)
    obs = torch.randn(5, 3)
    action, log_prob, value = agent.act(obs)
    assert action.shape == (5, 2) and log_prob.shape == (5,) and value.shape == (5,)

    agent.actor[-1].weight.data.zero_()
    agent.actor[-1].bias.data = torch.tensor([0.3, -0.2])
    mean_action, _, _ = agent.act(obs, deterministic=True)
    torch.testing.assert_close(mean_action, torch.tensor([[0.3, -0.2]]).expand(5, 2))


def test_discrete_agent_deterministic_action_is_the_most_likely_choice() -> None:
    agent = ppo.Agent(3, 1, 3, (8,), -0.5)
    agent.actor[-1].weight.data.zero_()
    agent.actor[-1].bias.data = torch.tensor([0.0, 5.0, 0.0])
    action, _, _ = agent.act(torch.randn(4, 3), deterministic=True)
    assert action.tolist() == [1, 1, 1, 1] and action.dtype == torch.long


@pytest.mark.parametrize("n_actions", [None, 4])
def test_log_prob_of_a_sampled_action_is_the_same_when_evaluated_again(n_actions: int | None) -> None:
    agent = ppo.Agent(5, 3, n_actions, (16, 16), -0.5)
    obs = torch.randn(7, 5)
    action, log_prob, value = agent.act(obs)
    new_log_prob, entropy, new_value = agent.evaluate_actions(obs, action)
    torch.testing.assert_close(new_log_prob, log_prob)
    torch.testing.assert_close(new_value, value)
    assert entropy.shape == (7,)


def test_initial_actor_is_nearly_neutral_and_the_log_std_starts_where_asked() -> None:
    agent = ppo.Agent(6, 4, None, (32, 32), -0.5)
    action_mean = agent.actor(torch.randn(100, 6))
    assert action_mean.abs().max() < 0.1  # last-layer gain 0.01
    torch.testing.assert_close(agent.log_std.data, torch.full((4,), -0.5))


# --------------------------------------------------------------------------
# Whole training runs
# --------------------------------------------------------------------------


def test_training_with_the_same_seed_gives_identical_networks_and_a_different_seed_does_not() -> None:
    def run(seed: int) -> list[torch.Tensor]:
        result = ppo.train(ConstantEnv(4, 5, "terminated"), tiny_config(total_timesteps=4 * 32 * 5, seed=seed))
        return [p.detach().clone() for p in result.policy.agent.parameters()]

    a, b, c = run(7), run(7), run(8)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    assert any(not torch.equal(x, y) for x, y in zip(a, c, strict=True))


def test_checkpoints_are_written_on_schedule_and_reload_to_the_same_policy(tmp_path: Path) -> None:
    env = ConstantEnv(4, 5, "terminated")
    config = tiny_config(total_timesteps=4 * 32 * 5, save_every_updates=2, normalize_obs=True, seed=11)
    result = ppo.train(env, config, run_dir=tmp_path)

    names = sorted(p.name for p in tmp_path.glob("checkpoint_*.pt"))
    assert names == ["checkpoint_000002.pt", "checkpoint_000004.pt", "checkpoint_000005.pt"]

    loaded, info = ppo.load_checkpoint(tmp_path / "checkpoint_000005.pt")
    obs = np.random.default_rng(5).normal(size=(6, 1)).astype(np.float32)
    np.testing.assert_array_equal(result.policy.act(obs), loaded.act(obs))
    np.testing.assert_array_equal(result.policy.obs_rms.mean, loaded.obs_rms.mean)  # statistics are saved
    np.testing.assert_array_equal(result.policy.obs_rms.var, loaded.obs_rms.var)
    assert info["update"] == 5 and info["seed"] == 11 and info["global_step"] == 4 * 32 * 5
    assert info["config"]["gamma"] == 0.9 and "commit" in info["git"]


def test_a_run_writes_its_config_commit_and_seed_and_one_metrics_line_per_update(tmp_path: Path) -> None:
    ppo.train(ConstantEnv(4, 5, "terminated"), tiny_config(total_timesteps=4 * 32 * 3, seed=21), run_dir=tmp_path)

    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 21 and meta["num_envs"] == 4 and meta["config"]["rollout_len"] == 32
    assert "commit" in meta["git"]

    lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    last = json.loads(lines[-1])
    for key in ("update", "global_step", "ep_return_mean", "policy_loss", "value_loss", "entropy", "approx_kl",
                "clip_fraction", "explained_variance", "learning_rate", "steps_per_second", "action_std"):
        assert key in last
    assert (tmp_path / "tb").is_dir()


def test_observation_statistics_are_updated_during_training_and_saved() -> None:
    # An environment whose observation is always 5.0: the saved mean must end up at 5.0.
    result = ppo.train(
        ConstantEnv(4, 5, "terminated", obs_value=5.0), tiny_config(total_timesteps=4 * 32 * 3, normalize_obs=True)
    )
    assert result.policy.obs_rms.mean[0] == pytest.approx(5.0, abs=1e-3)
    assert result.policy.obs_rms.count > 4 * 32 * 3


def test_evaluate_returns_the_return_of_each_episode_with_a_deterministic_policy() -> None:
    result = ppo.train(ConstantEnv(4, 5, "terminated"), tiny_config(total_timesteps=4 * 32))
    scores = ppo.evaluate(result.policy, ConstantEnv(4, 5, "terminated"), episodes=8)
    np.testing.assert_array_equal(scores, np.full(8, 5.0))  # 5 steps of reward 1 per episode


def test_evaluate_counts_only_the_first_episodes_of_each_environment() -> None:
    # 3 environments and 7 requested episodes: each environment contributes ceil(7 / 3) = 3.
    result = ppo.train(ConstantEnv(4, 5, "terminated"), tiny_config(total_timesteps=4 * 32))
    assert len(ppo.evaluate(result.policy, ConstantEnv(3, 5, "terminated"), episodes=7)) == 9


def test_learning_rate_decays_linearly_to_zero_when_annealing() -> None:
    config = tiny_config(total_timesteps=4 * 32 * 4, anneal_lr=True, learning_rate=1e-3)
    result = ppo.train(ConstantEnv(4, 5, "terminated"), config)
    rates = [m["learning_rate"] for m in result.history]
    np.testing.assert_allclose(rates, [1e-3, 0.75e-3, 0.5e-3, 0.25e-3])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")
def test_training_runs_on_the_gpu() -> None:
    result = ppo.train(ConstantEnv(4, 5, "terminated"), tiny_config(total_timesteps=4 * 32 * 3, device="cuda"))
    assert str(next(result.policy.agent.parameters()).device).startswith("cuda")
    assert all(np.isfinite(m["value_loss"]) for m in result.history)


def test_ppo_learns_cartpole() -> None:
    # CleanRL's CartPole settings. A random policy scores about 22; a solved one 500. Reaching a
    # mean of 150 over the last 100 episodes inside 120,000 steps is a clear sign of learning
    # that leaves a wide margin for run-to-run differences between machines.
    config = ppo.PPOConfig(
        total_timesteps=120_000, rollout_len=128, num_minibatches=4, update_epochs=4, gamma=0.99,
        gae_lambda=0.95, clip_coef=0.2, learning_rate=2.5e-4, ent_coef=0.01, max_grad_norm=0.5,
        hidden_sizes=(64, 64), normalize_obs=False, seed=1, device="cpu",
    )
    env = GymVecEnv("CartPole-v1", 4)
    result = ppo.train(env, config)
    best = max(m["ep_return_mean"] for m in result.history if not np.isnan(m["ep_return_mean"]))
    assert best >= 150.0
    scores = ppo.evaluate(result.policy, GymVecEnv("CartPole-v1", 5), episodes=10, seed=123)
    assert scores.mean() >= 100.0
