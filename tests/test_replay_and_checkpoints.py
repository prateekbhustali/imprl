import random
import shutil
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

import imprl.agents
import imprl.envs
from imprl.agents.primitives.exploration_schedulers import LinearExplorationScheduler
from imprl.agents.primitives.replay_memory import AbstractReplayMemory


def _assert_replay_memory_invariants(memory):
    """Assert fixed replay-memory invariants for eviction and sampling semantics."""
    # Store 4 experiences into size-3 replay: oldest must be evicted.
    exp0 = ("b0", "a0", "nb0", 0.0, False, False)
    exp1 = ("b1", "a1", "nb1", 1.0, False, False)
    exp2 = ("b2", "a2", "nb2", 2.0, True, False)
    exp3 = ("b3", "a3", "nb3", 3.0, False, True)
    for exp in (exp0, exp1, exp2, exp3):
        memory.store_experience(*exp)

    assert len(memory.memory) == 3
    assert list(memory.memory) == [exp1, exp2, exp3]

    # Sampling should preserve tuple ordering (field-wise lists with equal lengths).
    random.seed(7)
    batch = memory.sample_batch(batch_size=2)
    assert len(batch) == len(exp0)
    assert all(len(field_values) == 2 for field_values in batch)

    sampled_experiences = list(zip(*batch))
    valid_experiences = set(memory.memory)
    for sampled in sampled_experiences:
        assert sampled in valid_experiences


def test_abstract_replay_memory_invariants():
    """Validate replay-memory invariants on the standalone abstract memory class."""
    memory = AbstractReplayMemory(size=3)
    _assert_replay_memory_invariants(memory)


def _assert_linear_scheduler_invariants(
    scheduler, initial_eps: float, final_eps: float, num_episodes: int
):
    """Assert linear scheduler starts high, decays monotonically, and saturates."""
    assert scheduler.eps == pytest.approx(initial_eps)

    values = [scheduler.step() for _ in range(num_episodes + 5)]

    assert all(final_eps <= v <= initial_eps for v in values)
    assert all(curr >= nxt for curr, nxt in zip(values, values[1:]))
    assert values[num_episodes - 1] == pytest.approx(final_eps)
    assert values[-1] == pytest.approx(final_eps)


def test_abstract_linear_exploration_scheduler_invariants():
    """Validate scheduler invariants on a standalone LinearExplorationScheduler."""
    initial_eps, final_eps, num_episodes = 1.0, 0.1, 10
    scheduler = LinearExplorationScheduler(
        final_eps=final_eps,
        num_episodes=num_episodes,
        initial_eps=initial_eps,
    )
    _assert_linear_scheduler_invariants(
        scheduler=scheduler,
        initial_eps=initial_eps,
        final_eps=final_eps,
        num_episodes=num_episodes,
    )


CHECKPOINT_ALGOS = {
    "DDQN": ("q_network",),
    "JAC": ("actor", "critic"),
    "DCMAC": ("actor", "critic"),
    "DDMAC": ("actor", "critic"),
    "IAC": ("actor", "critic"),
    "IACC": ("actor", "critic"),
    "IAC_PS": ("actor", "critic"),
    "IACC_PS": ("actor", "critic"),
    "VDN_PS": ("q_network",),
    "QMIX_PS": ("q_network", "q_mixer"),
}
CONFIG_DIR = Path(__file__).resolve().parents[1] / "imprl" / "agents" / "configs"


def _build_env(algorithm):
    """Build the matrix-game test environment with correct single/multi-agent mode."""
    return imprl.envs.make(
        "matrix_game",
        "climb_game",
        single_agent=imprl.agents.REGISTRY[algorithm]["formulation"] == "POMDP",
    )

@pytest.mark.parametrize("algorithm", CHECKPOINT_ALGOS.keys())
def test_replay_memory_invariants_for_each_algorithm(algorithm):
    """Check replay-memory invariants against each algorithm's replay buffer instance."""
    env = _build_env(algorithm)
    config = OmegaConf.to_container(
        OmegaConf.load(CONFIG_DIR / f"{algorithm}.yaml"), resolve=True
    )
    agent = imprl.agents.make(algorithm, env, config, device="cpu")

    # Use a tiny buffer so eviction behavior is deterministic in this invariant test.
    agent.replay_memory = AbstractReplayMemory(size=3)
    _assert_replay_memory_invariants(agent.replay_memory)


@pytest.mark.parametrize("algorithm", CHECKPOINT_ALGOS.keys())
def test_exploration_scheduler_invariants_for_each_algorithm(algorithm):
    """Check epsilon scheduler invariants for each algorithm from its test config."""
    env = _build_env(algorithm)
    config = OmegaConf.to_container(
        OmegaConf.load(CONFIG_DIR / f"{algorithm}.yaml"), resolve=True
    )
    agent = imprl.agents.make(algorithm, env, config, device="cpu")

    strategy = config["EXPLORATION_STRATEGY"]
    _assert_linear_scheduler_invariants(
        scheduler=agent.exp_scheduler,
        initial_eps=float(strategy["max_value"]),
        final_eps=float(strategy["min_value"]),
        num_episodes=int(strategy["num_episodes"]),
    )


def _assert_module_state_equal(module_a, module_b):
    """Assert exact parameter equality between two torch modules (or module lists)."""
    def _states(module):
        if hasattr(module, "state_dict"):
            return [module.state_dict()]
        if hasattr(module, "networks"):
            return [network.state_dict() for network in module.networks]
        raise AttributeError(f"Unsupported module type for state comparison: {type(module)}")

    states_a = _states(module_a)
    states_b = _states(module_b)
    assert len(states_a) == len(states_b)

    for state_a, state_b in zip(states_a, states_b):
        assert state_a.keys() == state_b.keys()
        for key in state_a:
            assert torch.equal(state_a[key], state_b[key]), f"Mismatch in key '{key}'"


@pytest.mark.parametrize("algorithm,module_names", CHECKPOINT_ALGOS.items())
def test_checkpoint_round_trip_exact_params(algorithm, module_names, tmp_path):
    """Save and reload checkpoints, then assert exact parameter round-trip equality."""
    env = _build_env(algorithm)
    config = OmegaConf.to_container(
        OmegaConf.load(CONFIG_DIR / f"{algorithm}.yaml"), resolve=True
    )

    agent = imprl.agents.make(algorithm, env, config, device="cpu")
    loaded_agent = imprl.agents.make(algorithm, env, config, device="cpu")

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    episode = 7

    agent.save_weights(checkpoint_dir, episode)
    loaded_agent.load_weights(checkpoint_dir, episode)

    for module_name in module_names:
        _assert_module_state_equal(
            getattr(agent, module_name), getattr(loaded_agent, module_name)
        )

    # Explicit cleanup to avoid leaving files around after test runs.
    shutil.rmtree(checkpoint_dir)
    assert not checkpoint_dir.exists()
