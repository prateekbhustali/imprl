"""
Training tests for the climb game environment.

This module covers both off-policy and on-policy algorithms on
`matrix_game`. The goal is to verify that the training
and evaluation pipeline runs end-to-end.
These tests are not meant to prove final performance.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

import imprl.agents
import imprl.envs
from tests.training_helpers import (
    OFF_POLICY_ALGORITHMS,
    ON_POLICY_ALGORITHMS,
    _run_off_policy_training_loop,
    _run_on_policy_training_loop,
)

ENV_NAME = "matrix_game"
ENV_SETTING = "climb_game"
ENV_KWARGS = {"time_limit": 25}
INFERENCE_ENV_KWARGS = {"time_limit": 25}
env = imprl.envs.make(ENV_NAME, ENV_SETTING)

NUM_INFERENCE_EPISODES = 50
LONGER_TRAINING_ALGORITHMS = {"JAC", "DCMAC", "DDMAC"}


@pytest.mark.parametrize("algorithm", OFF_POLICY_ALGORITHMS)
def test_off_policy_training_loop(algorithm):
    config_dir = Path(__file__).resolve().parents[1] / "imprl" / "agents" / "configs"
    cfg = OmegaConf.load(config_dir / f"{algorithm}.yaml")
    cfg.ENV_CONFIG.env_name = ENV_NAME
    cfg.ENV_CONFIG.env_setting = ENV_SETTING
    cfg.ENV_CONFIG.kwargs = ENV_KWARGS
    cfg.ENV_CONFIG.inference_env_kwargs = INFERENCE_ENV_KWARGS
    cfg.NUM_INFERENCE_EPISODES = NUM_INFERENCE_EPISODES
    cfg.WANDB.mode = "disabled"
    cfg.DISCOUNT_FACTOR = 0
    cfg.EXPLORATION_STRATEGY.num_episodes = 100
    cfg.NUM_TRAIN_EPISODES = (
        2_000 if algorithm in LONGER_TRAINING_ALGORITHMS else 500
    )
    mean_return = _run_off_policy_training_loop(algorithm=algorithm, cfg=cfg)

    # check if single-agent
    if imprl.agents.REGISTRY[algorithm]["formulation"] == "POMDP":
        # allowing a margin of floating point error
        assert mean_return >= env.core.ep_length * (11 - 2)
    else:
        # For multi-agent algorithms, we expect a lower reward
        # allowing a margin of floating point error
        assert mean_return >= env.core.ep_length * (5 - 1)


@pytest.mark.parametrize("algorithm", ON_POLICY_ALGORITHMS)
def test_on_policy_training_loop(algorithm):
    config_dir = Path(__file__).resolve().parents[1] / "imprl" / "agents" / "configs"
    cfg = OmegaConf.load(config_dir / f"{algorithm}.yaml")
    cfg.ENV_CONFIG.env_name = ENV_NAME
    cfg.ENV_CONFIG.env_setting = ENV_SETTING
    cfg.ENV_CONFIG.kwargs = ENV_KWARGS
    cfg.ENV_CONFIG.inference_env_kwargs = INFERENCE_ENV_KWARGS
    cfg.NUM_INFERENCE_EPISODES = NUM_INFERENCE_EPISODES
    cfg.WANDB.mode = "disabled"
    cfg.total_timesteps = 100_000
    mean_return = _run_on_policy_training_loop(algorithm=algorithm, cfg=cfg)

    # check if single-agent
    if imprl.agents.REGISTRY[algorithm]["formulation"] == "POMDP":
        # allowing a margin of floating point error
        assert mean_return >= env.core.ep_length * (11 - 2)
    else:
        # For multi-agent algorithms, we expect a lower reward
        # allowing a margin of floating point error
        assert mean_return >= env.core.ep_length * (5 - 1)
