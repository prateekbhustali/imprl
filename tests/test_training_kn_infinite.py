"""
Training tests for the infinite-horizon k-out-of-n environment.
Runtime: ~23 minutes

This module covers both off-policy and on-policy algorithms on
`k_out_of_n_infinite`. The goal is to verify that the training
and evaluation pipeline runs end-to-end.
These tests are not meant to prove final performance.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf
import imprl
from tests.training_helpers import (
    OFF_POLICY_ALGORITHMS,
    ON_POLICY_ALGORITHMS,
    _run_off_policy_training_loop,
    _run_on_policy_training_loop,
)

ENV_NAME = "k_out_of_n_infinite"
ENV_SETTING = "hard-4-of-4_infinite"
ENV_KWARGS = {"time_limit": 50}
INFERENCE_ENV_KWARGS = {"time_limit": 20}
env = imprl.envs.make(ENV_NAME, ENV_SETTING)
baseline_cost = env.core.baselines["FailureReplace"]["mean"]

NUM_INFERENCE_EPISODES = 1_000


@pytest.mark.parametrize("algorithm", OFF_POLICY_ALGORITHMS)
def test_off_policy_training_loop(algorithm):
    config_dir = Path(__file__).resolve().parents[1] / "imprl" / "agents" / "configs"
    cfg = OmegaConf.load(config_dir / f"{algorithm}.yaml")
    cfg.ENV_CONFIG.env_name = ENV_NAME
    cfg.ENV_CONFIG.env_setting = ENV_SETTING
    cfg.ENV_CONFIG.kwargs = ENV_KWARGS
    cfg.ENV_CONFIG.inference_env_kwargs = INFERENCE_ENV_KWARGS
    cfg.EXPLORATION_STRATEGY.num_episodes = 1_000
    cfg.NUM_TRAIN_EPISODES = 2_500
    cfg.NUM_INFERENCE_EPISODES = NUM_INFERENCE_EPISODES
    cfg.WANDB.mode = "disabled"
    mean_eval_cost = _run_off_policy_training_loop(algorithm=algorithm, cfg=cfg)
    assert mean_eval_cost < baseline_cost


@pytest.mark.parametrize("algorithm", ON_POLICY_ALGORITHMS)
def test_on_policy_training_loop(algorithm):
    config_dir = Path(__file__).resolve().parents[1] / "imprl" / "agents" / "configs"
    cfg = OmegaConf.load(config_dir / f"{algorithm}.yaml")
    cfg.ENV_CONFIG.env_name = ENV_NAME
    cfg.ENV_CONFIG.env_setting = ENV_SETTING
    cfg.total_timesteps = 2_000_000 if algorithm == "PPO" else 200_000
    cfg.NUM_INFERENCE_EPISODES = NUM_INFERENCE_EPISODES
    cfg.WANDB.mode = "disabled"
    mean_eval_cost = _run_on_policy_training_loop(algorithm=algorithm, cfg=cfg)
    assert mean_eval_cost < baseline_cost
