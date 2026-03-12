"""
Training tests for the finite-horizon k-out-of-n environment.

This module covers both off-policy and on-policy algorithms on
`k_out_of_n_finite`. The goal is to verify that the training
and evaluation pipeline runs end-to-end.
These tests are not meant to prove final performance.
"""

from pathlib import Path

import pytest
import imprl.envs
from omegaconf import OmegaConf
from tests.training_helpers import (
    OFF_POLICY_ALGORITHMS,
    ON_POLICY_ALGORITHMS,
    _run_off_policy_training_loop,
    _run_on_policy_training_loop,
)

ENV_NAME = "k_out_of_n_finite"
ENV_SETTING = "hard-5-of-5"
ENV_KWARGS = {}
INFERENCE_ENV_KWARGS = {}
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
    cfg.NUM_INFERENCE_EPISODES = NUM_INFERENCE_EPISODES
    cfg.EXPLORATION_STRATEGY.num_episodes = 1_000
    cfg.NUM_TRAIN_EPISODES = 3_000
    cfg.WANDB.mode = "disabled"
    mean_eval_cost = _run_off_policy_training_loop(algorithm=algorithm, cfg=cfg)
    assert mean_eval_cost < baseline_cost


@pytest.mark.parametrize("algorithm", ON_POLICY_ALGORITHMS)
def test_on_policy_training_loop(algorithm):
    config_dir = Path(__file__).resolve().parents[1] / "imprl" / "agents" / "configs"
    cfg = OmegaConf.load(config_dir / f"{algorithm}.yaml")
    cfg.ENV_CONFIG.env_name = ENV_NAME
    cfg.ENV_CONFIG.env_setting = ENV_SETTING
    cfg.ENV_CONFIG.kwargs = ENV_KWARGS
    cfg.ENV_CONFIG.inference_env_kwargs = INFERENCE_ENV_KWARGS
    cfg.NUM_INFERENCE_EPISODES = NUM_INFERENCE_EPISODES
    cfg.total_timesteps = 2_000_000 if algorithm == "PPO" else 200_000
    cfg.WANDB.mode = "disabled"
    mean_eval_cost = _run_on_policy_training_loop(algorithm=algorithm, cfg=cfg)
    assert mean_eval_cost < baseline_cost
