import random

import torch
import numpy as np
from omegaconf import OmegaConf

import imprl.agents
import imprl.envs
from imprl.runners.serial import training_rollout
from imprl.runners.parallel import parallel_agent_rollout

from imprl.agents.PPO import main as ppo_main
from imprl.agents.MAPPO_PS import main as mappo_main
from imprl.agents.IPPO_PS import main as ippo_main


OFF_POLICY_ALGORITHMS = [
    name
    for name, metadata in imprl.agents.REGISTRY.items()
    if metadata["policy_regime"] == "off-policy"
]

ON_POLICY_ALGORITHMS = [
    name
    for name, metadata in imprl.agents.REGISTRY.items()
    if metadata["policy_regime"] == "on-policy"
]


def _run_off_policy_training_loop(*, algorithm: str, cfg) -> float:
    is_single_agent = imprl.agents.REGISTRY[algorithm]["formulation"] == "POMDP"
    env_name = cfg.ENV_CONFIG.env_name
    env_setting = cfg.ENV_CONFIG.env_setting
    env_kwargs = OmegaConf.to_container(cfg.ENV_CONFIG.kwargs, resolve=True)
    inference_env_kwargs = OmegaConf.to_container(
        cfg.ENV_CONFIG.inference_env_kwargs, resolve=True
    )

    env = imprl.envs.make(
        env_name,
        env_setting,
        single_agent=is_single_agent,
        **env_kwargs,
    )
    inference_env = imprl.envs.make(
        env_name,
        env_setting,
        single_agent=is_single_agent,
        **inference_env_kwargs,
    )
    alg_config = OmegaConf.to_container(cfg, resolve=True)
    agent = imprl.agents.make(algorithm, env, alg_config, device="cpu")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True

    for ep in range(cfg.NUM_TRAIN_EPISODES):
        training_rollout(inference_env, agent)

    eval_costs = parallel_agent_rollout(
        inference_env, agent, cfg.NUM_INFERENCE_EPISODES
    )
    mean_eval_cost = np.mean(eval_costs)

    return mean_eval_cost


def _run_on_policy_training_loop(*, algorithm: str, cfg) -> float:
    main = {"PPO": ppo_main, "MAPPO_PS": mappo_main, "IPPO_PS": ippo_main}[algorithm]
    mean_return = main.__wrapped__(cfg)

    return mean_return
