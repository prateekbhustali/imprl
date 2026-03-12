"""
Entry point and agent definition for training PPO. Code largely based on CleanRL implementation of PPO.

Usage:
    # default: imprl/agents/configs/PPO.yaml
    python imprl/agents/PPO.py

    # override environment setting and seed
    python imprl/agents/PPO.py ENV_CONFIG.env_setting=hard-2-of-4_infinite SEED=7

    # run without uploading to wandb servers
    python imprl/agents/PPO.py WANDB.mode=disabled
"""

import os
import random
import logging
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

import imprl.envs
from imprl.runners.parallel import parallel_generic_rollout
from imprl.agents.primitives.running_mean_and_std import RunningMeanStd


# ----- Runtime defaults -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_WORKER_COUNT = os.cpu_count() or 1

# ----- Logging -----
logger = logging.getLogger(__name__)


class ProgressLogger:
    def __init__(self, total: int | None = None):
        self.start_time = time.perf_counter()
        self.total = total

    def log(self, message: str, *args, completed: int | None = None) -> None:
        elapsed = max(0, int(time.perf_counter() - self.start_time))
        if completed is None or self.total is None:
            logger.info(message, *args)
            return

        completed_steps = max(int(completed), 1)
        if completed_steps < 2:
            logger.info(message, *args)
            return

        remaining = max(int(self.total) - completed_steps, 0)
        eta = int(elapsed * remaining / completed_steps)
        logger.info(
            "[ETA %s] " + message,
            time.strftime("%H:%M:%S", time.gmtime(max(0, eta))),
            *args,
        )


def config_sanity_check(cfg: DictConfig):
    assert (
        cfg.EVAL_INTERVAL % cfg.CHECKPOINT_INTERVAL == 0
    ), "EVAL_INTERVAL must be a multiple of CHECKPOINT_INTERVAL"


def set_global_seed(seed: int, deterministic: bool = True):
    """Seed Python, NumPy, and PyTorch for reproducible runs."""
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False


def is_time(iteration: int, interval: int | None, cfg: DictConfig) -> bool:
    if interval is None:
        return True
    return ((iteration - 1) % int(interval) == 0) or iteration == cfg.num_iterations


class WrapperForVecEnv:
    """Adapt the single-agent env to a fixed numeric Box space for rollouts.

    The PPO runner stores flat numeric arrays, so this wrapper converts the
    env output to the exact array layout returned by `reset()` and `step()`.
    """

    def __init__(self, env):
        self.env = env
        self.metadata = None
        # Define the Gym space from the actual array returned by reset/step.
        obs, _ = env.reset()
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=obs.shape, dtype=np.float32
        )
        self.action_space = env.action_space

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)

    def close(self):
        pass


def evaluate_agent(env, agent):
    terminated, truncated = False, False
    obs, _ = env.reset()
    total_reward = 0

    while not truncated and not terminated:
        # select action
        with torch.no_grad():
            action = agent.select_action(torch.Tensor(obs))

        # step in the environment
        next_obs, reward, terminated, truncated, _ = env.step(action)

        # process rewards
        total_reward += -reward if env.core.reward_to_cost else reward

        # overwrite obs
        obs = next_obs

    return total_reward


class ProximalPolicyOptimization:
    name = "PPO"
    full_name = "Proximal Policy Optimization"

    # Algorithm taxonomy.
    paradigm = "CTCE"
    formulation = "POMDP"
    algorithm_type = "actor-critic"
    policy_regime = "on-policy"
    policy_type = "stochastic"

    uses_replay_memory = False
    parameter_sharing = True

    def __init__(self, env, config=None, device=None, num_envs=4):
        self.base_env = env
        self.device = device or torch.device("cpu")
        envs = self.create_vectorized_envs(env, num_envs)

        actor_input = np.array(envs.single_observation_space.shape).prod()
        actor_output = envs.single_action_space.n
        critic_input = np.array(envs.single_observation_space.shape).prod()
        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(critic_input, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 1), std=1.0),
        ).to(self.device)
        self.actor = nn.Sequential(
            self.layer_init(nn.Linear(actor_input, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, actor_output), std=0.01),
        ).to(self.device)

    @staticmethod
    def create_vectorized_envs(env, num_envs):
        wrapped_env = WrapperForVecEnv(env)
        envs = gym.vector.SyncVectorEnv(
            [lambda: deepcopy(wrapped_env) for _ in range(num_envs)]
        )
        return envs

    @staticmethod
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def parameters(self):
        return list(self.actor.parameters()) + list(self.critic.parameters())

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def select_action(self, x):
        """Select one action for inference.

        This method expects the processed observation produced by the
        single-agent vectorized wrapper or evaluation helpers.
        """
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        return probs.sample().cpu().numpy()

    def save_weights(self, path, id):
        torch.save(self.actor.state_dict(), f"{path}/actor_{id}.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic_{id}.pth")

    def load_weights(self, path, id):
        # load actor weights
        full_path = f"{path}/actor_{id}.pth"
        self.actor.load_state_dict(
            torch.load(full_path, map_location=torch.device("cpu"))
        )

        # load critic weights
        full_path = f"{path}/critic_{id}.pth"
        self.critic.load_state_dict(
            torch.load(full_path, map_location=torch.device("cpu"))
        )


@hydra.main(config_path="configs", config_name="PPO", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    OmegaConf.set_struct(cfg, False)
    cfg.batch_size = int(cfg.num_envs * cfg.num_steps)
    cfg.minibatch_size = int(cfg.batch_size // cfg.num_minibatches)
    cfg.num_iterations = int(cfg.total_timesteps // cfg.batch_size)
    if cfg.CHECKPOINT_INTERVAL is None:
        cfg.CHECKPOINT_INTERVAL = max(1, int(cfg.num_iterations // 19))
    if cfg.EVAL_INTERVAL is None:
        cfg.EVAL_INTERVAL = cfg.CHECKPOINT_INTERVAL
    progress = ProgressLogger(total=cfg.num_iterations)

    config_sanity_check(cfg)
    progress.log("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    set_global_seed(cfg.SEED, cfg.DETERMINISTIC)

    env = imprl.envs.make(
        cfg.ENV_CONFIG.env_name,
        cfg.ENV_CONFIG.env_setting,
        single_agent=bool(cfg.SINGLE_AGENT),
        **(OmegaConf.to_container(cfg.ENV_CONFIG.kwargs, resolve=True) or {}),
    )
    inference_env = imprl.envs.make(
        cfg.ENV_CONFIG.env_name,
        cfg.ENV_CONFIG.env_setting,
        single_agent=bool(cfg.SINGLE_AGENT),
        **(
            OmegaConf.to_container(cfg.ENV_CONFIG.inference_env_kwargs, resolve=True)
            or {}
        ),
    )
    baseline = env.core.baselines
    baseline_metrics = {f"baselines/{k}": v for k, v in baseline.items()}
    reward_centering = RunningMeanStd()
    alg_config = OmegaConf.to_container(cfg, resolve=True) or {}
    learning_agent = ProximalPolicyOptimization(env, alg_config, device=device)
    envs = learning_agent.create_vectorized_envs(env, cfg.num_envs)
    optimizer = torch.optim.Adam(
        learning_agent.parameters(), lr=cfg.learning_rate, eps=1e-5
    )

    wandb_cfg = OmegaConf.to_container(cfg.WANDB, resolve=True) or {}
    run = wandb.init(**wandb_cfg)
    OmegaConf.set_struct(cfg.WANDB, False)
    cfg.WANDB.run_id = run.id
    progress.log("wandb run with ID: %s", run.id)

    checkpoint_dir = Path(cfg.CHECKPOINT_DIR) / run.id
    cfg.CHECKPOINT_DIR = str(checkpoint_dir)
    progress.log("Checkpoint directory set to: %s", checkpoint_dir)

    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
    wandb.define_metric("inference_mean", summary="min")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_weights_dir = checkpoint_dir / "model_weights"
    model_weights_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    worker_count = min(DEFAULT_WORKER_COUNT, cfg.NUM_INFERENCE_EPISODES)
    progress.log("Running inference with %d worker processes", worker_count)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (cfg.num_steps, cfg.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (cfg.num_steps, cfg.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    rewards = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    dones = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    values = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)

    global_step = 0
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(cfg.num_envs).to(device)
    mean_return = np.inf

    for iteration in range(1, cfg.num_iterations + 1):
        if cfg.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        for step in range(0, cfg.num_steps):
            global_step += cfg.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = learning_agent.get_action_and_value(
                    next_obs
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, _infos = envs.step(
                action.cpu().numpy()
            )
            next_done = terminations
            tensor_rewards = torch.tensor(reward).to(device).view(-1)
            reward_centering.update(tensor_rewards.view(-1, 1))
            normalized_rewards = (tensor_rewards - reward_centering.mean) / torch.sqrt(
                reward_centering.var + 1e-8
            )
            rewards[step] = normalized_rewards
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                next_done
            ).to(device)

        with torch.no_grad():
            next_value = learning_agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(cfg.batch_size)
        clipfracs = []
        old_approx_kl = torch.tensor(0.0, device=device)
        approx_kl = torch.tensor(0.0, device=device)
        for _epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = learning_agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(learning_agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if is_time(iteration, cfg.EVAL_INTERVAL, cfg):
            mean_return = parallel_generic_rollout(
                inference_env,
                learning_agent,
                evaluate_agent,
                cfg.NUM_INFERENCE_EPISODES,
                num_workers=worker_count,
            ).mean()
            progress.log(
                "Iteration %d - Global Step %d - Mean Return %.3f",
                iteration,
                global_step,
                mean_return,
                completed=iteration,
            )
        if is_time(iteration, cfg.CHECKPOINT_INTERVAL, cfg):
            learning_agent.save_weights(str(model_weights_dir), global_step)
            progress.log(
                "Saved checkpoint at iteration %d (step %d)",
                iteration,
                global_step,
                completed=iteration,
            )

        training_log = {
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "losses/explained_variance": float(explained_var),
            "inference_mean": float(mean_return),
            "metrics/reward_mean": reward_centering.mean.item(),
            "charts/SPS": int(
                global_step / max(time.perf_counter() - progress.start_time, 1e-8)
            ),
        }
        training_log.update(baseline_metrics)
        if is_time(iteration, cfg.LOGGING_INTERVAL, cfg):
            wandb.log(training_log, step=global_step)
            progress.log("iteration: %6d | mean_return: %6.2f", iteration, mean_return)

    envs.close()
    total_training_time = time.strftime(
        "%H:%M:%S", time.gmtime(max(0, int(time.perf_counter() - progress.start_time)))
    )
    progress.log(
        "Final mean return %.6f | total training time %s",
        mean_return,
        total_training_time,
    )
    wandb.finish()

    return mean_return


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
