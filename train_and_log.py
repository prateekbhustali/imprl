"""
Entry point for training off-policy DRL agents and logging results to Weights & Biases (wandb).
Supported algs: JAC, DDQN, DCMAC, DDMAC, IACC(-PS), IAC(-PS), VDN-PS, QMIX-PS.

Usage:
    # default: DDQN from imprl/agents/configs/DDQN.yaml
    python train_and_log.py

    # choose a different off-policy config
    python train_and_log.py --config-name DCMAC

    # override environment setting and seed
    python train_and_log.py --config-name DCMAC ENV_CONFIG.env_setting=hard-2-of-4_infinite SEED=7

    # run without checkpointing and uploading to wandb
    python train_and_log.py --config-name DDQN WANDB.mode=disabled
"""

import os
import random
import math
import logging
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
import wandb
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf

import imprl.agents
import imprl.envs
from imprl.runners.serial import training_rollout
from imprl.runners.parallel import parallel_agent_rollout


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


# ------ Helper functions ------
def config_sanity_check(cfg: DictConfig):
    assert (
        cfg.EVAL_INTERVAL % cfg.CHECKPOINT_INTERVAL == 0
    ), "EVAL_INTERVAL must be a multiple of CHECKPOINT_INTERVAL"


def set_global_seed(seed: int, deterministic: bool = True):
    """Seed Python, NumPy, and PyTorch for reproducible runs."""
    seed = int(seed)

    # Python + NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False


def is_time(ep: int, interval: int, cfg: DictConfig) -> bool:
    return ep % interval == 0 or ep == cfg.NUM_TRAIN_EPISODES - 1


@hydra.main(config_path="imprl/agents/configs", config_name="DDQN", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    progress = ProgressLogger(total=cfg.NUM_TRAIN_EPISODES)

    # ----- validate config
    config_sanity_check(cfg)

    # ----- basic config echo
    progress.log("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    # ----- reproducibility
    set_global_seed(cfg.SEED, cfg.DETERMINISTIC)

    # ----- create training env
    env = imprl.envs.make(
        cfg.ENV_CONFIG.env_name,
        cfg.ENV_CONFIG.env_setting,
        single_agent=bool(cfg.SINGLE_AGENT),
        **(OmegaConf.to_container(cfg.ENV_CONFIG.kwargs, resolve=True) or {}),
    )
    # ----- create inference/evaluation env
    # Keep this separate so eval settings can differ from training settings.
    inference_env = imprl.envs.make(
        cfg.ENV_CONFIG.env_name,
        cfg.ENV_CONFIG.env_setting,
        single_agent=bool(cfg.SINGLE_AGENT),
        **(
            OmegaConf.to_container(cfg.ENV_CONFIG.inference_env_kwargs, resolve=True)
            or {}
        ),
    )

    # Baseline performance metrics of the environment.
    baseline = env.core.baselines
    baseline_metrics = {f"baselines/{k}": v for k, v in baseline.items()}

    # ----- create learning + inference agents
    # The learning agent updates weights; inference agent only evaluates checkpoints.
    alg_config = OmegaConf.to_container(cfg, resolve=True)
    learning_agent = imprl.agents.make(cfg.ALGORITHM, env, alg_config, device)
    inference_agent = imprl.agents.make(
        cfg.ALGORITHM, inference_env, alg_config, device
    )

    # ------ Weights & Biases setup ------
    wandb_cfg = OmegaConf.to_container(cfg.WANDB, resolve=True) or {}
    run = wandb.init(**wandb_cfg)
    # allow storing runtime metadata back into the config
    OmegaConf.set_struct(cfg.WANDB, False)
    cfg.WANDB.run_id = run.id  # store run_id in cfg for reference
    progress.log("wandb run with ID: %s", run.id)

    # ----- set up checkpoint directories for this run
    checkpoint_dir = Path(cfg.CHECKPOINT_DIR) / run.id
    cfg.CHECKPOINT_DIR = str(checkpoint_dir)
    progress.log("Checkpoint directory set to: %s", checkpoint_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_weights_dir = checkpoint_dir / "model_weights"
    model_weights_dir.mkdir(parents=True, exist_ok=True)
    # add config to checkpoint dir (required for inference)
    (checkpoint_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    # Store full resolved config in wandb and set summary metric behavior.
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
    wandb.define_metric("inference_mean", summary="min")

    # Evaluate in parallel across CPU cores for faster inference rollouts.
    worker_count = min(DEFAULT_WORKER_COUNT, cfg.NUM_INFERENCE_EPISODES)
    progress.log("Running inference with %d worker processes", worker_count)

    # Track the best checkpoint according to lowest inference mean cost.
    best_cost = math.inf
    best_checkpt = 0

    # ----- main training loop
    for ep in range(cfg.NUM_TRAIN_EPISODES):
        inference_metrics = {}

        # 1) Train for one episode.
        training_rollout(env, learning_agent)

        # 2) Save model weights periodically.
        if is_time(ep, cfg.CHECKPOINT_INTERVAL, cfg):
            learning_agent.save_weights(str(model_weights_dir), ep)
            progress.log(
                "Saved checkpoint at episode %d",
                ep,
                completed=ep + 1,
            )

        # 3) Run periodic evaluation using the inference agent.
        if is_time(ep, cfg.EVAL_INTERVAL, cfg):
            inference_agent.load_weights(str(model_weights_dir), ep)
            eval_costs = parallel_agent_rollout(
                inference_env,
                inference_agent,
                cfg.NUM_INFERENCE_EPISODES,
                num_workers=worker_count,
            )
            inference_mean = np.mean(eval_costs)
            inference_stderr = np.std(eval_costs) / np.sqrt(len(eval_costs))

            if inference_mean < best_cost:
                best_cost, best_checkpt = inference_mean, ep
                progress.log(
                    "New best cost %.3f at checkpoint %d",
                    inference_mean,
                    ep,
                    completed=ep + 1,
                )

            # Evaluation metrics for wandb.
            inference_metrics = {
                "inference_ep": ep,
                "inference_mean": inference_mean,
                "inference_stderr": inference_stderr,
                "best_cost": best_cost,
                "best_checkpt": best_checkpt,
            }

        # 4) Log training + baseline + evaluation metrics.
        if is_time(ep, cfg.LOGGING_INTERVAL, cfg):
            log_payload = {
                **learning_agent.logger,
                **baseline_metrics,
                **inference_metrics,
            }
            wandb.log(log_payload, step=ep)
            progress.log(
                "episode: %6d | episode_cost: %6.2f",
                ep,
                learning_agent.logger["episode_cost"],
            )

    # ----- finalize wandb run summary and close wandb cleanly
    if wandb.run is not None:
        wandb.run.summary["best_cost"] = best_cost
        wandb.run.summary["best_checkpt"] = best_checkpt
    total_training_time = time.strftime(
        "%H:%M:%S", time.gmtime(max(0, int(time.perf_counter() - progress.start_time)))
    )
    progress.log(
        "Best cost %.6f at episode %d | total training time %s",
        best_cost,
        best_checkpt,
        total_training_time,
    )

    wandb.finish()


if __name__ == "__main__":
    # "spawn" is the safest start method for multiprocessing with torch.
    mp.set_start_method("spawn", force=True)
    main()
