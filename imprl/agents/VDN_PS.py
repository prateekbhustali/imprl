"""
Single-file implementation of VDN-PS.

VDN-PS: Value-Decomposition Network with Parameter Sharing.
This module implements an off-policy value-based multi-agent learner with:
- a shared local Q-network (one Q-head per agent observation), and
- additive value decomposition:
  Q_tot = sum_i Q_i(b_i, a_i)
"""

import random
import numpy as np
import torch

from imprl.agents.primitives.exploration_schedulers import LinearExplorationScheduler
from imprl.agents.primitives.replay_memory import AbstractReplayMemory
from imprl.agents.primitives.MLP import NeuralNetwork


class ValueDecompositionNetworkParameterSharing:
    name = "VDN-PS"  # display names used by experiment scripts/loggers.
    full_name = "Value-Decomposition Network with Parameter Sharing"

    # Algorithm taxonomy.
    paradigm = "CTDE"
    formulation = "Dec-POMDP"
    algorithm_type = "value-based"
    policy_regime = "off-policy"
    policy_type = "epsilon-greedy"

    # Training/runtime properties.
    uses_replay_memory = True
    parameter_sharing = True

    def __init__(self, env, config, device):
        """Initialize shared Q/target networks, replay, exploration, and counters."""
        assert (
            env.__class__.__name__ == "MultiAgentWrapper"
        ), "VDN-PS only supports multi-agent environments"

        # ---------- Core references and counters ----------
        self.env, self.device, self.cfg = env, device, config
        self.episode = 0
        self.total_time = 0
        self.time = 0
        self.episode_return = 0

        # ---------- Evaluation discount + replay ----------
        try:
            self.eval_discount_factor = env.core.discount_factor
        except AttributeError:
            self.eval_discount_factor = 1.0

        self.replay_memory = AbstractReplayMemory(self.cfg["MAX_MEMORY_SIZE"])
        # Set a short warmup period to populate replay before training starts.
        self.warmup_threshold = 10 * self.cfg["BATCH_SIZE"]

        # ---------- Exploration ----------
        self.exploration_strategy = self.cfg["EXPLORATION_STRATEGY"]
        self.exploration_param = self.exploration_strategy["max_value"]
        self.exp_scheduler = LinearExplorationScheduler(
            self.exploration_strategy["min_value"],
            num_episodes=self.exploration_strategy["num_episodes"],
        )

        # ---------- Target-network update cadence ----------
        self.target_network_reset = self.cfg["TARGET_NETWORK_RESET"]

        # ---------- Environment dimensions ----------
        self.n_agent_actions = env.action_space.n
        self.n_agents = env.n_agents

        obs, _ = env.reset()
        local_obs = env.multiagent_idx_percept(obs)
        n_inputs = local_obs.shape[-1]

        # ---------- Network architectures ----------
        network_arch = [n_inputs] + self.cfg["NETWORK_CONFIG"]["hidden_layers"]
        network_arch += [self.n_agent_actions]

        # ---------- Q / target modules ----------
        self.q_network = NeuralNetwork(
            network_arch,
            initialization="orthogonal",
            optimizer=self.cfg["NETWORK_CONFIG"]["optimizer"],
            learning_rate=self.cfg["NETWORK_CONFIG"]["lr"],
            lr_scheduler=self.cfg["NETWORK_CONFIG"]["lr_scheduler"],
        ).to(device)
        self.target_network = NeuralNetwork(network_arch).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # ---------- Logging fields ----------
        self.logger = {
            "TD_loss": None,
            "learning_rate": self.cfg["NETWORK_CONFIG"]["lr"],
            "exploration_param": self.exploration_param,
        }

    def reset_episode(self, training=True):
        """Reset episode counters; update epsilon/LR and sync target network on schedule."""
        self.episode_return = 0
        self.episode += 1
        self.time = 0

        if not training:
            return

        # Epsilon schedule for behavior policy.
        self.exploration_param = self.exp_scheduler.step()
        self.logger["exploration_param"] = self.exploration_param

        # Start scheduler/target updates only when replay batches are trainable.
        if self.total_time > self.warmup_threshold:
            if self.episode % self.target_network_reset == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            self.q_network.lr_scheduler.step()
            self.logger["learning_rate"] = self.q_network.lr_scheduler.get_last_lr()[0]

    def select_action(self, observation, training):
        # Epsilon-greedy exploration during training.
        if (
            training
            and self.exploration_strategy["name"] == "epsilon_greedy"
            and self.exploration_param > random.random()
        ):
            action = [self.env.action_space.sample() for _ in range(self.n_agents)]
            return action, torch.tensor(action)

        # Greedy local actions from shared Q-network.
        local_obs = self.env.multiagent_idx_percept(observation)
        t_local_obs = torch.as_tensor(local_obs, device=self.device)
        q_values = self.q_network.forward(t_local_obs, training=training)
        t_action = torch.argmax(q_values, dim=-1)
        action = t_action.cpu().detach().numpy()
        if training:
            return action, t_action.detach().cpu()
        return action

    def process_experience(
        self, belief, t_action, next_belief, reward, terminated, truncated
    ):
        """Store transition, run one replay update after warmup, and log episode end."""
        # Update episode return/time counters.
        self.process_rewards(reward)

        self.replay_memory.store_experience(
            self.env.multiagent_idx_percept(belief),
            t_action,
            self.env.multiagent_idx_percept(next_belief),
            reward,
            terminated,
            truncated,
        )

        # Train from replay after a short warmup.
        if self.total_time > self.warmup_threshold:
            sample_batch = self.replay_memory.sample_batch(self.cfg["BATCH_SIZE"])
            self.train(*sample_batch)

        # Episode-level logging.
        if terminated or truncated:
            self.logger["episode"] = self.episode
            self.logger["episode_cost"] = -self.episode_return

    def compute_loss(self, *args):
        """
        Compute TD loss for one replay batch using VDN mixing.

        Core steps:
        - Current values use replayed local actions:
          Q_tot(s, a) = sum_i Q_i(o_i, a_i)
        - Future values use per-agent greedy actions from target network:
          Q_tot(s', a') = sum_i Q_i_target(o'_i, argmax_a Q_i_target(o'_i, a))
        - TD target:
          y = r + gamma * (1 - done) * Q_tot(s', a')
        """
        beliefs, actions, next_beliefs, rewards, terminations, _truncations = args

        # ---------- Tensorize replay samples ----------
        t_beliefs = torch.as_tensor(np.asarray(beliefs), device=self.device)
        t_actions = torch.stack(actions).to(self.device)
        t_next_beliefs = torch.as_tensor(np.asarray(next_beliefs), device=self.device)
        t_rewards = torch.as_tensor(rewards, device=self.device).reshape(-1, 1)
        t_terminations = torch.as_tensor(
            terminations, dtype=torch.int, device=self.device
        ).reshape(-1, 1)

        # ---------- Current Q_tot estimates ----------
        q_values = self.q_network.forward(t_beliefs)
        chosen_q_values = torch.gather(q_values, dim=2, index=t_actions.unsqueeze(2))
        current_values = chosen_q_values.sum(dim=1)

        # ---------- VDN targets ----------
        with torch.no_grad():
            target_q_values = self.target_network.forward(t_next_beliefs)
            best_actions = torch.argmax(target_q_values, dim=2)
            future_values = torch.gather(
                target_q_values, dim=2, index=best_actions.unsqueeze(2)
            )
            future_values = future_values.sum(dim=1)
            not_terminals = 1 - t_terminations
            td_targets = (
                t_rewards + self.cfg["DISCOUNT_FACTOR"] * future_values * not_terminals
            )

        return self.q_network.loss_function(current_values, td_targets)

    def train(self, *args):
        """Run one gradient update for Q-network from one replay batch."""
        loss = self.compute_loss(*args)

        self.q_network.optimizer.zero_grad()
        loss.backward()
        self.q_network.optimizer.step()

        self.logger["TD_loss"] = loss.detach()

    def process_rewards(self, reward):
        """Accumulate discounted episode return and advance time counters."""
        self.episode_return += reward * self.eval_discount_factor**self.time
        self.time += 1
        self.total_time += 1

    def report(self, stats=None):
        """Print episode-level progress to stdout."""
        print(f"Ep:{self.episode:05}| Cost: {-self.episode_return:.2f}", flush=True)
        if stats is not None:
            print(stats)

    def save_weights(self, path, episode):
        """Save Q-network parameters for a checkpoint id."""
        torch.save(self.q_network.state_dict(), f"{path}/q_network_{episode}.pth")

    def load_weights(self, path, episode):
        """Load Q-network parameters from a checkpoint id."""
        self.q_network.load_state_dict(
            torch.load(
                f"{path}/q_network_{episode}.pth", map_location=torch.device("cpu")
            )
        )
