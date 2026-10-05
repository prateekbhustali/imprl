"""
Single-file implementation of DDQN.

DDQN: Double Deep Q-Network.
This module implements an off-policy value-based agent with:
- epsilon-greedy behavior policy for exploration, and
- Double-DQN targets using online argmax + target-network evaluation.

Architecture:
    Q-network (state/belief -> action values):
    [b] -> [Q] -> [Q(a1), ..., Q(aK)]

    Target network (same shape, delayed updates):
    [b'] -> [Q_target] -> [Q_target(a1), ..., Q_target(aK)]
"""

import random
import numpy as np
import torch

from imprl.agents.primitives.exploration_schedulers import LinearExplorationScheduler
from imprl.agents.primitives.replay_memory import AbstractReplayMemory
from imprl.agents.primitives.MLP import NeuralNetwork


class DDQNAgent:
    name = "DDQN"  # display names used by experiment scripts/loggers.
    full_name = "Double Deep Q-Network"

    # Algorithm taxonomy.
    paradigm = "CTCE"
    formulation = "POMDP"
    algorithm_type = "value-based"
    policy_regime = "off-policy"
    policy_type = "epsilon-greedy"

    # Training/runtime properties.
    uses_replay_memory = True
    parameter_sharing = True

    def __init__(self, env, config, device):
        """Initialize Q/target networks, replay memory, exploration, and counters."""
        assert (
            env.__class__.__name__ == "SingleAgentWrapper"
        ), "DDQN only supports single-agent environments"

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

        # ---------- Network architectures ----------
        n_inputs = self.env.perception_dim
        n_outputs = self.env.action_dim
        network_arch = [n_inputs] + self.cfg["NETWORK_CONFIG"]["hidden_layers"]
        network_arch += [n_outputs]

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
            action = self.env.action_space.sample()
            return action, torch.tensor(action)

        # Greedy action from online Q-network.
        t_observation = torch.as_tensor(observation, device=self.device)
        q_values = self.q_network.forward(t_observation, training=training)
        t_action = torch.argmax(q_values)
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
            belief,
            t_action,
            next_belief,
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
        Compute TD loss for one replay batch using Double-DQN targets.

        Core steps:
        - Current values use online network and replayed actions:
          Q(s, a)
        - Next-action selection uses online network:
          a* = argmax_a Q_online(s', a)
        - Next-action evaluation uses target network:
          Q_target(s', a*)
        - TD target:
          y = r + gamma * (1 - done) * Q_target(s', a*)
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

        # ---------- Current Q estimates ----------
        q_values = self.q_network.forward(t_beliefs)
        current_values = torch.gather(q_values, dim=1, index=t_actions.unsqueeze(1))

        # ---------- Double-DQN targets ----------
        with torch.no_grad():
            q_next_online = self.q_network.forward(t_next_beliefs)
            best_next_actions = torch.argmax(q_next_online, dim=1, keepdim=True)
            q_next_target = self.target_network.forward(t_next_beliefs)
            future_values = torch.gather(
                q_next_target, dim=1, index=best_next_actions
            )
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
