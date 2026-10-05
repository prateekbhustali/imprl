"""
Single-file implementation of JAC.

JAC: Joint Actor-Critic.
This module implements an off-policy actor-critic for single-agent wrappers
where the action represents a joint decision.

Architecture:
    Critic (belief/state -> scalar):
    [b1|...|bN] -> [CRITIC] -> V in R

    Actor (belief/state -> action index):
    [b1|...|bN] -> [ACTOR] -> A
"""

import random
import numpy as np
import torch

from imprl.agents.primitives.exploration_schedulers import LinearExplorationScheduler
from imprl.agents.primitives.replay_memory import AbstractReplayMemory
from imprl.agents.primitives.ActorNetwork import ActorNetwork
from imprl.agents.primitives.MLP import NeuralNetwork


class JointActorCritic:
    name = "JAC"  # display names used by experiment scripts/loggers.
    full_name = "Joint Actor-Critic"

    # Algorithm taxonomy.
    paradigm = "CTCE"
    formulation = "POMDP"
    algorithm_type = "actor-critic"
    policy_regime = "off-policy"
    policy_type = "stochastic"

    # Training/runtime properties.
    uses_replay_memory = True
    parameter_sharing = True

    def __init__(self, env, config, device):
        """Initialize centralized actor/critic networks and optimizer metadata."""
        assert (
            env.__class__.__name__ == "SingleAgentWrapper"
        ), "JAC only supports single-agent environments"

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
        # Clip ratio for importance weights; configurable to reduce variance.
        self.importance_clip_max = self.cfg.get("IMPORTANCE_WEIGHT_CLIP_MAX", 2.0)
        self.exploration_strategy = self.cfg["EXPLORATION_STRATEGY"]
        self.exploration_param = self.exploration_strategy["max_value"]
        self.exp_scheduler = LinearExplorationScheduler(
            self.exploration_strategy["min_value"],
            num_episodes=self.exploration_strategy["num_episodes"],
        )

        # ---------- Network architectures ----------
        n_inputs = self.env.perception_dim
        actor_arch = [n_inputs] + self.cfg["ACTOR_CONFIG"]["hidden_layers"]
        actor_arch += [self.env.action_dim]
        critic_arch = [n_inputs] + self.cfg["CRITIC_CONFIG"]["hidden_layers"] + [1]

        # ---------- Actor / critic modules ----------
        self.actor = ActorNetwork(
            actor_arch,
            initialization="orthogonal",
            optimizer=self.cfg["ACTOR_CONFIG"]["optimizer"],
            learning_rate=self.cfg["ACTOR_CONFIG"]["lr"],
            lr_scheduler=self.cfg["ACTOR_CONFIG"]["lr_scheduler"],
        ).to(device)
        self.critic = NeuralNetwork(
            critic_arch,
            initialization="orthogonal",
            optimizer=self.cfg["CRITIC_CONFIG"]["optimizer"],
            learning_rate=self.cfg["CRITIC_CONFIG"]["lr"],
            lr_scheduler=self.cfg["CRITIC_CONFIG"]["lr_scheduler"],
        ).to(device)

        # ---------- Logging fields ----------
        self.logger = {
            "actor_loss": None,
            "critic_loss": None,
            "lr_actor": self.cfg["ACTOR_CONFIG"]["lr"],
            "lr_critic": self.cfg["CRITIC_CONFIG"]["lr"],
            "exploration_param": self.exploration_param,
        }

    def reset_episode(self, training=True):
        """Reset episode counters and step LR schedules once replay warmup is complete."""
        self.episode_return = 0
        self.episode += 1
        self.time = 0

        if training:
            self.exploration_param = self.exp_scheduler.step()
            self.logger["exploration_param"] = self.exploration_param

            # Start scheduler updates only when replay batches are trainable.
            if self.total_time > self.warmup_threshold:
                self.actor.lr_scheduler.step()
                self.critic.lr_scheduler.step()
                self.logger["lr_actor"] = self.actor.lr_scheduler.get_last_lr()[0]
                self.logger["lr_critic"] = self.critic.lr_scheduler.get_last_lr()[0]

    def select_action(self, observation, training):
        if (
            training
            and self.exploration_strategy["name"] == "epsilon_greedy"
            and self.exploration_param > random.random()
        ):
            action = self.env.action_space.sample()
            t_action = torch.as_tensor(action, device=self.device)
            action_prob = torch.tensor(1.0 / self.env.action_dim, device=self.device)
            return action, t_action.detach().cpu(), action_prob.detach().cpu()

        t_observation = torch.as_tensor(observation, device=self.device)
        action_dist = self.actor.forward(t_observation, training=training)

        t_action = action_dist.sample()
        action = t_action.cpu().detach().numpy()
        if not training:
            return action

        # Proposal probability used for importance correction in replay updates.
        # Assumption: conditioned on B, per-agent actions are independent, so
        # joint behavior probability is the product of per-agent probabilities.
        log_prob = action_dist.log_prob(t_action)
        action_prob = torch.exp(log_prob)
        return action, t_action.detach().cpu(), action_prob.detach().cpu()

    def process_experience(
        self, belief, t_action, action_prob, next_belief, reward, terminated, truncated
    ):
        """Store transition, run one replay update after warmup, and log episode end."""
        # Update episode return/time counters.
        self.process_rewards(reward)

        self.replay_memory.store_experience(
            belief,
            t_action,
            action_prob,
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
        Compute actor and critic objectives from one replay batch.

        Core steps:
        - Build TD(0) targets with centralized critic:
          td_target = r + gamma * V(B_next) * (1 - done)

        - Evaluate current joint log-probability under actor:
          log pi_theta(a|B) = sum_i log pi_theta(a_i|B)

        - Apply clipped importance weighting for off-policy correction:
          w = clip(pi_theta(a|B) / mu(a|B), max=importance_clip_max)
          where mu(a|B) is the replayed action_probs value.
        """
        (
            beliefs,
            actions,
            action_probs,
            next_beliefs,
            rewards,
            terminations,
            _truncations,
        ) = args

        # ---------- Tensorize replay samples ----------
        t_beliefs = torch.as_tensor(np.asarray(beliefs), device=self.device)
        t_actions = torch.stack(actions).to(self.device)
        t_action_probs = torch.as_tensor(action_probs, device=self.device)
        t_next_beliefs = torch.as_tensor(np.asarray(next_beliefs), device=self.device)
        t_rewards = torch.as_tensor(rewards, device=self.device).reshape(-1)
        t_terminations = torch.as_tensor(
            terminations, dtype=torch.int, device=self.device
        )

        # ---------- Critic targets ----------
        current_values = self.critic.forward(t_beliefs).squeeze(-1)
        with torch.no_grad():
            next_values = self.critic.forward(t_next_beliefs).squeeze(-1)
            not_terminals = 1 - t_terminations
            td_targets = (
                t_rewards + self.cfg["DISCOUNT_FACTOR"] * next_values * not_terminals
            )

        # ---------- Policy likelihood + importance weights ----------
        action_dists = self.actor.forward(t_beliefs)
        log_probs = action_dists.log_prob(t_actions)
        probs_new = torch.exp(log_probs)
        weights = torch.clamp(
            probs_new / t_action_probs, max=self.importance_clip_max
        ).detach()

        # ---------- Losses ----------
        # critic_loss = E[w * (V(B) - td_target)^2]
        critic_loss = torch.mean(weights * torch.square(current_values - td_targets))

        # actor_loss = E[-log pi_theta(a|B) * advantage * w]
        # where advantage = (td_target - V(B)) detached from gradients.
        advantage = (td_targets - current_values).detach()
        actor_loss = torch.mean(-log_probs * advantage * weights)
        return actor_loss, critic_loss

    def train(self, *args):
        """Run one gradient update for actor and critic from one replay batch."""
        actor_loss, critic_loss = self.compute_loss(*args)

        # Actor step.
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Critic step.
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        self.logger["actor_loss"] = actor_loss.detach()
        self.logger["critic_loss"] = critic_loss.detach()

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
        """Save actor and critic parameters for a checkpoint id."""
        torch.save(self.actor.state_dict(), f"{path}/actor_{episode}.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic_{episode}.pth")

    def load_weights(self, path, episode):
        """Load actor and critic parameters from a checkpoint id."""
        self.actor.load_state_dict(
            torch.load(f"{path}/actor_{episode}.pth", map_location=torch.device("cpu"))
        )
        self.critic.load_state_dict(
            torch.load(f"{path}/critic_{episode}.pth", map_location=torch.device("cpu"))
        )
