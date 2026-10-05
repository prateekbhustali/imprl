"""
Single-file implementation of DDMAC.

DDMAC: Deep Decentralised Multi-Agent Actor-Critic.
This module implements an off-policy actor-critic with:
- independent actor networks per agent (no parameter sharing), and
- a centralized critic over the system belief/state.

Architecture:
    Critic (centralized -> scalar):
    [b1|...|bN] -> [CRITIC] -> V in R

    Actors (decentralized, one network per agent):
    [b1|...|bN] -> [ACTOR_1,...,ACTOR_N] -> (A1 x ... x AN)
"""

import random
import numpy as np
import torch

from imprl.agents.primitives.exploration_schedulers import LinearExplorationScheduler
from imprl.agents.primitives.replay_memory import AbstractReplayMemory
from imprl.agents.primitives.MLP import NeuralNetwork
from imprl.agents.primitives.MultiAgentActors import MultiAgentActors


class DeepDecentralisedMultiAgentActorCritic:
    name = "DDMAC"
    full_name = "Deep Decentralised Multi-Agent Actor-Critic"

    # Algorithm taxonomy.
    paradigm = "CTCE"
    formulation = "MPOMDP"
    algorithm_type = "actor-critic"
    policy_regime = "off-policy"
    policy_type = "stochastic"

    # Training/runtime properties.
    uses_replay_memory = True
    parameter_sharing = False

    def __init__(self, env, config, device):
        """Initialize centralized actor/critic networks and optimizer metadata."""
        assert (
            env.__class__.__name__ == "MultiAgentWrapper"
        ), "DDMAC only supports multi-agent environments"

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

        # ---------- Environment dimensions ----------
        self.n_agent_actions = env.action_space.n
        self.n_agents = env.n_agents

        obs, _ = env.reset()
        system_obs = env.system_percept(obs)
        n_inputs = system_obs.shape[-1]

        # ---------- Network architectures ----------
        actor_arch = [n_inputs] + self.cfg["ACTOR_CONFIG"]["hidden_layers"]
        actor_arch += [self.n_agent_actions]
        critic_arch = [n_inputs] + self.cfg["CRITIC_CONFIG"]["hidden_layers"] + [1]

        # ---------- Actor / critic modules ----------
        self.actor = MultiAgentActors(
            self.n_agents,
            self.n_agent_actions,
            {
                "architecture": actor_arch,
                "optimizer": self.cfg["ACTOR_CONFIG"]["optimizer"],
                "lr": self.cfg["ACTOR_CONFIG"]["lr"],
                "lr_scheduler": self.cfg["ACTOR_CONFIG"]["lr_scheduler"],
            },
            device,
        )
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
            action = [self.env.action_space.sample() for _ in range(self.n_agents)]
            t_action = torch.as_tensor(action, device=self.device)
            action_prob = torch.tensor(
                (1.0 / self.n_agent_actions) ** self.n_agents, device=self.device
            )
            return action, t_action.detach().cpu(), action_prob.detach().cpu()

        system_obs = self.env.system_percept(observation)
        t_observation = torch.as_tensor(system_obs, device=self.device)

        if training:
            action, t_action, action_prob = self.actor.forward(
                t_observation, training=True, ind_obs=False
            )
            return action, t_action.detach().cpu(), action_prob.detach().cpu()

        return self.actor.forward(t_observation, training=False, ind_obs=False)

    def process_experience(
        self, belief, t_action, action_prob, next_belief, reward, terminated, truncated
    ):
        """Store transition, run one replay update after warmup, and log episode end."""
        # Update episode return/time counters.
        self.process_rewards(reward)

        self.replay_memory.store_experience(
            self.env.system_percept(belief),
            t_action,
            action_prob,
            self.env.system_percept(next_belief),
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

    def compute_log_prob(self, t_beliefs, t_actions):
        log_probs = torch.ones((t_beliefs.shape[0], self.n_agents)).to(self.device)
        for k, actor_network in enumerate(self.actor.networks):
            action_dists = actor_network.forward(t_beliefs)
            log_probs[:, k] = action_dists.log_prob(t_actions[:, k])
        return torch.sum(log_probs, dim=-1)

    def compute_loss(self, *args):
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
        t_action_probs = torch.as_tensor(action_probs, device=self.device).reshape(-1)
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
        joint_log_probs = self.compute_log_prob(t_beliefs, t_actions)
        probs_new = torch.exp(joint_log_probs)
        weights = torch.clamp(
            probs_new / t_action_probs, max=self.importance_clip_max
        ).detach()

        # ---------- Losses ----------
        # critic_loss = E[w * (V(B) - td_target)^2]
        critic_loss = torch.mean(weights * torch.square(current_values - td_targets))

        # actor_loss = E[-log pi_theta(a|B) * advantage * w]
        # where advantage = (td_target - V(B)) detached from gradients.
        advantage = (td_targets - current_values).detach()
        actor_loss = torch.mean(-joint_log_probs * advantage * weights)
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
        for i, actor_network in enumerate(self.actor.networks):
            torch.save(actor_network.state_dict(), f"{path}/actor_{i+1}_{episode}.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic_{episode}.pth")

    def load_weights(self, path, episode):
        """Load actor and critic parameters from a checkpoint id."""
        for i, actor_network in enumerate(self.actor.networks):
            actor_network.load_state_dict(
                torch.load(
                    f"{path}/actor_{i+1}_{episode}.pth",
                    map_location=torch.device("cpu"),
                )
            )
        self.critic.load_state_dict(
            torch.load(f"{path}/critic_{episode}.pth", map_location=torch.device("cpu"))
        )
