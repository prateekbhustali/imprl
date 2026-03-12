"""
Single-file implementation of IAC-PS.

IAC-PS: Independent Actor-Critic with Parameter Sharing.
This module implements an off-policy actor-critic for cooperative multi-agent
control with:
- a shared local actor head (one action per agent), and
- a shared local critic head (one value per agent).

Architecture:
    Critic (local+id per agent -> local values):
    [[b1|id1],...,[bN|idN]] -> [SHARED CRITIC] -> [V1,...,VN]

    Actor (local+id per agent, shared parameters):
    [[b1|id1],...,[bN|idN]] -> [SHARED ACTOR] -> (A1 x ... x AN)
"""

import random
import numpy as np
import torch

from imprl.agents.primitives.replay_memory import AbstractReplayMemory
from imprl.agents.primitives.exploration_schedulers import LinearExplorationScheduler
from imprl.agents.primitives.ActorNetwork import ActorNetwork
from imprl.agents.primitives.MLP import NeuralNetwork


class IndependentActorCriticParameterSharing:
    name = "IAC-PS"  # display names used by experiment scripts/loggers.
    full_name = "Independent Actor-Critic with Parameter Sharing"

    # Algorithm taxonomy.
    paradigm = "DTDE"
    formulation = "Dec-POMDP"
    algorithm_type = "actor-critic"
    policy_regime = "off-policy"
    policy_type = "stochastic"

    # Training/runtime properties.
    uses_replay_memory = True
    parameter_sharing = True

    def __init__(self, env, config, device):
        """Initialize shared local actor/critic, replay buffer, and counters."""
        assert (
            env.__class__.__name__ == "MultiAgentWrapper"
        ), "IAC-PS only supports multi-agent environments"

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
        local_obs = env.multiagent_idx_percept(obs)
        n_inputs = local_obs.shape[-1]

        # ---------- Network architectures ----------
        actor_arch = [n_inputs] + self.cfg["ACTOR_CONFIG"]["hidden_layers"]
        actor_arch += [self.n_agent_actions]
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
            action = [self.env.action_space.sample() for _ in range(self.n_agents)]
            t_action = torch.as_tensor(action, device=self.device)
            action_prob = torch.tensor(
                (1.0 / self.n_agent_actions) ** self.n_agents, device=self.device
            )
            return action, t_action.detach().cpu(), action_prob.detach().cpu()

        # Shared actor over local+id observations for each agent.
        local_obs = self.env.multiagent_idx_percept(observation)
        t_local_obs = torch.as_tensor(local_obs, device=self.device)
        action_dist = self.actor.forward(t_local_obs, training=training)

        # Joint action sampled from per-agent categorical heads.
        t_action = action_dist.sample()
        action = t_action.cpu().detach().numpy()
        if not training:
            return action

        # Proposal probability used for importance correction in replay updates.
        # Assumption: conditioned on B, per-agent actions are independent, so
        # joint behavior probability is the product of per-agent probabilities.
        log_probs = action_dist.log_prob(t_action)
        action_prob = torch.exp(log_probs).prod(dim=-1)
        return action, t_action.detach().cpu(), action_prob.detach().cpu()

    def process_experience(
        self, belief, t_action, action_prob, next_belief, reward, terminated, truncated
    ):
        """Store transition, run one replay update after warmup, and log episode end."""
        # Update episode return/time counters.
        self.process_rewards(reward)

        self.replay_memory.store_experience(
            self.env.multiagent_idx_percept(belief),
            t_action,
            action_prob,
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
        Compute actor and critic objectives from one replay batch.

        Core steps:
        - Build TD(0) targets with shared local critic:
          td_target_i = r + gamma * V_i(B_next) * (1 - done)

        - Evaluate current joint log-probability under shared actor:
          log pi_theta(a|B) = sum_i log pi_theta(a_i|b_i,id_i)

        - Apply clipped importance weighting for off-policy correction:
          w = clip(pi_theta(a|B) / mu(a|B), max=importance_clip_max)
          where mu(a|B) is the replayed action_probs value.
        """
        (
            ma_beliefs,
            actions,
            action_probs,
            ma_next_beliefs,
            rewards,
            terminations,
            _truncations,
        ) = args

        # ---------- Tensorize replay samples ----------
        t_ma_beliefs = torch.as_tensor(np.asarray(ma_beliefs), device=self.device)
        t_actions = torch.stack(actions).to(self.device)
        t_action_probs = torch.as_tensor(action_probs, device=self.device)
        t_ma_next_beliefs = torch.as_tensor(
            np.asarray(ma_next_beliefs), device=self.device
        )
        t_rewards = torch.as_tensor(rewards, device=self.device).reshape(-1, 1)
        t_terminations = torch.as_tensor(
            terminations, dtype=torch.int, device=self.device
        ).reshape(-1, 1)

        # ---------- Critic targets ----------
        current_values = self.critic.forward(t_ma_beliefs).squeeze(-1)
        with torch.no_grad():
            next_values = self.critic.forward(t_ma_next_beliefs).squeeze(-1)
            not_terminals = 1 - t_terminations
            td_targets = (
                t_rewards + self.cfg["DISCOUNT_FACTOR"] * next_values * not_terminals
            )

        # ---------- Policy likelihood + importance weights ----------
        action_dists = self.actor.forward(t_ma_beliefs)
        log_probs = action_dists.log_prob(t_actions)
        joint_log_probs = log_probs.sum(dim=-1)
        probs_new = torch.exp(joint_log_probs)
        weights = (
            torch.clamp(probs_new / t_action_probs, max=self.importance_clip_max)
            .detach()
            .reshape(-1, 1)
        )

        # ---------- Losses ----------
        # critic_loss = sum_i E[w * (V_i(B_i) - td_target_i)^2]
        critic_loss = torch.mean(
            torch.square(current_values - td_targets) * weights, dim=0
        ).sum()

        # actor_loss = E[-sum_i log pi_theta(a_i|B_i) * advantage_i * w]
        # where advantage_i = (td_target_i - V_i(B_i)) detached from gradients.
        advantage = (td_targets - current_values).detach()
        actor_loss = torch.mean(
            -torch.sum(log_probs * advantage, dim=1, keepdim=True) * weights
        )
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
