"""
A finite-horizon k-out-of-n environment.

This variant preserves the version used in the paper:
"Assessing the Optimality of Decentralized Inspection and Maintenance
Policies for Stochastically Degrading Engineering Systems".

Why this differs from `k_out_of_n_infinite`:
1. Finite horizon with explicit time perception.
2. Failure is self-announcing.
3. Rewards:
   - No mobilisation rewards,
   - Time-discounted rewards at each step,
   - No reward shaping.
4. Deterministic reset to all components in the healthiest state.
"""

import numpy as np

from imprl.envs.structural_envs.k_out_of_n_infinite import KOutOfN as BaseKOutOfN


class KOutOfN(BaseKOutOfN):
    """Finite-horizon classic k-out-of-n env built on the infinite base."""

    def __init__(
        self,
        env_config: dict,
        baselines: dict = None,
        percept_type: str = "belief",
        reward_shaping: bool = False,
    ) -> None:
        cfg = env_config

        super().__init__(
            cfg,
            baselines=baselines,
            percept_type=percept_type,
            time_limit=cfg["time_horizon"],
            time_perception=True,
            reward_shaping=reward_shaping,
            return_discounted_rewards=False,
            is_failure_self_announcing=True,
        )

    def get_reward(
        self, state: list, belief: np.array, action: list, next_belief: np.array
    ) -> tuple[float, float, float, float, float]:
        _, reward_replacement, reward_inspection, reward_system, _ = super().get_reward(
            state, belief, action, next_belief
        )

        reward_replacement *= self.discount_factor**self.time
        reward_inspection *= self.discount_factor**self.time
        reward_system *= self.discount_factor**self.time

        reward = reward_replacement + reward_inspection + reward_system

        return reward, reward_replacement, reward_inspection, reward_system, 0.0

    def step(self, action: list) -> tuple[np.array, float, bool, bool, dict]:
        percept, reward, _terminated, truncated, info = super().step(action)
        terminated = truncated
        info["reward_mobilisation"] = 0.0
        return percept, reward, terminated, False, info

    def reset(self, **kwargs) -> tuple[np.array, np.array]:
        initial_damage = 0
        self.damage_state = np.array([initial_damage] * self.n_components, dtype=int)
        self.observation = np.array([initial_damage] * self.n_components, dtype=int)

        self.belief = np.zeros((self.n_components, self.n_damage_states))
        self.belief[:, initial_damage] = 1.0

        self.time = 0
        self.norm_time = self.time / self.time_limit

        info = {
            "system_failure": False,
            "reward_replacement": 0.0,
            "reward_inspection": 0.0,
            "reward_system": 0.0,
            "reward_mobilisation": 0.0,
            "state": self._get_state(),
            "observation": self.observation,
        }
        return self._get_percept(), info
