import itertools
import numpy as np

import gymnasium as gym
from gymnasium import spaces


class SingleAgentWrapper(gym.Env):

    def __init__(self, env) -> None:

        self.core = env
        self.dtype = np.float32  # or np.float64

        # Gym Spaces (joint spaces)

        # Percept space (belief/state/obs space)
        # Structural envs emit state/observation as one-hot matrices per component,
        # so we model state space as a Box instead of MultiDiscrete indices.
        self.state_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.core.n_components, self.core.n_damage_states),
            dtype=self.dtype,
        )
        self.belief_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.core.n_damage_states * self.core.n_components,),
            dtype=self.dtype,
        )
        if env.time_perception:
            self.state_space = gym.spaces.Tuple(
                (
                    # normalized time
                    gym.spaces.Box(0, 1, shape=(1,), dtype=self.dtype),
                    # state space
                    self.state_space,
                )
            )
            self.belief_space = gym.spaces.Tuple(
                (
                    # normalized time
                    gym.spaces.Box(0, 1, shape=(1,), dtype=self.dtype),
                    # belief space
                    self.belief_space,
                )
            )

        self.observation_space = self.state_space

        if self.core.percept_type in ["belief", "Belief"]:
            self.perception_space = self.belief_space
        elif self.core.percept_type in ["state", "State"]:
            self.perception_space = self.state_space
        elif self.core.percept_type in ["obs", "Obs"]:
            self.perception_space = self.observation_space

        # Action space
        self.action_space = spaces.Discrete(
            self.core.n_comp_actions**self.core.n_components
        )

        _action_space = list(
            itertools.product(
                np.arange(self.core.n_comp_actions), repeat=self.core.n_components
            )
        )
        self.joint_action_map = [list(action) for action in _action_space]

        self.perception_dim = gym.spaces.utils.flatdim(self.perception_space)
        self.action_dim = gym.spaces.utils.flatdim(self.action_space)

    def step(self, action: int):
        action = self.joint_action_map[action]

        next_percept, reward, terminated, truncated, info = self.core.step(action)

        return (
            self.system_percept(next_percept),
            self.dtype(reward),
            terminated,
            truncated,
            info,
        )

    def reset(self, **kwargs):
        obs, info = self.core.reset(**kwargs)
        return self.system_percept(obs), info

    def system_percept(self, percept):
        # Keep the legacy single-agent flattening order `(damage_state, component)`
        # so older single-agent checkpoints remain compatible after the core moved
        # to component-major `(component, damage_state)` beliefs/states.
        if self.core.time_perception:
            time, local_percept = percept
            percept = (time, local_percept.T)
        else:
            percept = percept.T
        return spaces.utils.flatten(self.perception_space, percept)
