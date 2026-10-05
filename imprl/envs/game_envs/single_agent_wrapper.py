"""
A gymnasium

"""

import itertools
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Discrete, Tuple, Box


class SingleAgentWrapper(gym.Env):

    def __init__(self, env) -> None:

        self.core = env
        self.dtype = np.float32  # or np.float64

        # Gym Spaces (joint spaces)

        # Percept space
        self.local_observation_space = Box(
            0, 1, shape=(self.core.n_agents,), dtype=self.dtype
        )
        if env.time_perception:
            # add time to the observation space
            self.shared_observation_space = Box(0, 1, shape=(1,), dtype=self.dtype)
            self.perception_space = Tuple(
                (self.shared_observation_space, self.local_observation_space)
            )
        else:
            self.perception_space = self.local_observation_space

        # Action space
        self.action_space = spaces.Discrete(
            np.prod(np.array(self.core.per_agent_actions))
        )

        # Joint action space
        _iterables = [range(k) for k in self.core.per_agent_actions]
        _action_space = list(itertools.product(*_iterables))
        self.joint_action_map = [list(action) for action in _action_space]

        self.perception_dim = gym.spaces.utils.flatdim(self.perception_space)
        self.action_dim = gym.spaces.utils.flatdim(self.action_space)

    def system_percept(self, percept):

        if self.core.time_perception:
            local_percept = spaces.utils.flatten(
                self.local_observation_space, percept[1]
            )
            local_percept = local_percept.flatten()
            shared_percept = spaces.utils.flatten(
                self.shared_observation_space, percept[0]
            )
            return np.concatenate((shared_percept, local_percept), axis=0)
        else:
            return spaces.utils.flatten(self.local_observation_space, percept)

    def _cast(self, percept):
        # with time perception the percept is (time, obs)
        if self.core.time_perception:
            return tuple(np.asarray(x, dtype=self.dtype) for x in percept)
        return percept.astype(self.dtype)

    def step(self, action: int):
        actions = self.joint_action_map[action]
        next_percept, reward, terminated, truncated, info = self.core.step(actions)
        return (
            self._cast(next_percept),
            self.dtype(reward),
            terminated,
            truncated,
            info,
        )

    def reset(self, **kwargs):
        next_percept, info = self.core.reset(**kwargs)
        return self._cast(next_percept), info
