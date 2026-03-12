import numpy as np

from gymnasium import spaces
from gymnasium.spaces import Discrete, Tuple, Box


class MultiAgentWrapper:

    def __init__(self, env) -> None:
        self.core = env
        self.dtype = np.float32  # or np.float64
        self.n_agents = self.core.n_components

        if self.core.percept_type in ["obs", "Obs", "state", "State"]:
            self.local_observation_space = tuple(
                [Discrete(self.core.n_damage_states)] * self.core.n_components,
            )
        elif self.core.percept_type in ["belief", "Belief"]:
            self.local_observation_space = Box(
                0,
                1,
                shape=(self.core.n_components, self.core.n_damage_states),
                dtype=self.dtype,
            )

        self.local_observation_space = self.local_observation_space

        if env.time_perception:
            # add time to the observation space
            self.shared_observation_space = Box(0, 1, shape=(1,), dtype=self.dtype)
            self.perception_space = Tuple(
                (self.shared_observation_space, self.local_observation_space),
            )
        else:
            self.perception_space = self.local_observation_space

        self.action_space = Discrete(self.core.n_comp_actions)

    def step(self, actions):
        next_percept, reward, terminated, truncated, info = self.core.step(actions)
        return next_percept, self.dtype(reward), terminated, truncated, info

    def reset(self, **kwargs):
        next_percept, info = self.core.reset(**kwargs)
        return next_percept, info

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
            local_percept = spaces.utils.flatten(self.local_observation_space, percept)
            local_percept = local_percept.flatten()
            return local_percept

    def multiagent_percept(self, percept):
        if self.core.time_perception:
            local_percept = spaces.utils.flatten(
                self.local_observation_space, percept[1]
            )
            local_percept = local_percept.reshape(self.core.n_components, -1)
            shared_percept = spaces.utils.flatten(
                self.shared_observation_space, percept[0]
            )
            shared_percept = np.broadcast_to(shared_percept, (self.core.n_components, 1))

            return np.concatenate((shared_percept, local_percept), axis=1)
        else:
            return percept.astype(self.dtype)

    def multiagent_idx_percept(self, percept):

        # (id, shared_percept, local_percept)
        eye = np.eye(self.core.n_components, dtype=self.dtype)
        _ma_percept = self.multiagent_percept(percept)

        return np.concatenate((eye, _ma_percept), axis=1)
