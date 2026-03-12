import numpy as np


from gymnasium import spaces
from gymnasium.spaces import Discrete, Tuple, Box


class MultiAgentWrapper:

    def __init__(self, env) -> None:
        self.core = env
        self.dtype = np.float32  # or np.float64
        self.n_agents = self.core.n_agents

        # Percept space
        self.shared_observation_space = Box(0, 1, shape=(1,), dtype=self.dtype)
        self.local_observation_space = Box(
            0, 1, shape=(self.core.n_agents, 1), dtype=self.dtype
        )
        if env.time_perception:
            self.perception_space = Tuple(
                (self.shared_observation_space, self.local_observation_space),
            )
        else:
            self.perception_space = self.local_observation_space

        self.action_space = Discrete(self.core.per_agent_actions[0])

    def step(self, actions):
        next_percept, reward, terminated, truncated, info = self.core.step(actions)
        next_percept = next_percept.reshape(-1, 1)
        return next_percept, self.dtype(reward), terminated, truncated, info

    def reset(self, **kwargs):
        next_percept, info = self.core.reset(**kwargs)
        next_percept = next_percept.reshape(-1, 1)
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
            return local_percept

    def multiagent_percept(self, percept):

        if self.core.time_perception:
            shared_percept = spaces.utils.flatten(
                self.shared_observation_space, percept[0]
            )
            shared_percept = np.broadcast_to(shared_percept, (1, self.core.n_agents))

            local_percept = spaces.utils.flatten(
                self.local_observation_space, percept[1].T
            )
            local_percept = local_percept.reshape(-1, self.core.n_agents, order="F")

            return np.concatenate((shared_percept, local_percept), axis=0).T

        else:
            local_percept = spaces.utils.flatten(
                self.local_observation_space, percept.T
            )
            local_percept = local_percept.reshape(-1, self.core.n_agents, order="F")

            return local_percept.T

    def multiagent_idx_percept(self, percept):

        # (id, shared_percept, local_percept)
        eye = np.eye(self.core.n_agents, dtype=self.dtype)
        _ma_percept = self.multiagent_percept(percept)

        return np.concatenate((eye, _ma_percept), axis=1)
