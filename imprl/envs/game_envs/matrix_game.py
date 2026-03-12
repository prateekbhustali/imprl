import numpy as np


class MatrixGame:
    def __init__(self, env_config, baselines, time_perception=False, **env_kwargs):

        self.reward_to_cost = env_config["reward_to_cost"]
        self.time_perception = time_perception

        self.payoff = np.array(env_config["payoff_matrix"])
        self.ep_length = env_config["ep_length"]

        self.n_agents = self.payoff.ndim
        self.per_agent_actions = self.payoff.shape

        self.obs = np.zeros(self.n_agents, dtype=int)

        self.baselines = baselines

    def reset(self, **kwargs):
        self.time = 0

        if self.time_perception:
            obs = (np.array(self.time / self.ep_length), self.obs)
        else:
            obs = self.obs

        info = {"state": obs}

        return obs, info

    def step(self, action):
        self.time += 1

        reward = self.payoff[tuple(action)]

        if self.time_perception:
            next_obs = (np.array(self.time / self.ep_length), self.obs)
        else:
            next_obs = self.obs

        terminated = self.time >= self.ep_length

        info = {"state": next_obs}

        return next_obs, reward, terminated, False, info
