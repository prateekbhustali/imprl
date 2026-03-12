class RandomAgent:

    def __init__(self, env) -> None:
        self.env = env

        # Initialization
        self.episode = 0
        self.total_time = 0  # total time steps in lifetime
        self.time = 0  # time steps in current episode
        self.episode_return = 0  # return in current episode

        # Evaluation parameters
        # try to use the discount factor from the environment
        try:
            self.eval_discount_factor = env.core.discount_factor
        # if env doesn't specify, compute undiscounted return
        except AttributeError:
            self.eval_discount_factor = 1.0

    def select_action(self, percept, training=False):

        return [0, 1, 2, 0]

    def reset_episode(self, training=True):

        self.episode_return = 0
        self.episode += 1
        self.time = 0

    def process_rewards(self, reward):

        # discounting
        self.episode_return += reward * self.eval_discount_factor**self.time

        self.time += 1
        self.total_time += 1
