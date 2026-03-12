import time
import numpy as np

import imprl.envs
from imprl.runners.parallel import parallel_agent_rollout
from imprl.post_process import mean_with_ci


class FailureReplace:

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

        if self.env.core.time_perception:
            percept = percept[-1][-1]
        else:
            percept = percept[:, -1]

        # replace if in last damage state
        action = [1 if x == 1 else 0 for x in percept]

        return action

    def reset_episode(self, training=True):

        self.episode_return = 0
        self.episode += 1
        self.time = 0

    def process_rewards(self, reward):

        # discounting
        self.episode_return += reward * self.eval_discount_factor**self.time

        self.time += 1
        self.total_time += 1


if __name__ == "__main__":

    # set seed for reproducibility
    SEED = 42
    NUM_ROLLOUTS = 10_000
    np.random.seed(SEED)
    print(f"using seed {SEED}")
    print(f"Number of rollouts: {NUM_ROLLOUTS}")

    for k in range(1, 5):
        print(f"\nRunning heuristic search for k = {k}...")

        # create the environment
        ENV_NAME = "k_out_of_n_infinite"
        ENV_SETTING = f"n4_k{k}_nomob_fpf1.5"
        ENV_KWARGS = {
            "percept_type": "obs",
            "single_agent": False,
            "time_limit": 20,
        }
        env = imprl.envs.make(ENV_NAME, ENV_SETTING, **ENV_KWARGS)

        print(f"Environment: {ENV_NAME} with setting {ENV_SETTING}")

        fr_heuristic = FailureReplace(env)

        start_time = time.time()

        rewards = parallel_agent_rollout(env, fr_heuristic, NUM_ROLLOUTS)
        mean_reward, _l, _u = mean_with_ci(rewards)
        print(
            f"Best total reward (SEM 95% CI): {mean_reward:.2f} [{_l:.2f}, {_u:.2f}]"
        )

        elapsed = time.time() - start_time
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Total time taken: {int(hours):02d}h:{int(minutes):02d}m:{seconds:.2f}s")
