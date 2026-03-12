import time
import numpy as np
import itertools

from imprl.runners.parallel import parallel_generic_rollout
import imprl.envs
from imprl.post_process import mean_with_ci


class InspectRepairHeuristicAgent:
    name = "Heuristic"
    full_name = "InspectRepairHeuristicAgent"

    def __init__(self, env):

        # Load optimal policy parameters from the environment's baselines
        baselines = env.core.baselines["InspectRepair"]["policy_params"]
        try:
            inspection_interval = baselines["inspection_interval"]
            num_inspection_components = baselines["num_inspection_components"]
            replacement_threshold = baselines["replacement_threshold"]
            self.policy = Policy(
                inspection_interval,
                num_inspection_components,
                replacement_threshold,
            )
            print(
                f"Policy parameters |  inspection_interval: {inspection_interval}, "
                f"components to inspect: {num_inspection_components}, "
                f"replacement threshold: {replacement_threshold}"
            )
        except KeyError:
            raise ValueError("InspectRepair baseline not found for this environment.")

    def select_action(self, time, insp_outcomes, beliefs):
        """
        Select action based on the current time, inspection outcomes, and beliefs.
        The action is determined by the policy parameters.
        """
        return self.policy(time, insp_outcomes, beliefs)


class Policy:
    def __init__(
        self, inspection_interval, num_inspection_components, replacement_threshold
    ):
        self.inspection_interval = inspection_interval
        self.num_inspection_components = num_inspection_components
        self.replacement_threshold = replacement_threshold

    def __call__(self, time, insp_outcomes, beliefs):
        """
        Generate actions based on the observation and policy parameters.
        The actions are determined by
        (i) inspection interval,
        (ii) number of components to inspect,
        (iii) replacement threshold.
        """
        actions = np.zeros(len(insp_outcomes), dtype=int)

        # Schedule inspections
        # At periodic intervals and prioritize based on failure probabilities
        if time > 0 and time % self.inspection_interval == 0:
            failure_probs = beliefs[:, -1]
            inspection_indices = (-failure_probs).argsort()[
                : self.num_inspection_components
            ]
            actions[inspection_indices] = 2

        # Schedule replacements
        # If inspection outcome >= replacement threshold
        replacement_indices = np.where(insp_outcomes >= self.replacement_threshold)[0]
        actions[replacement_indices] = 1

        return actions


class InspectionRepairHeuristic:
    def __init__(self, env) -> None:
        self.env = env
        self.policy_space = self.get_policy_space()

    def get_policy_space(self):
        """
        Define the policy space based on the policy parameters.
        The policy space is defined by
        (i) inspection intervals,
        (ii) number of components to inspect,
        (iii) replacement thresholds.
        """
        inspection_intervals = np.arange(1, self.env.core.time_limit)
        num_inspection_components = np.arange(1, self.env.core.n_components + 1)
        replacement_threshold = np.arange(1, self.env.core.n_damage_states)

        n1 = len(inspection_intervals)
        n2 = len(num_inspection_components)
        n3 = len(replacement_threshold)
        print(f"Number of inspection intervals: {n1}")
        print(f"Number of inspection components: {n2}")
        print(f"Number of replacement thresholds: {n3}")
        print(
            f"Total number of parameter combinations: {n1} x {n2} x {n3} = {n1 * n2 * n3}"
        )

        policy_space = []
        for policy_params in itertools.product(
            inspection_intervals, num_inspection_components, replacement_threshold
        ):
            policy_space.append(Policy(*policy_params))

        return policy_space

    @staticmethod
    def rollout(env, policy):
        beliefs, info = env.reset()
        insp_outcomes = info["inspection_outcomes"]
        time = 0
        terminated, truncated = False, False
        total_reward = 0

        while not terminated and not truncated:
            actions = policy(time, insp_outcomes, beliefs)
            beliefs, reward, terminated, truncated, info = env.step(actions)
            total_reward += reward * env.core.discount_factor**time
            time += 1
            insp_outcomes = info["inspection_outcomes"]

        return total_reward


if __name__ == "__main__":

    # set seed for reproducibility
    SEED = 42
    NUM_ROLLOUTS = 100_000
    np.random.seed(SEED)
    print(f"using seed {SEED}")
    print(f"Number of rollouts: {NUM_ROLLOUTS}")

    for k in range(1, 5):
        print(f"\nRunning heuristic search for k = {k}...")

        # create the environment
        ENV_NAME = "k_out_of_n_infinite"
        ENV_SETTING = f"hard-{k}-of-4_infinite"  # "hard-{k}-of-4_infinite", "n4_k{k}_nomob_fpf1.5", "n3_k{k}_nomob", "n2_k{k}_nomob"
        ENV_KWARGS = {
            "single_agent": False,
            "time_limit": 20,
        }
        env = imprl.envs.make(ENV_NAME, ENV_SETTING, **ENV_KWARGS)

        print(f"Environment: {ENV_NAME} with setting {ENV_SETTING}")

        simple_heuristic = InspectionRepairHeuristic(env)
        policy_space = simple_heuristic.policy_space
        best_policy = None
        best_policy_index = -1
        best_reward = float("-inf")

        print("Starting search for the best policy...")
        start_time = time.time()

        for p, policy in enumerate(policy_space):
            rewards = parallel_generic_rollout(
                env, policy, simple_heuristic.rollout, num_episodes=NUM_ROLLOUTS
            )
            mean_reward = np.mean(rewards)

            if mean_reward > best_reward:
                best_reward = mean_reward
                best_policy = policy
                best_policy_index = p

                print(
                    f"New best policy found at index {best_policy_index} with mean reward: {best_reward:.2f}"
                )

        print("\nSearch completed!")
        print(f"Best policy index: {best_policy_index}")
        best_evals = parallel_generic_rollout(
            env, best_policy, simple_heuristic.rollout, num_episodes=NUM_ROLLOUTS
        )
        mu, _l, _u = mean_with_ci(best_evals)
        print(
            f"Best total reward (SEM 95% CI): {mu:.2f} [{_l:.2f}, {_u:.2f}]"
        )
        print("Best policy parameters:")
        print(f"Inspection interval: {best_policy.inspection_interval}")
        print(
            f"Number of inspection components: {best_policy.num_inspection_components}"
        )
        print(f"Replacement threshold: {best_policy.replacement_threshold}")

        elapsed = time.time() - start_time
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Total time taken: {int(hours):02d}h:{int(minutes):02d}m:{seconds:.2f}s")
