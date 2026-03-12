"""
k-out-of-n structural maintenance environment.

Environment used in the paper "The price of decentralization in managing 
engineered systems through multi-agent reinforcement learning"

This module implements a partially observable maintenance environment for
multi-component k-out-of-n system, where the system is functional if at least `k` out
of `n` components are still working. Each component evolves through discrete
damage states over time according to a component-specific transition model.

At every step, choose one action per component:
1. `0`: do nothing
2. `1`: replace
3. `2`: inspect

The environment then:
1. Samples next damage states from the transition model.
2. Samples observations from the observation model (inspection-dependent).
3. Updates per-component belief states with a Bayes filter.
4. Computes reward/cost from replacement, inspection, mobilisation, and
    system-failure risk (or state-based failure penalty when shaping is off).

Remark:
Vectorized variants (`*_vec`) for scaling to larger problems are provided at the end
for experimentation. In our benchmarks (e.g., 4-component setups), end-to-end
speedups were small. So the default `step()` path stays loop-based for readability.
"""

import numpy as np


class KOutOfN:

    def __init__(
        self,
        env_config: dict,
        baselines: dict = None,
        percept_type: str = "belief",
        time_limit: int = 20,
        time_perception: bool = False,
        reward_shaping: bool = True,
        return_discounted_rewards: bool = False,
        is_failure_self_announcing: bool = False,
    ) -> None:

        self.reward_to_cost = True
        self.time_limit = time_limit
        self.reward_shaping = reward_shaping
        self.time_perception = time_perception
        # for on-policy RL methods, we need to return discounted rewards
        self.return_discounted_rewards = return_discounted_rewards
        self.is_failure_self_announcing = is_failure_self_announcing

        self.k = env_config["k"]
        self.time_horizon = env_config["time_horizon"]
        self.discount_factor = env_config["discount_factor"]
        self.FAILURE_PENALTY_FACTOR = env_config["failure_penalty_factor"]
        self.n_components = env_config["n_components"]
        self.n_damage_states = env_config["n_damage_states"]
        assert self.n_damage_states > 1, "n_damage_states must be greater than 1."
        self.n_comp_actions = env_config["n_comp_actions"]
        self.action_map = {0: "do_nothing", 1: "replace", 2: "inspect"}
        try:
            self.initial_belief = env_config["initial_belief"]
        except KeyError:
            print("Initial belief not specified")

        ####################### Transition Model #######################

        # shape: (n_components, n_damage_states, n_damage_states)
        self.deterioration_table = np.array(env_config["transition_model"])

        self.replacement_table = np.zeros(
            (self.n_components, self.n_damage_states, self.n_damage_states)
        )

        for c in range(self.n_components):
            # Example for 3 states (r = replacement accuracy):
            # [[1,   0,   0  ],
            #  [r, 1-r,   0  ],
            #  [r,   0, 1-r ]]
            r = env_config["replacement_accuracies"][c]
            replacement = np.zeros((self.n_damage_states, self.n_damage_states))
            replacement[0, 0] = 1.0
            for s in range(1, self.n_damage_states):
                replacement[s, 0] = r  # replace successfully to state 0
                replacement[s, s] = 1 - r  # fail to replace, stay in same state
            self.replacement_table[c] = replacement

        self.transition_model = np.zeros(
            (
                self.n_components,
                self.n_comp_actions,
                self.n_damage_states,
                self.n_damage_states,
            )
        )

        for c in range(self.n_components):

            # do nothing: deterioration
            self.transition_model[c, 0, :, :] = self.deterioration_table[c, :, :]

            # replacement: replace instantly + deterioration
            # D^T @ R^T @ belief ==> (R @ D)^T @ belief
            self.transition_model[c, 1, :, :] = (
                self.replacement_table[c] @ self.deterioration_table[c, :, :]
            )

            # inspect: deterioration
            self.transition_model[c, 2, :, :] = self.deterioration_table[c, :, :]

        assert np.allclose(
            self.transition_model.sum(axis=-1), 1.0, atol=1e-8
        ), "Transition probabilities must sum to 1 along the last axis."

        ######################### Reward Model #########################

        self.rewards_table = np.zeros(
            (self.n_components, self.n_damage_states, self.n_comp_actions)
        )

        self.rewards_table[:, :, 1] = np.array(
            [env_config["replacement_rewards"]] * self.n_damage_states
        ).T
        self.rewards_table[:, :, 2] = np.array(
            [env_config["inspection_rewards"]] * self.n_damage_states
        ).T

        self.system_replacement_reward = sum(env_config["replacement_rewards"])
        try:
            self.mobilisation_reward = env_config["mobilisation_reward"]
        except KeyError:
            print("Mobilisation reward not specified.")

        ####################### Observation Model ######################

        inspection_model = np.zeros(
            (self.n_components, self.n_damage_states, self.n_damage_states)
        )
        no_inspection_model = np.zeros(
            (self.n_components, self.n_damage_states, self.n_damage_states)
        )
        for c in range(self.n_components):

            p = env_config["obs_accuracies"][c]
            try:
                f_p = env_config["failure_obs_accuracies"][c]
            except KeyError:
                print("Failure observation accuracy not specified.")

            # ----- Part A: inspection model (inspection action is chosen)
            # Example for 3 states:
            # [[p,      1-p,      0 ],
            #  [(1-p)/2, p,   (1-p)/2],
            #  [?,       ?,      ? ]]
            # Build common rows for non-failed states first.
            model = np.zeros((self.n_damage_states, self.n_damage_states))
            model[0, 0] = p
            model[0, 1] = 1 - p
            for s in range(1, self.n_damage_states - 1):
                model[s, s] = p  # prob. of observing true damage state s
                model[s, s - 1] = (1 - p) / 2
                model[s, s + 1] = (1 - p) / 2
            if self.is_failure_self_announcing:
                # ----- inspection model path: failure is self-announcing
                # Example for 3 states:
                # [[p,      1-p,      0 ],
                #  [(1-p)/2, p,   (1-p)/2],
                #  [0,       0,      1 ]]
                model[-1, -1] = 1.0
            else:
                # ----- inspection model path: failure is NOT self-announcing
                # Example for 3 states:
                # [[p,      1-p,      0 ],
                #  [(1-p)/2, p,   (1-p)/2],
                #  [0,      1-f_p,   f_p]]
                model[-1, -2] = 1 - f_p
                model[-1, -1] = f_p
            inspection_model[c] = model

            # ----- Part B: no-inspection model (action other than inspection is chosen)
            if self.is_failure_self_announcing:
                # ----- no-inspection model path: failure is self-announcing
                # even if you don't inspect, you can still observe the failure state with probability 1.
                # Example for 3 states:
                # [[1/2, 1/2, 0],
                #  [1/2, 1/2, 0],
                #  [0,   0,   1]]
                model = np.zeros((self.n_damage_states, self.n_damage_states))
                model[:, :-1] = 1 / (self.n_damage_states - 1)
                model[-1, :-1] = 0.0
                model[-1, -1] = 1.0
                no_inspection_model[c] = model
            else:
                # ----- no-inspection model path: failure is NOT self-announcing
                # No-inspection model when failure is NOT self-announcing.
                # Example for 3 states:
                # [[1/3, 1/3, 1/3],
                #  [1/3, 1/3, 1/3],
                #  [1/3, 1/3, 1/3]]
                no_inspection_model[c] = np.full(
                    (self.n_damage_states, self.n_damage_states),
                    1 / self.n_damage_states,
                )

        self.observation_model = np.zeros(
            (
                self.n_components,
                self.n_comp_actions,
                self.n_damage_states,
                self.n_damage_states,
            )
        )

        for c in range(self.n_components):

            # do nothing
            self.observation_model[c, 0, :, :] = no_inspection_model[c]

            # replacement
            self.observation_model[c, 1, :, :] = no_inspection_model[c]

            # inspection
            self.observation_model[c, 2, :, :] = inspection_model[c]

        assert np.allclose(
            self.observation_model.sum(axis=-1), 1.0, atol=1e-8
        ), "Observation probabilities must sum to 1 along the last axis."

        self.percept_type = percept_type
        self.baselines = baselines

        self.state = self.reset()

    @staticmethod
    def pf_sys(pf, k):
        """Computes the system failure probability pf_sys for k-out-of-n components

        Args:
            pf: Numpy array with components' failure probability.
            k: Integer indicating k (out of n) components.

        Returns:
            PF_sys: Numpy array with the system failure probability.
        """
        n = pf.size
        nk = n - k
        m = k + 1
        A = np.zeros(m + 1)
        A[1] = 1
        L = 1
        for j in range(1, n + 1):
            h = j + 1
            Rel = 1 - pf[j - 1]
            if nk < j:
                L = h - nk
            if k < j:
                A[m] = A[m] + A[k] * Rel
                h = k
            for i in range(h, L - 1, -1):
                A[i] = A[i] + (A[i - 1] - A[i]) * Rel
        PF_sys = 1 - A[m]
        return PF_sys

    def _is_system_functional(self, state):

        # check number of failed components
        _is_failed = state // (self.n_damage_states - 1)  # 0 if working, 1 if failed
        n_working = self.n_components - np.sum(_is_failed)
        functional = n_working >= self.k

        return functional

    def get_reward(
        self, state: list, belief: np.array, action: list, next_belief: np.array
    ) -> tuple[float, float, float, float]:

        reward_replacement = 0
        reward_inspection = 0
        reward_system = 0

        failure_cost = self.system_replacement_reward * self.FAILURE_PENALTY_FACTOR

        for c in range(self.n_components):

            if action[c] == 1:
                reward_replacement += self.rewards_table[c, state[c], action[c]]

            elif action[c] == 2:
                reward_inspection += self.rewards_table[c, state[c], action[c]]

        # mobilisation cost
        mobilised = sum(action) > 0
        reward_mobilisation = mobilised * self.mobilisation_reward

        if self.reward_shaping:
            pf = belief[:, -1]
            pf_sys = self.pf_sys(pf, self.k)
            reward_system = failure_cost * pf_sys
        else:  # state-based reward
            if not self._is_system_functional(state):
                reward_system = failure_cost

        reward = (
            reward_replacement + reward_inspection + reward_system + reward_mobilisation
        )

        return (
            reward,
            reward_replacement,
            reward_inspection,
            reward_system,
            reward_mobilisation,
        )

    def get_next_state(self, state: np.array, action: list) -> np.array:

        _next_states = np.zeros(self.n_components, dtype=int)

        for c in range(self.n_components):

            next_damage_state = np.random.choice(
                np.arange(self.n_damage_states),
                p=self.transition_model[c, action[c], state[c], :],
            )

            _next_states[c] = next_damage_state

        return _next_states

    def get_observation(self, nextstate: list, action: list) -> np.array:

        _observations = np.zeros(self.n_components, dtype=int)
        _inspection_outcomes = np.ones(self.n_components, dtype=int)

        for c in range(self.n_components):

            obs = np.random.choice(
                np.arange(self.n_damage_states),
                p=self.observation_model[c, action[c], nextstate[c], :],
            )

            _observations[c] = obs
            _inspection_outcomes[c] = obs if action[c] == 2 else -1

        return _observations, _inspection_outcomes

    def belief_update(
        self, belief: np.array, action: list, observation: list
    ) -> np.array:

        next_belief = np.empty((self.n_components, self.n_damage_states))

        for c in range(self.n_components):

            belief_c = belief[c, :]

            # transition model
            belief_c = self.transition_model[c, action[c]].T @ belief_c

            # observation model
            state_probs = self.observation_model[c, action[c], :, observation[c]]
            belief_c = state_probs * belief_c

            # normalise
            belief_c = belief_c / np.sum(belief_c)

            next_belief[c, :] = belief_c

        return next_belief

    def step(self, action: list) -> tuple[np.array, float, bool, dict]:

        # compute next damage state
        next_state = self.get_next_state(self.damage_state, action)

        # compute observation
        self.observation, inspection_outcomes = self.get_observation(next_state, action)

        # update belief only if percept_type is belief to avoid unnecessary computation
        next_belief = self.belief_update(self.belief, action, self.observation)

        # collect reward: R(s,a)
        (
            reward,
            reward_replacement,
            reward_inspection,
            reward_system,  # could be risk or failure cost
            reward_mobilisation,
        ) = self.get_reward(self.damage_state, self.belief, action, next_belief)

        if self.return_discounted_rewards:
            reward *= self.discount_factor**self.time

        # check if system is functional
        has_system_failed = not self._is_system_functional(self.damage_state)

        self.damage_state = next_state
        self.belief = next_belief

        # update time
        self.time += 1
        self.norm_time = self.time / self.time_limit

        terminated = False

        truncated = True if self.time >= self.time_limit else False

        # update info dict
        info = {
            "system_failure": has_system_failed,
            "reward_replacement": reward_replacement,
            "reward_inspection": reward_inspection,
            "reward_system": reward_system,  # could be risk or failure cost
            "reward_mobilisation": reward_mobilisation,
            "state": self._get_state(),
            "observation": self.observation,
            "inspection_outcomes": inspection_outcomes,
        }

        return self._get_percept(), reward, terminated, truncated, info

    def reset(self) -> tuple[np.array, np.array]:

        # duplicate the initial belief for each component
        self.belief = np.tile(self.initial_belief, (self.n_components, 1))

        self.damage_state = np.random.choice(
            self.n_damage_states, p=self.initial_belief, size=self.n_components
        )
        self.observation = np.random.choice(
            self.n_damage_states, p=self.initial_belief, size=self.n_components
        )

        # reset the time
        self.time = 0
        self.norm_time = self.time / self.time_limit

        info = {
            "system_failure": False,
            "reward_replacement": 0,
            "reward_inspection": 0,
            "reward_system": 0,
            "reward_mobilisation": 0,
            "state": self._get_state(),
            "observation": self.observation,
            "inspection_outcomes": np.ones(self.n_components, dtype=int) * -1,
        }

        return self._get_percept(), info

    def _get_percept(self) -> tuple[np.array, np.array]:

        if self.percept_type in ["belief", "Belief"]:
            return self._get_belief()
        elif self.percept_type in ["state", "State"]:
            return self._get_state()
        elif self.percept_type in ["obs", "Obs"]:
            return self._get_observation()

    def _get_state(self) -> tuple[np.array, np.array]:
        one_hot = np.zeros((self.n_components, self.n_damage_states))
        one_hot[np.arange(self.n_components), self.damage_state] = 1
        if self.time_perception:
            return (np.array([self.norm_time]), one_hot)
        else:
            return one_hot

    def _get_observation(self) -> tuple[np.array, np.array]:
        one_hot = np.zeros((self.n_components, self.n_damage_states))
        one_hot[np.arange(self.n_components), self.observation] = 1
        if self.time_perception:
            return (np.array([self.norm_time]), one_hot)
        else:
            return one_hot

    def _get_belief(self) -> tuple[np.array, np.array]:
        if self.time_perception:
            return (np.array([self.norm_time]), self.belief)
        else:
            return self.belief

    # ----- Optional vectorized variants kept for benchmarking/readability trade-off.
    # These are intentionally not used in step().
    @staticmethod
    def _sample_categorical_vec(probs: np.array) -> np.array:
        # probs: (n_components, n_categories)
        cdf = np.cumsum(probs, axis=1)
        draws = np.random.random((probs.shape[0], 1))
        return (draws > cdf).sum(axis=1).astype(int)

    def get_next_state_vec(self, state: np.array, action: list) -> np.array:
        # state/action: (n_components,)
        # probs/cdf: (n_components, n_damage_states)
        probs = self.transition_model[np.arange(self.n_components), action, state]
        return self._sample_categorical_vec(probs)  # next_state: (n_components,)

    def get_observation_vec(
        self, nextstate: list, action: list
    ) -> tuple[np.array, np.array]:
        # nextstate/action: (n_components,)
        # probs/cdf: (n_components, n_damage_states)
        probs = self.observation_model[np.arange(self.n_components), action, nextstate]
        # observations/inspection_outcomes: (n_components,)
        observations = self._sample_categorical_vec(probs)
        inspection_outcomes = np.where(action == 2, observations, -1)
        return observations, inspection_outcomes

    def belief_update_vec(
        self, belief: np.array, action: list, observation: list
    ) -> np.array:
        # belief: (n_components, n_damage_states)
        # trans/weighted: (n_components, n_damage_states, n_damage_states)
        trans = self.transition_model[np.arange(self.n_components), action]
        weighted = trans * belief[:, :, None]
        # predicted: (n_components, n_damage_states)
        predicted = weighted.sum(axis=1)

        # obs_likelihood/posterior: (n_components, n_damage_states)
        obs_likelihood = self.observation_model[
            np.arange(self.n_components), action, :, observation
        ]
        posterior = obs_likelihood * predicted
        posterior /= posterior.sum(axis=1, keepdims=True)
        return posterior

    def get_reward_vec(
        self, state: list, belief: np.array, action: list, next_belief: np.array
    ) -> tuple[float, float, float, float]:

        # comp/state/action/masks: (n_components,)
        comp = np.arange(self.n_components)

        replacement_mask = action == 1
        inspection_mask = action == 2

        reward_replacement = (
            self.rewards_table[comp[replacement_mask], state[replacement_mask], 1].sum()
            if replacement_mask.any()
            else 0.0
        )
        reward_inspection = (
            self.rewards_table[comp[inspection_mask], state[inspection_mask], 2].sum()
            if inspection_mask.any()
            else 0.0
        )

        failure_cost = self.system_replacement_reward * self.FAILURE_PENALTY_FACTOR
        if self.reward_shaping:
            reward_system = failure_cost * self.pf_sys(belief[:, -1], self.k)
        else:
            reward_system = (
                failure_cost if not self._is_system_functional(state) else 0.0
            )

        reward_mobilisation = self.mobilisation_reward if np.any(action > 0) else 0.0
        reward = (
            reward_replacement + reward_inspection + reward_system + reward_mobilisation
        )

        return (
            reward,
            reward_replacement,
            reward_inspection,
            reward_system,
            reward_mobilisation,
        )
