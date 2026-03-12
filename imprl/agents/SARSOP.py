"""
This script creates a SARSOP agent (for POMDPs) that can select actions
based on a policy defined by alpha vectors.

The state and actions spaces follow the ordering used in Julia's POMDPs.jl package,
which is not in lexicographic order. The agent can handle joint states and actions
for multiple components, making it suitable for multi-agent systems. Therefore, the agent expects the environment to be wrapped with a MultiAgentWrapper (not SingleAgentWrapper).

It also implements one-step lookahead search
(albeit a bit slow due to the use of a for loop)
"""

import xmltodict
import numpy as np
import itertools

from imprl.envs.structural_envs.multi_agent_wrapper import MultiAgentWrapper


class SARSOPAgent:
    name = "SARSOP"  # display names used by experiment scripts/loggers.
    full_name = (
        "Successive Approximations of the Reachable Space under Optimal Policies"
    )

    # Algorithm taxonomy.
    paradigm = "CTCE"
    formulation = "POMDP"
    algorithm_type = "planning"
    policy_regime = None  # not applicable since this is a planning algorithm, not a learning algorithm
    policy_type = "deterministic"

    # Training/runtime properties.
    uses_replay_memory = False
    parameter_sharing = False
    collect_state_info = True

    def __init__(self, env, policy_path, pomdp_path=None):
        """
        Initializes the SARSOP agent with the environment and policy.

        Inputs:
        -------
        env: The environment object from imprl
        policy_path: The file path to the policy XML
        pomdp_path: The file path to the POMDP XML (optional)
        """

        if not isinstance(env, MultiAgentWrapper):
            raise AssertionError(
                "SARSOPAgent expects a MultiAgentWrapper; pass imprl.envs.make(..., single_agent=False)."
            )

        self.env = env.core  # unwrap the core environment from the MultiAgentWrapper
        self.time_horizon = self.env.time_limit
        self.num_damage_states = self.env.n_damage_states
        self.num_components = self.env.n_components
        num_comp_actions = self.env.n_comp_actions

        # State space in Julia not in lexicographic order!
        # (cannot use itertools.product directly)
        # For example, for 4 components and 3 damage states,
        # [(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (0, 1, 0, 0), ...]
        self.joint_state_space = [
            tuple(list(reversed(s)))
            for s in itertools.product(
                range(self.num_damage_states), repeat=self.num_components
            )
        ]

        # Action space in Julia not in lexicographic order!
        # (cannot use itertools.product directly)
        # For example, for 4 components and 3 actions per component
        # [(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (0, 1, 0, 0), ...]
        joint_action_space = [
            tuple(list(reversed(s)))
            for s in itertools.product(
                range(num_comp_actions), repeat=self.num_components
            )
        ]
        self.joint_action_space = np.array(joint_action_space)

        self.num_joint_states = len(self.joint_state_space)
        self.num_joint_actions = len(self.joint_action_space)

        self.joint_state_space_array = np.array(self.joint_state_space)

        # POMDP
        if pomdp_path is not None:

            with open(pomdp_path) as f:
                pomdp = xmltodict.parse(f.read())

            pomdp_dict = pomdp["pomdpx"]

            self.discount = float(pomdp_dict["Discount"])
            self.transition_model = self._get_transition_model(pomdp_dict)
            self.observation_model = self._get_observation_model(pomdp_dict)
            self.reward_model = self._get_reward_model(pomdp_dict)
            self.initial_belief = self._get_initial_belief(pomdp_dict)

            assert np.sum(self.transition_model, axis=2).all() == 1
            assert np.sum(self.observation_model, axis=2).all() == 1

        # ALPHA VECTORS
        with open(policy_path) as f:
            policy = xmltodict.parse(f.read())

        vectorLength = policy["Policy"]["AlphaVector"]["@vectorLength"]
        numVectors = policy["Policy"]["AlphaVector"]["@numVectors"]

        self.alpha_vectors = np.zeros((int(numVectors), int(vectorLength)))

        self.vectorDict = policy["Policy"]["AlphaVector"]["Vector"]

        for i, vector in enumerate(self.vectorDict):
            self.alpha_vectors[i] = vector["#text"].split(" ")
        # Cache alpha-vector action ids to avoid dict parsing at runtime.
        self.alpha_vector_action_ids = np.array(
            [int(v["@action"]) for v in self.vectorDict], dtype=int
        )

        # Initialization
        self.episode = 0
        self.total_time = 0  # total time steps in lifetime
        self.time = 0  # time steps in current episode
        self.episode_return = 0  # return in current episode

    def reset_episode(self, training=False):

        self.episode_return = 0
        self.episode += 1
        self.time = 0

    def process_rewards(self, reward):

        _discount = self.env.discount_factor**self.time
        self.episode_return += reward * _discount
        self.time += 1
        self.total_time += 1

    def get_joint_belief(self, belief):

        if self.env.time_perception:
            _, belief = belief
        if belief.shape == (self.num_components, self.num_damage_states):
            belief = belief.T

        belief_values = belief[
            self.joint_state_space_array,
            np.arange(self.joint_state_space_array.shape[1]),
        ]

        # Compute the product along the second axis (components)
        joint_belief = np.prod(belief_values, axis=1)

        return joint_belief

    def select_action(self, belief, training=False, lookahead=True):
        # belief is expected shaped as (n_damage_states, n_components)
        if lookahead:
            return self.lookahead_alpha_vector_policy(belief)
        else:
            return self.alpha_vector_policy(belief)

    def alpha_vector_policy(self, belief):
        joint_belief = self.get_joint_belief(belief)

        act_id = int(np.argmax(self.alpha_vectors @ joint_belief))
        joint_act_id = self.alpha_vector_action_ids[act_id]

        return self.joint_action_space[joint_act_id]

    def lookahead_alpha_vector_policy(self, belief) -> np.ndarray:

        joint_belief = self.get_joint_belief(belief)

        q_b_a = np.empty(self.num_joint_actions, dtype=float)
        for joint_act_id in range(self.num_joint_actions):
            q_b_a[joint_act_id] = self._estimate_action_return(
                joint_belief, joint_act_id
            )

        joint_act_id = int(np.argmax(q_b_a))

        return self.joint_action_space[joint_act_id]

    def _estimate_action_return(self, joint_belief, joint_act_id) -> float:
        """
        Same result as `_estimate_action_return`, but vectorized over observations:
            Q(b, a) = R(b, a) + gamma * sum_o P(o|b,a) * V(b'_o)

        Scalar-to-vector mapping:
        - `R_b_a = dot(R[a], b)`
        - `pred = T[a].T @ b`                          # (S',)
        - `unnorm = O[a] * pred[:, None]`              # (S', O)
        - `p_o_b_a = unnorm.sum(axis=0)`               # (O,)
        - `b_prime[:, valid] = unnorm / p_o_b_a`       # (S', O), valid only
        - `v_b_prime = max(alpha_vectors @ b_prime)`   # per observation
        - `future = dot(p_o_b_a, v_b_prime)`
        """
        # R(b, a) = ∑_s R(s,a) * b(s)
        R_b_a = np.dot(self.reward_model[joint_act_id, :], joint_belief)

        # Predict next-state distribution: b̄(s') = ∑_s T(s,a,s') b(s)
        t_a = self.transition_model[joint_act_id, :, :]  # (S, S')
        pred = t_a.T @ joint_belief  # (S',)

        # Unnormalized posterior for every observation:
        # u(s', o) = O(a,s',o) * b̄(s')
        o_a = self.observation_model[joint_act_id, :, :]  # (S', O)
        # Broadcast pred over observations:
        #   o_a has shape (S', O) and pred has shape (S',).
        #   pred[:, None] reshapes pred to (S', 1), then NumPy broadcasts the 1 to O,
        #   effectively replicating pred[s'] across all columns.
        # Result: (unnormalized joint P(s', o | b, a))
        unnorm = o_a * pred[:, None]  # (S', O)

        # P(o|b,a) = ∑_{s'} u(s', o)
        p_o_b_a = np.sum(unnorm, axis=0)  # (O,)
        valid = p_o_b_a > 0
        if not np.any(valid):
            return float(R_b_a)

        # Normalize only where P(o|b,a) > 0
        b_prime = np.zeros_like(unnorm)
        b_prime[:, valid] = unnorm[:, valid] / p_o_b_a[valid]

        # V(b') = max_α α·b'  for each observation
        v_b_prime = np.max(self.alpha_vectors @ b_prime[:, valid], axis=0)

        # future = ∑_o P(o|b,a) * V(b')
        future = np.dot(p_o_b_a[valid], v_b_prime)

        # Q(b,a) = R(b,a) + γ * future
        return float(R_b_a + self.discount * future)

    # # scalar version (only for reference)
    # def _estimate_action_return(self, joint_belief, joint_act_id) -> float:

    #     # R(b, a) = ∑_s R(s,a) * b(s)
    #     R_b_a = np.dot(self.reward_model[joint_act_id, :], joint_belief)

    #     # Compute the expected value
    #     # future = ∑_o P(o|b,a) * V(b')
    #     future = 0
    #     for o in range(self.num_joint_states):

    #         # belief update
    #         _b = self.transition_model[joint_act_id, :, :].T @ joint_belief
    #         _b = _b * self.observation_model[joint_act_id, :, o]
    #         b_prime = _b / np.sum(_b)

    #         # compute Poba
    #         t_slice = self.transition_model[joint_act_id, :, :]
    #         o_slice = self.observation_model[joint_act_id, :, o]
    #         Posa = np.dot(t_slice, o_slice)
    #         Poba = np.dot(Posa, joint_belief)

    #         V_b_prime = np.max(self.alpha_vectors @ b_prime)

    #         future += Poba * V_b_prime

    #     # V(b) = R(b,a) + γ * ∑_o P(o|b,a) * V(b')
    #     return R_b_a + self.discount * future

    def _get_transition_model(self, pomdp_dict):
        nA = self.num_joint_actions
        nS = self.num_joint_states
        transition_model = np.zeros((nA, nS, nS))

        entries = pomdp_dict["StateTransitionFunction"]["CondProb"]["Parameter"][
            "Entry"
        ]

        for entry in entries:
            instance = entry["Instance"]
            indices = indices = tuple([int(part[1:]) for part in instance.split()])
            prob = float(entry["ProbTable"])

            transition_model[indices] = prob

        return transition_model

    def _get_observation_model(self, pomdp_dict):
        nA = self.num_joint_actions
        nS = self.num_joint_states
        nO = self.num_joint_states
        observation_model = np.zeros((nA, nS, nO))

        entries = pomdp_dict["ObsFunction"]["CondProb"]["Parameter"]["Entry"]

        for entry in entries:
            instance = entry["Instance"]
            indices = tuple([int(part[1:]) for part in instance.split()])
            prob = float(entry["ProbTable"])

            observation_model[indices] = prob

        return observation_model

    def _get_reward_model(self, pomdp_dict):
        nA = self.num_joint_actions
        nS = self.num_joint_states
        reward_model = np.zeros((nA, nS))

        entries = pomdp_dict["RewardFunction"]["Func"]["Parameter"]["Entry"]

        for entry in entries:
            instance = entry["Instance"]
            indices = tuple([int(part[1:]) for part in instance.split()])
            reward = float(entry["ValueTable"])

            reward_model[indices] = reward

        return reward_model

    def _get_initial_belief(self, pomdp_dict):
        nS = self.num_joint_states
        initial_belief = np.zeros(nS)

        entries = pomdp_dict["InitialStateBelief"]["CondProb"]["Parameter"]["Entry"]

        for entry in entries:
            instance = entry["Instance"]
            indices = tuple([int(part[1:]) for part in instance.split()])
            prob = float(entry["ProbTable"])

            initial_belief[indices] = prob

        return initial_belief
