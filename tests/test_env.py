import numpy as np


def test_init(kn_env):
    """check if the environment is initialised correctly"""

    assert kn_env.n_components == 5
    assert kn_env.n_damage_states == 4
    assert kn_env.n_comp_actions == 3
    assert kn_env.time_horizon == 50
    assert kn_env.discount_factor == 0.99
    assert kn_env.FAILURE_PENALTY_FACTOR == 3

    # check deterioration table shape
    assert kn_env.deterioration_table.shape == (5, 4, 4)

    # check if probabilities add up to 1
    assert np.isclose(np.sum(kn_env.deterioration_table, axis=2), 1, rtol=1e-3).all()

    # check if rewards table is correct
    assert (kn_env.rewards_table[0, :, 0] == 0.0).all()  # do-nothing
    assert (kn_env.rewards_table[0, :, 1] == -30).all()  # repair
    assert (kn_env.rewards_table[0, :, 2] == -20).all()  # inspect

    # check transition model shape
    assert kn_env.transition_model.shape == (5, 3, 4, 4)

    # check transition model probabilities add up to 1
    assert np.isclose(np.sum(kn_env.transition_model, axis=3), 1, rtol=1e-3).all()

    # check if observation model probabilities add up to 1
    assert np.isclose(np.sum(kn_env.observation_model, axis=3), 1, rtol=1e-3).all()


def test_reset(kn_env):

    # reset the environment
    state, info = kn_env.reset()

    # check if normalized time is 0.0
    assert state[0] == 0.0

    initial_belief = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )

    # check if belief for the initial state is correct
    assert (state[1] == initial_belief).all()


def test_belief_update(kn_env):

    belief = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )

    action = np.array([1, 0, 0, 0, 2])

    # set observation
    observation = np.array([1, 0, 0, 3, 0])

    # belief update
    new_belief = kn_env.belief_update(belief, action, observation)

    # manually calculated beliefs
    belief_0 = np.array([0.82, 0.13, 0.05, 0.0])
    belief_1 = np.array([0.72, 0.19, 0.09, 0.00])
    belief_3 = np.array([0.00, 0.00, 0.00, 1.00])
    belief_4 = np.array([0.9832, 0.0167, 0.000, 0.000])

    # check belief update
    assert np.isclose(new_belief[0], belief_0, rtol=1e-2).all()
    assert np.isclose(new_belief[1], belief_1, rtol=1e-2).all()
    assert np.isclose(new_belief[3], belief_3, rtol=1e-2).all()
    assert np.isclose(new_belief[4], belief_4, rtol=1e-2).all()


def test_reward(kn_env):
    """check if the reward is calculated correctly"""

    # reset the environment
    _ = kn_env.reset()

    damage_state = np.array([3, 0, 0, 0, 0])
    action = [2, 0, 0, 0, 2]

    belief = kn_env.belief
    next_belief = belief.copy()

    # get reward
    (
        reward,
        reward_replacement,
        reward_inspection,
        reward_penalty,
        reward_mobilisation,
    ) = kn_env.get_reward(damage_state, belief, action, next_belief)

    # check reward
    assert reward_penalty == -2400
    assert reward_mobilisation == 0
    assert reward_replacement == 0
    assert reward_inspection == -120
    assert reward == -2520


# -------------------- Tests for vectorised methods --------------------
def test_next_state_vec(kn_inf_env):
    """Seeded parity check: vec next-state sampler should match loop sampler."""
    state = np.array([0, 1, 2, 1], dtype=int)
    action = np.array([0, 1, 2, 0], dtype=int)

    np.random.seed(1234)
    next_state_loop = kn_inf_env.get_next_state(state, action)

    np.random.seed(1234)
    next_state_vec = kn_inf_env.get_next_state_vec(state, action)

    np.testing.assert_array_equal(next_state_vec, next_state_loop)


def test_observation_vec(kn_inf_env):
    """Seeded parity check: vec observation sampler should match loop sampler."""
    next_state = np.array([1, 2, 0, 1], dtype=int)
    action = np.array([0, 1, 2, 2], dtype=int)

    np.random.seed(4321)
    obs_loop, insp_loop = kn_inf_env.get_observation(next_state, action)

    np.random.seed(4321)
    obs_vec, insp_vec = kn_inf_env.get_observation_vec(next_state, action)

    np.testing.assert_array_equal(obs_vec, obs_loop)
    np.testing.assert_array_equal(insp_vec, insp_loop)


def test_belief_update_vec(kn_inf_env):
    """Deterministic parity check: vec belief update equals loop update."""
    belief = np.array(
        [
            [0.7, 0.3, 0.0],
            [0.2, 0.8, 0.0],
            [0.1, 0.5, 0.4],
            [0.0, 0.4, 0.6],
        ]
    )
    action = np.array([1, 0, 2, 1], dtype=int)
    observation = np.array([0, 1, 2, 1], dtype=int)

    updated_loop = kn_inf_env.belief_update(belief, action, observation)
    updated_vec = kn_inf_env.belief_update_vec(belief, action, observation)

    np.testing.assert_allclose(updated_vec, updated_loop, rtol=1e-12, atol=1e-12)


def test_reward_shaping_on_vec(kn_inf_env):
    """Deterministic parity check for vec reward with reward shaping enabled."""
    kn_inf_env.reward_shaping = True

    state = np.array([2, 0, 1, 2], dtype=int)
    belief = np.array(
        [
            [0.1, 0.3, 0.6],
            [0.9, 0.1, 0.0],
            [0.2, 0.5, 0.3],
            [0.0, 0.4, 0.6],
        ]
    )
    action = np.array([2, 0, 1, 2], dtype=int)
    next_belief = belief.copy()

    reward_loop = kn_inf_env.get_reward(state, belief, action, next_belief)
    reward_vec = kn_inf_env.get_reward_vec(state, belief, action, next_belief)

    np.testing.assert_allclose(np.array(reward_vec), np.array(reward_loop))


def test_reward_shaping_off_vec(kn_inf_env):
    """Deterministic parity check for vec reward with reward shaping disabled."""
    kn_inf_env.reward_shaping = False

    state = np.array([2, 2, 2, 2], dtype=int)
    belief = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    action = np.array([0, 1, 2, 0], dtype=int)
    next_belief = belief.copy()

    reward_loop = kn_inf_env.get_reward(state, belief, action, next_belief)
    reward_vec = kn_inf_env.get_reward_vec(state, belief, action, next_belief)

    np.testing.assert_allclose(np.array(reward_vec), np.array(reward_loop))
