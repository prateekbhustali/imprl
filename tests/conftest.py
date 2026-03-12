import pytest

import imprl.envs


@pytest.fixture
def kn_env():
    return imprl.envs.make(
        "k_out_of_n_finite",
        "hard-5-of-5",
        single_agent=False,
        percept_type="belief",
    ).core


@pytest.fixture
def kn_inf_env():
    return imprl.envs.make(
        "k_out_of_n_infinite",
        "hard-1-of-4_infinite",
        single_agent=False,
        percept_type="belief",
    ).core
