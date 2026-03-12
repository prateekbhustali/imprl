import numpy as np
import pytest

import imprl.envs
from imprl.baselines.failure_replace import FailureReplace
from imprl.baselines.do_nothing import DoNothing
from imprl.runners.parallel import parallel_agent_rollout


@pytest.mark.parametrize(
    "env_name, env_setting",
    [
        # ("k_out_of_n_finite", "hard-5-of-5"), # ToDo
        ("k_out_of_n_infinite", "hard-1-of-4_infinite"),
    ],
)
def test_failure_replace(env_name, env_setting):
    env = imprl.envs.make(env_name, env_setting, single_agent=False, percept_type="obs")
    fr_agent = FailureReplace(env)

    # check if episode return is not None
    mean_returns = np.mean(
        parallel_agent_rollout(env, fr_agent, 1000)
    )
    assert np.isclose(
        mean_returns,
        env.core.baselines["FailureReplace"]["mean"],
        rtol=0.1,
    )


@pytest.mark.parametrize(
    "env_name, env_setting",
    [
        # ("k_out_of_n_finite", "hard-5-of-5"), # ToDo
        ("k_out_of_n_infinite", "hard-1-of-4_infinite"),
    ],
)
def test_do_nothing(env_name, env_setting):
    env = imprl.envs.make(
        env_name, env_setting, single_agent=False, percept_type="obs"
    )

    dn_agent = DoNothing(env)

    # check if episode return is not None
    mean_returns = np.mean(
        parallel_agent_rollout(env, dn_agent, 1000)
    )
    assert np.isclose(
        mean_returns, env.core.baselines["DoNothing"]["mean"], rtol=0.1
    )
