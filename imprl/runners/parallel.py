import itertools
import numpy as np
import multiprocessing as mp

from imprl.runners.serial import evaluate_heuristic, evaluate_agent

############################## MPI #####################################


def mpi_generic_rollout(
    env, heuristic, rollout_method, num_episodes, rank=0, num_procs=1
):

    result = []
    for _ in range(rank, num_episodes, num_procs):
        episode_cost = rollout_method(env, heuristic)
        result.append(episode_cost)

    return result


def mpi_heuristic_rollouts(env, heuristic, num_episodes, rank=0, num_procs=1):

    return mpi_generic_rollout(
        env, heuristic, evaluate_heuristic, num_episodes, rank, num_procs
    )


def mpi_agent_rollouts(env, agent, num_episodes, rank=0, num_procs=1):

    return mpi_generic_rollout(
        env, agent, evaluate_agent, num_episodes, rank, num_procs
    )


########################## Multiprocessing #############################


def parallel_generic_rollout(
    env,
    heuristic,
    rollout_method,
    num_episodes,
    verbose=False,
    num_workers=None,
):

    # cpu count = number of cores * logical cores
    if num_workers is None:
        cpu_count = mp.cpu_count()
    else:
        cpu_count = max(1, int(num_workers))

    if verbose:
        print(f"CPU count: {cpu_count}")

    # create an iterable for the starmap
    iterable = zip(
        itertools.repeat(env, num_episodes), itertools.repeat(heuristic, num_episodes)
    )

    with mp.Pool(cpu_count) as pool:
        list_func_evaluations = pool.starmap(rollout_method, iterable)

    results = np.hstack(list_func_evaluations)

    return results


def parallel_heursitic_rollout(
    env, heuristic, num_episodes, verbose=False, num_workers=None
):

    return parallel_generic_rollout(
        env, heuristic, evaluate_heuristic, num_episodes, verbose, num_workers
    )


def parallel_agent_rollout(
    env, agent, num_episodes, verbose=False, num_workers=None
):

    return parallel_generic_rollout(
        env, agent, evaluate_agent, num_episodes, verbose, num_workers
    )
