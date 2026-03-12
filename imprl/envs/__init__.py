from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
import yaml

ENV_REGISTRY = {
    "k_out_of_n_finite": {
        "module": "imprl.envs.structural_envs.k_out_of_n_finite",
        "class_name": "KOutOfN",
        "single_wrapper_module": "imprl.envs.structural_envs.single_agent_wrapper",
        "multi_wrapper_module": "imprl.envs.structural_envs.multi_agent_wrapper",
        "config_path": "structural_envs/env_configs",
        "baselines_path": "structural_envs/baselines.yaml",
        "settings": [
            "hard-1-of-5",
            "hard-2-of-5",
            "hard-3-of-5",
            "hard-4-of-5",
            "hard-5-of-5",
        ],
    },
    "k_out_of_n_infinite": {
        "module": "imprl.envs.structural_envs.k_out_of_n_infinite",
        "class_name": "KOutOfN",
        "single_wrapper_module": "imprl.envs.structural_envs.single_agent_wrapper",
        "multi_wrapper_module": "imprl.envs.structural_envs.multi_agent_wrapper",
        "config_path": "structural_envs/env_configs",
        "baselines_path": "structural_envs/baselines.yaml",
        "settings": [
            "hard-1-of-4_infinite",
            "hard-2-of-4_infinite",
            "hard-3-of-4_infinite",
            "hard-4-of-4_infinite",
            "n2_k1_nomob",
            "n2_k2_nomob",
            "n3_k1_nomob",
            "n3_k2_nomob",
            "n3_k3_nomob",
            "n4_k1_nomob_fpf1.5",
            "n4_k2_nomob_fpf1.5",
            "n4_k3_nomob_fpf1.5",
            "n4_k4_nomob_fpf1.5",
        ],
    },
    "matrix_game": {
        "module": "imprl.envs.game_envs.matrix_game",
        "class_name": "MatrixGame",
        "single_wrapper_module": "imprl.envs.game_envs.single_agent_wrapper",
        "multi_wrapper_module": "imprl.envs.game_envs.multi_agent_wrapper",
        "config_path": "game_envs/env_configs",
        "baselines_path": "game_envs/baselines.yaml",
        "settings": ["climb_game", "penalty_game"],
    },
}


def make(name, setting, single_agent=False, **env_kwargs):
    if name not in ENV_REGISTRY:
        raise ValueError(
            f"Unknown environment '{name}'. Available environments: {list(ENV_REGISTRY)}"
        )

    env_entry = SimpleNamespace(**ENV_REGISTRY[name])
    allowed_settings = env_entry.settings
    if setting not in allowed_settings:
        raise ValueError(
            f"Invalid setting '{setting}' for environment '{name}'. "
            f"Available settings: {allowed_settings}"
        )

    env_module = import_module(env_entry.module)
    env_class = getattr(env_module, env_entry.class_name)

    # get the environment config
    env_root = Path(__file__).resolve().parent
    env_config_path = env_root / env_entry.config_path / f"{setting}.yaml"
    baselines_path = env_root / env_entry.baselines_path

    with env_config_path.open() as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)

    with baselines_path.open() as file:
        all_baselines = yaml.load(file, Loader=yaml.FullLoader)

    # get the baselines for this environment and setting
    baselines = all_baselines[name][setting]

    # create the environment
    env = env_class(env_config, baselines, **env_kwargs)

    # wrap the environment
    if single_agent:
        env = import_module(env_entry.single_wrapper_module).SingleAgentWrapper(env)
    else:
        env = import_module(env_entry.multi_wrapper_module).MultiAgentWrapper(env)

    return env
