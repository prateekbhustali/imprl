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


def _setting_pattern(setting, k):
    # replace the k-value token (e.g. "-1-" or "_k1_") with a "{k}" placeholder
    for token in (f"_k{k}_", f"-{k}-"):
        if token in setting:
            return setting.replace(token, token.replace(str(k), "{k}"), 1)
    return setting


def describe_registry():
    """Return a Markdown table summarizing the environments/settings in ENV_REGISTRY."""
    env_root = Path(__file__).resolve().parent

    # group settings that only differ in `k` into a single row
    groups = {}
    for env_name, metadata in ENV_REGISTRY.items():
        for setting in metadata["settings"]:
            config_path = env_root / metadata["config_path"] / f"{setting}.yaml"
            with config_path.open() as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
            k = config.get("k")

            if k is None:
                # not a k-out-of-n style setting (e.g. matrix_game) -> no shared features to group on
                key = (env_name, None)
                groups.setdefault(key, {"settings": []})["settings"].append(setting)
                continue

            pattern = _setting_pattern(setting, k)
            horizon = "finite" if config.get("time_horizon") else "infinite"
            mobilisation = config.get("mobilisation_reward", 0) or "none"
            failure_penalty = config.get("failure_penalty_factor")

            key = (env_name, pattern, config.get("n_components"), horizon, mobilisation, failure_penalty)
            groups.setdefault(key, {"ks": []})["ks"].append(k)

    table_lines = [
        "| env_name | horizon | n (components) | k | mobilisation cost | failure penalty | setting pattern |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, group in groups.items():
        env_name = key[0]
        if key[1] is None:
            settings_text = ", ".join(f"`{s}`" for s in group["settings"])
            table_lines.append(f"| `{env_name}` | — | — | — | — | — | {settings_text} |")
        else:
            _, pattern, n, horizon, mobilisation, failure_penalty = key
            ks = sorted(group["ks"])
            k_text = f"{ks[0]}–{ks[-1]}" if len(ks) > 1 else str(ks[0])
            table_lines.append(
                f"| `{env_name}` | {horizon} | {n} | {k_text} | {mobilisation} | {failure_penalty} | `{pattern}` |"
            )

    return "\n".join(table_lines)


def print_registry():
    """Print the available environment names and display a Markdown summary table."""
    from IPython.display import Markdown, display

    print("Available environments:", list(ENV_REGISTRY))
    display(Markdown(describe_registry()))


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
