"""YAML config loading with command-line overrides (``--opts KEY.SUB value``)."""
import copy
import yaml


def load_config(path, opts=None):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if opts:
        cfg = merge_opts(cfg, opts)
    return cfg


def merge_opts(cfg, opts):
    """Apply overrides given as a flat list: ["SOLVER.EPOCHS", "10", "DATA.TEST_SIZE", "1600"]."""
    if len(opts) % 2 != 0:
        raise ValueError(f"--opts needs KEY VALUE pairs, got: {opts}")
    cfg = copy.deepcopy(cfg)
    for key, raw in zip(opts[0::2], opts[1::2]):
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                raise KeyError(f"Unknown config key: {key}")
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"Unknown config key: {key}")
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def dump_config(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
