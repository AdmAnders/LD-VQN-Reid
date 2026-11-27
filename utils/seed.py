"""Seeding helpers.

Note: the run reported in the paper was not seeded, so results obtained with
this code are expected to match the paper within normal run-to-run variance,
not bit-for-bit.
"""
import os
import random

import numpy as np
import torch


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    """DataLoader ``worker_init_fn``: derive numpy/random seeds from torch's per-worker seed."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g
