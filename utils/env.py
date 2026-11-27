"""Collect software / hardware information so every run records the environment it used."""
import platform
import sys
from importlib import metadata

PACKAGES = ["torch", "torchvision", "transformers", "tokenizers", "sentencepiece", "huggingface_hub",
            "numpy", "scipy", "pillow", "scikit-learn", "PyYAML"]


def collect_env():
    import torch
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for p in PACKAGES:
        try:
            info[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            info[p] = "not installed"
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / 1024 ** 3, 1)
        info["cuda"] = torch.version.cuda
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    return info


def format_env(info):
    return "\n".join(f"{k:>16}: {v}" for k, v in info.items())


def write_env(path):
    info = collect_env()
    with open(path, "w", encoding="utf-8") as f:
        f.write(format_env(info) + "\n")
    return info
