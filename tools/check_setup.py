"""Pre-flight check: environment, GPU, caption files and images for one config.

    python tools/check_setup.py --config configs/veri776_ldvqn.yml
    python tools/check_setup.py --config configs/vehicleid_ldvqn.yml --opts DATA.IMAGE_ROOT /content/data/VehicleID

Prints the number of caption lines / identities per split, the number of
missing images and a GPU-memory hint. Exit code 1 if something blocks training.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import build_records  # noqa: E402
from utils import collect_env, format_env, load_config  # noqa: E402

# Full caption-file sizes (lines) of the paper's splits; used to detect demo / incomplete files.
EXPECTED = {
    "veri776": {"train": 37778, "query": 1678, "gallery": 11579},
}

# Peak memory of the paper setting is high (CLIP ViT-L/14-336, batch 96, fp32).
PAPER_SETTING_MIN_GB = 75


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--opts", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.opts)
    d = cfg["DATA"]
    name = d["DATASET"]
    ok = True

    print("=" * 60 + "\nEnvironment\n" + "=" * 60)
    env = collect_env()
    print(format_env(env))
    if not env["cuda_available"]:
        print("\n[!] No CUDA GPU found. Training on CPU is not practical "
              "(Colab: Runtime > Change runtime type > GPU).")
        ok = False
    else:
        mem = env["gpu_memory_gb"]
        if mem < PAPER_SETTING_MIN_GB:
            print(f"\n[i] GPU has {mem} GB. The paper setting (batch 96, fp32) was run on a large GPU and may not fit.\n"
                  f"    Low-memory option (results can differ from the paper):\n"
                  f"      --opts SOLVER.AMP true DATALOADER.BATCH_SIZE 48 SOLVER.ACCUMULATION_STEPS 8\n"
                  f"    Evaluation with the released weights works on smaller GPUs "
                  f"(lower DATALOADER.TEST_BATCH_SIZE if needed).")

    print("\n" + "=" * 60 + f"\nData: {name}\n" + "=" * 60)
    print(f"IMAGE_ROOT : {d['IMAGE_ROOT']}  (exists: {os.path.isdir(d['IMAGE_ROOT'])})")
    print(f"CAPTION_DIR: {d['CAPTION_DIR']}  (exists: {os.path.isdir(d['CAPTION_DIR'])})")
    try:
        splits = dict(zip(["train", "query", "gallery"], build_records(cfg)))
    except (FileNotFoundError, KeyError) as e:
        print(f"[!] {e}")
        sys.exit(1)

    for split, recs in splits.items():
        missing = sum(not os.path.isfile(r["img_path"]) for r in recs)
        n_ids = len({r["pid"] for r in recs})
        fallback = sum(not (r.get("caption") or "").strip() or r["caption"].strip() == d["DEFAULT_CAPTION"]
                       for r in recs)
        line = (f"{split:>8}: {len(recs):>7} captions | {n_ids:>6} ids | {missing:>7} missing images | "
                f"{fallback:>6} fallback captions")
        exp = EXPECTED.get(name, {}).get(split)
        if exp and len(recs) != exp:
            line += f"   [!] expected {exp} (demo or incomplete caption file?)"
        print(line)
        if len(recs) == 0 or missing == len(recs):
            ok = False
        if missing:
            print(f"          e.g. {[r['img_path'] for r in recs if not os.path.isfile(r['img_path'])][:2]}")

    print("\n" + ("[OK] ready to run." if ok else "[!] fix the issues above before training."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
