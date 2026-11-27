"""LD-VQN evaluation (mAP / CMC, optional k-reciprocal re-ranking).

    python test.py --config configs/veri776_ldvqn.yml --weights weights/ldvqn_veri776.pth
    python test.py --config configs/vehicleid_ldvqn.yml --weights weights/ldvqn_vehicleid.pth --test_sizes 800 1600 2400
    python test.py --config configs/veriwild_ldvqn.yml --weights weights/ldvqn_veriwild.pth --test_sizes 3000 --no_rerank
"""
import argparse
import copy
import json
import os

import torch

from datasets import make_dataloaders
from models import LDVQN, load_weights, num_classes_from_checkpoint
from utils import collect_env, evaluate, format_env, load_config, rank_k, set_seed


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate LD-VQN")
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True, help="ldvqn_best.pth or a training checkpoint")
    ap.add_argument("--test_sizes", nargs="*", type=int, default=None,
                    help="VehicleID / VeRi-Wild: evaluate several test splits (overrides DATA.TEST_SIZE)")
    ap.add_argument("--no_rerank", action="store_true", help="disable k-reciprocal re-ranking")
    ap.add_argument("--output", default=None, help="optional JSON file for the results")
    ap.add_argument("--opts", nargs=argparse.REMAINDER, default=[])
    return ap.parse_args()


def main():
    args = parse_args()
    base_cfg = load_config(args.config, args.opts)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(base_cfg["EXPERIMENT"]["SEED"])
    T = base_cfg["TEST"]
    use_rerank = T["RERANKING"] and not args.no_rerank
    print(format_env(collect_env()))

    model = LDVQN(num_classes_from_checkpoint(args.weights), base_cfg["MODEL"])
    load_weights(model, args.weights, device="cpu")
    model = model.to(device).float().eval()

    sizes = args.test_sizes or [base_cfg["DATA"].get("TEST_SIZE")]
    results = {}
    for size in sizes:
        cfg = copy.deepcopy(base_cfg)
        if size is not None:
            cfg["DATA"]["TEST_SIZE"] = size
        _, query_loader, gallery_loader, _ = make_dataloaders(cfg, build_train=False)
        mAP, cmc = evaluate(model, query_loader, gallery_loader, device, use_reranking=use_rerank,
                            k1=T["RERANK_K1"], k2=T["RERANK_K2"], lambda_value=T["RERANK_LAMBDA"])
        tag = f"{cfg['DATA']['DATASET']}" + (f"-{size}" if size is not None else "")
        r1, r5, r10 = rank_k(cmc, 1), rank_k(cmc, 5), rank_k(cmc, 10)
        results[tag] = {"mAP": float(mAP), "rank1": r1, "rank5": r5, "rank10": r10, "reranking": use_rerank}
        print(f"[{tag}] mAP: {mAP:.2%}  Rank-1: {r1:.2%}  Rank-5: {r5:.2%}  "
              f"Rank-10: {r10:.2%}  (re-ranking={'on' if use_rerank else 'off'})")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
