"""t-SNE of LD-VQN retrieval embeddings (Fig. 11).

Extracts the BN-neck features of all gallery images of ``--num_ids`` randomly
chosen identities (with their captions) and plots a 2-D t-SNE.

    python tools/tsne_visualization.py --config configs/veri776_ldvqn.yml \
        --weights weights/ldvqn_veri776.pth --num_ids 10 --output outputs/tsne_veri776.png

Pass ``--pids 2 5 13`` to pick the identities yourself.
"""
import argparse
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import TextImageReIDDataset, build_records, build_transforms  # noqa: E402
from models import LDVQN, load_weights, num_classes_from_checkpoint  # noqa: E402
from transformers import DebertaV2Tokenizer  # noqa: E402
from utils import extract_features, load_config  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="t-SNE of LD-VQN embeddings")
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", choices=["query", "gallery"], default="gallery")
    ap.add_argument("--num_ids", type=int, default=10)
    ap.add_argument("--pids", nargs="*", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--output", default="outputs/tsne.png")
    ap.add_argument("--opts", nargs=argparse.REMAINDER, default=[])
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.opts)
    d = cfg["DATA"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, query_rec, gallery_rec = build_records(cfg)
    records = gallery_rec if args.split == "gallery" else query_rec
    all_pids = sorted({r["pid"] for r in records})
    pids = args.pids or random.Random(args.seed).sample(all_pids, min(args.num_ids, len(all_pids)))
    records = [r for r in records if r["pid"] in set(pids)]
    print(f"{len(records)} images of {len(pids)} identities")

    _, test_tf = build_transforms(d)
    tokenizer = DebertaV2Tokenizer.from_pretrained(cfg["MODEL"]["TEXT_BACKBONE"])
    ds = TextImageReIDDataset(records, tokenizer, transform=test_tf, is_train=False, img_size=d["IMG_SIZE"],
                              max_token_len=d["MAX_TOKEN_LEN"], default_caption=d["DEFAULT_CAPTION"])
    loader = DataLoader(ds, batch_size=cfg["DATALOADER"]["TEST_BATCH_SIZE"], shuffle=False)

    model = LDVQN(num_classes_from_checkpoint(args.weights), cfg["MODEL"])
    load_weights(model, args.weights)
    model = model.to(device).float()

    feats, labels, _ = extract_features(model, loader, device)
    feats = torch.nn.functional.normalize(feats, dim=1).numpy()
    labels = np.array(labels)

    perplexity = min(args.perplexity, len(feats) - 1)
    emb = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=args.seed).fit_transform(feats)

    plt.figure(figsize=(8, 7), dpi=300)
    cmap = plt.get_cmap("tab20")
    for i, pid in enumerate(sorted(set(labels))):
        m = labels == pid
        plt.scatter(emb[m, 0], emb[m, 1], s=40, color=cmap(i % 20), edgecolors="k", linewidths=0.4,
                    label=str(pid))
    plt.legend(title="ID", fontsize=7, markerscale=0.8, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
