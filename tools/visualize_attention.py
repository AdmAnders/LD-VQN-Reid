"""Cross-modal attention maps of D-TCR (Fig. 9).

Shows where the latent component nodes look in the image (node-to-image-token
attention of ``cross_image_attn``), conditioned on the given caption.

    python tools/visualize_attention.py --config configs/veri776_ldvqn.yml \
        --weights weights/ldvqn_veri776.pth \
        --image data/VeRi/image_query/0002_c002_00030600_0.jpg \
        --caption "A yellow hatchback viewed from the rear." \
        --output outputs/attention/0002_c002.png

``--layer`` picks the D-TCR block (default: last), ``--component`` a single
node (default: mean over all nodes).
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.build_dataloader import build_transforms  # noqa: E402
from models import LDVQN, load_weights, num_classes_from_checkpoint  # noqa: E402
from transformers import DebertaV2Tokenizer  # noqa: E402
from utils import load_config  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="Visualize D-TCR cross-modal attention")
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--caption", required=True)
    ap.add_argument("--output", default="outputs/attention/attention.png")
    ap.add_argument("--layer", type=int, default=-1, help="D-TCR block index (default: last)")
    ap.add_argument("--component", type=int, default=None, help="single node index (default: mean)")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--opts", nargs=argparse.REMAINDER, default=[])
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config, args.opts)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LDVQN(num_classes_from_checkpoint(args.weights), cfg["MODEL"])
    load_weights(model, args.weights)
    model = model.to(device).float().eval()

    _, test_tf = build_transforms(cfg["DATA"])
    tokenizer = DebertaV2Tokenizer.from_pretrained(cfg["MODEL"]["TEXT_BACKBONE"])
    pil = Image.open(args.image).convert("RGB")
    img = test_tf(pil).unsqueeze(0).to(device)
    tok = tokenizer(args.caption, padding="max_length", truncation=True,
                    max_length=cfg["DATA"]["MAX_TOKEN_LEN"], return_tensors="pt")

    _, attns = model(img, tok["input_ids"].to(device), tok["attention_mask"].to(device), return_attn=True)
    att = attns[args.layer]["img_attn"][0]            # (M, N_img), N_img = 1 (CLS) + H*W patches
    att = att[args.component] if args.component is not None else att.mean(0)
    att = att[1:]                                      # drop CLS token
    side = int(math.isqrt(att.numel()))
    att = att[: side * side].reshape(side, side).cpu().numpy()
    att = (att - att.min()) / (att.max() - att.min() + 1e-12)

    h, w = cfg["DATA"]["IMG_SIZE"]
    base = np.array(pil.resize((w, h)))[:, :, ::-1]   # RGB -> BGR for OpenCV
    heat = cv2.applyColorMap((cv2.resize(att, (w, h)) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base, 1 - args.alpha, heat, args.alpha, 0)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cv2.imwrite(args.output, np.concatenate([base, overlay], axis=1))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
