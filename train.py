"""LD-VQN training (Algorithm 1).

    python train.py --config configs/veri776_ldvqn.yml
    python train.py --config configs/vehicleid_ldvqn.yml --opts DATA.TEST_SIZE 800
    python train.py --config configs/veriwild_ldvqn.yml  --opts EXPERIMENT.OUTPUT_DIR outputs/vw_run2

Every epoch a resumable checkpoint is written to ``<OUTPUT_DIR>/ldvqn_checkpoint.pth``.
Every ``TEST.EVAL_PERIOD`` epochs the model is evaluated on query/gallery and,
if mAP improved, saved to ``<OUTPUT_DIR>/ldvqn_best.pth`` (as in the paper).
Re-running the same command resumes automatically from the last checkpoint.
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from datasets import make_dataloaders
from models import LDVQN
from models.losses import CrossBatchMemory, XBM_BatchHardTripletLoss, orthogonal_disentanglement_loss
from utils import dump_config, evaluate, format_env, load_config, rank_k, set_seed, write_env


def parse_args():
    ap = argparse.ArgumentParser(description="Train LD-VQN")
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None,
                    help="checkpoint to resume from (default: <OUTPUT_DIR>/ldvqn_checkpoint.pth if it exists)")
    ap.add_argument("--opts", nargs=argparse.REMAINDER, default=[],
                    help="config overrides: KEY.SUB value ...")
    return ap.parse_args()


def build_optimizer(model, cfg):
    pretrained_params, new_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "vision_encoder" in name or "deberta" in name:
            pretrained_params.append(param)
        else:
            new_params.append(param)
    s = cfg["SOLVER"]
    return torch.optim.AdamW([
        {"params": pretrained_params, "lr": s["LR_PRETRAINED"]},
        {"params": new_params, "lr": s["LR_NEW"]},
    ], weight_decay=s["WEIGHT_DECAY"])


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())   # checkpoint may have been loaded onto the GPU
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda"]])


def main():
    args = parse_args()
    cfg = load_config(args.config, args.opts)
    exp, s, L, T = cfg["EXPERIMENT"], cfg["SOLVER"], cfg["LOSS"], cfg["TEST"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(exp["OUTPUT_DIR"], exist_ok=True)
    dump_config(cfg, os.path.join(exp["OUTPUT_DIR"], "config.yml"))
    ckpt_path = os.path.join(exp["OUTPUT_DIR"], "ldvqn_checkpoint.pth")
    best_path = os.path.join(exp["OUTPUT_DIR"], "ldvqn_best.pth")
    set_seed(exp["SEED"], exp.get("DETERMINISTIC", False))
    print(format_env(write_env(os.path.join(exp["OUTPUT_DIR"], "env.txt"))))

    train_loader, query_loader, gallery_loader, num_classes = make_dataloaders(cfg, build_train=True)
    print(f"num_classes={num_classes}")

    model = LDVQN(num_classes, cfg["MODEL"]).to(device).float()
    xbm = CrossBatchMemory(memory_size=L["XBM_MEMORY_SIZE"], feature_dim=model.v_dim).to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=s["EPOCHS"])
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=L["LABEL_SMOOTHING"])
    criterion_tri = XBM_BatchHardTripletLoss(margin=L["TRIPLET_MARGIN"])

    acc = s["ACCUMULATION_STEPS"]
    max_iters = s.get("MAX_ITERS_PER_EPOCH", 0) or 0

    # Optional mixed precision (OFF for the paper results). bf16 needs no loss scaling.
    use_amp = bool(s.get("AMP", False)) and device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    if use_amp:
        print(f"[AMP] mixed precision enabled ({amp_dtype}) - results may differ slightly from the paper")
    if max_iters:
        print(f"[DEBUG] MAX_ITERS_PER_EPOCH={max_iters}: epochs are truncated (pipeline test only)")
    start_epoch, best_mAP = 0, 0.0
    resume = args.resume or (ckpt_path if os.path.exists(ckpt_path) else None)
    if resume:
        print(f"Loading checkpoint: {resume}")
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        if "scheduler_state_dict" in ck:
            scheduler.load_state_dict(ck["scheduler_state_dict"])
        if "xbm_state_dict" in ck:   # checkpoints written by the notebook do not have it
            xbm.load_state_dict(ck["xbm_state_dict"])
            xbm.load_extra_state(ck["xbm_extra"])
        if "scaler_state_dict" in ck and scaler.is_enabled():
            scaler.load_state_dict(ck["scaler_state_dict"])
        if "rng_state" in ck:
            set_rng_state(ck["rng_state"])
        start_epoch = ck["epoch"] + 1
        best_mAP = ck.get("best_mAP", 0.0)
        print(f"Resuming from epoch {start_epoch + 1} (best mAP so far: {best_mAP:.2%})")
    else:
        print("No checkpoint found, starting from scratch.")

    for epoch in range(start_epoch, s["EPOCHS"]):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (img, ids, msk, lbl, _) in enumerate(pbar):
            img, ids, msk, lbl = img.to(device), ids.to(device), msk.to(device), lbl.to(device)

            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                (logits, g_feat), attns = model(img, ids, msk, return_attn=True)
            if use_amp:   # losses in fp32
                logits, g_feat = logits.float(), g_feat.float()

            xbm.enqueue_dequeue(g_feat.detach(), lbl.detach())
            xbm_feats, xbm_lbls = xbm.get()

            loss_ce = criterion_ce(logits, lbl)
            if epoch > L["XBM_START_EPOCH"]:
                loss_tri = criterion_tri(g_feat, lbl, xbm_feats, xbm_lbls)
            else:
                loss_tri = criterion_tri(g_feat, lbl)

            loss = loss_ce + loss_tri
            if L["USE_ORTHO"]:
                loss += L["ORTHO_WEIGHT"] * orthogonal_disentanglement_loss(attns[-1]["nodes"])

            loss = loss / acc
            scaler.scale(loss).backward()

            last = (batch_idx + 1 == len(train_loader)) or (max_iters and batch_idx + 1 == max_iters)
            if ((batch_idx + 1) % acc == 0) or last:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), s["GRAD_CLIP"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            display_loss = loss.item() * acc
            total_loss += display_loss
            pbar.set_postfix({"Loss": f"{display_loss:.4f}", "LR": f"{scheduler.get_last_lr()[0]:.2e}"})
            if max_iters and batch_idx + 1 >= max_iters:
                break

        scheduler.step()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "xbm_state_dict": xbm.state_dict(),
            "xbm_extra": xbm.extra_state(),
            "scaler_state_dict": scaler.state_dict(),
            "rng_state": rng_state(),
            "best_mAP": best_mAP,
            "config": cfg,
        }
        torch.save(checkpoint, ckpt_path)
        print(f"[*] Checkpoint saved for epoch {epoch + 1} -> {ckpt_path}")

        if (epoch + 1) % T["EVAL_PERIOD"] == 0:
            mAP, cmc = evaluate(model, query_loader, gallery_loader, device, use_reranking=T["RERANKING"],
                                k1=T["RERANK_K1"], k2=T["RERANK_K2"], lambda_value=T["RERANK_LAMBDA"])
            print(f"Epoch {epoch + 1}: mAP: {mAP:.2%}, Rank-1: {rank_k(cmc, 1):.2%}, "
                  f"Rank-5: {rank_k(cmc, 5):.2%}, Rank-10: {rank_k(cmc, 10):.2%}")
            if mAP > best_mAP:
                best_mAP = mAP
                torch.save(model.state_dict(), best_path)
                print(f">> New best model saved -> {best_path}")
                checkpoint["best_mAP"] = best_mAP
                torch.save(checkpoint, ckpt_path)


if __name__ == "__main__":
    main()
