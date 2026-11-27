"""Feature extraction and mAP / CMC evaluation.

Protocol (unchanged from the paper): for every query, only the gallery entry
with the *same file name* as the query is removed from the ranking. Gallery
images of the same identity taken by the same camera are NOT filtered out.
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .reranking import re_ranking


@torch.no_grad()
def extract_features(model, loader, device, desc="Extracting"):
    model.eval()
    feats, pids, names = [], [], []
    for img, ids, msk, pid, name in tqdm(loader, desc=desc):
        feats.append(model(img.to(device), ids.to(device), msk.to(device)).cpu())
        pids.extend(pid.numpy())
        names.extend(name)
    return torch.cat(feats), pids, names


def evaluate(model, query_loader, gallery_loader, device, use_reranking=True,
             k1=20, k2=6, lambda_value=0.3):
    q_f, q_p, q_names = extract_features(model, query_loader, device, desc="Querying")
    g_f, g_p, g_names = extract_features(model, gallery_loader, device, desc="Gallery")

    q_f = F.normalize(q_f, p=2, dim=1)
    g_f = F.normalize(g_f, p=2, dim=1)
    distmat = 2 - 2 * torch.mm(q_f, g_f.t()).numpy()

    if use_reranking:
        print(">> Applying k-reciprocal re-ranking...")
        q_q_dist = 2 - 2 * torch.mm(q_f, q_f.t()).numpy()
        g_g_dist = 2 - 2 * torch.mm(g_f, g_f.t()).numpy()
        distmat = re_ranking(distmat, q_q_dist, g_g_dist, k1=k1, k2=k2, lambda_value=lambda_value)

    return compute_map_cmc(distmat, q_p, g_p, q_names, g_names)


def compute_map_cmc(distmat, q_p, g_p, q_names, g_names):
    num_q = len(q_p)
    mAP = 0.0
    cmc = np.zeros(len(g_p))
    g_p_arr = np.array(g_p)
    g_names_arr = np.array(g_names)

    for i in range(num_q):
        order = np.argsort(distmat[i])
        keep_mask = g_names_arr[order] != q_names[i]
        valid_order = order[keep_mask]
        matches = (g_p_arr[valid_order] == q_p[i]).astype(np.int32)

        if matches.sum() > 0:
            cmc_t = matches.cumsum()
            cmc_t[cmc_t > 1] = 1
            cmc[:len(cmc_t)] += cmc_t
            pos = np.where(matches == 1)[0]
            mAP += np.mean([(j + 1) / (pos[j] + 1) for j in range(len(pos))])

    return mAP / num_q, cmc / num_q


def rank_k(cmc, k):
    """CMC value at rank k (clipped to the gallery size)."""
    return float(cmc[min(k, len(cmc)) - 1])
