"""Cross-Batch Memory (XBM) and batch-hard triplet loss.

Kept identical to the paper's training code. Note that the current batch is
enqueued *before* the loss is computed, so the memory also contains the
(detached) features of the current batch.
"""
import torch
import torch.nn as nn


class CrossBatchMemory(nn.Module):
    def __init__(self, memory_size, feature_dim):
        super().__init__()
        self.memory_size = memory_size
        self.register_buffer("feats", torch.zeros(memory_size, feature_dim))
        self.register_buffer("targets", torch.zeros(memory_size, dtype=torch.long))
        self.ptr = 0
        self.is_full = False

    def get(self):
        if self.is_full:
            return self.feats, self.targets
        return self.feats[:self.ptr], self.targets[:self.ptr]

    def enqueue_dequeue(self, feats, targets):
        q_size = feats.size(0)
        if self.ptr + q_size > self.memory_size:
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.ptr = 0
            self.is_full = True
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size

    # pointer state is not a buffer, so it is saved explicitly for exact resuming
    def extra_state(self):
        return {"ptr": self.ptr, "is_full": self.is_full}

    def load_extra_state(self, state):
        self.ptr = state["ptr"]
        self.is_full = state["is_full"]


class XBM_BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs_q, targets_q, inputs_g=None, targets_g=None):
        if inputs_g is None:
            inputs_g, targets_g = inputs_q, targets_q

        n_q = inputs_q.size(0)
        n_g = inputs_g.size(0)

        # Euclidean distance matrix (n_q x n_g)
        dist = torch.pow(inputs_q, 2).sum(dim=1, keepdim=True).expand(n_q, n_g) + \
            torch.pow(inputs_g, 2).sum(dim=1, keepdim=True).expand(n_g, n_q).t()
        dist.addmm_(inputs_q, inputs_g.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()

        mask = targets_q.view(n_q, 1).eq(targets_g.view(1, n_g))

        dist_ap, dist_an = [], []
        for i in range(n_q):
            pos_dist = dist[i][mask[i]]
            if len(pos_dist) > 0:
                dist_ap.append(pos_dist.max().unsqueeze(0))
            else:
                dist_ap.append(dist[i][i].unsqueeze(0))

            neg_dist = dist[i][mask[i] == 0]
            if len(neg_dist) > 0:
                dist_an.append(neg_dist.min().unsqueeze(0))
            else:
                dist_an.append(torch.tensor([self.margin], device=dist.device))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)
