"""Multi-Level Adapter (MLA).

Takes the CLIP hidden states of the anchor layers, projects each one to
``embed_dim // 4`` and fuses the concatenation back to ``embed_dim``.
"""
import torch
import torch.nn as nn


class MultiLevelAdapter(nn.Module):
    def __init__(self, embed_dim, layers):
        super().__init__()
        self.layers = list(layers)
        self.adapters = nn.ModuleList([nn.Linear(embed_dim, embed_dim // 4) for _ in self.layers])
        self.proj = nn.Linear((embed_dim // 4) * len(self.layers), embed_dim)

    def forward(self, states):
        selected = [adj(states[i].to(self.proj.weight.dtype)) for adj, i in zip(self.adapters, self.layers)]
        return self.proj(torch.cat(selected, dim=-1))
