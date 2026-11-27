"""Orthogonal disentanglement loss: || N N^T - I ||_F on L2-normalised component nodes."""
import torch
import torch.nn.functional as F


def orthogonal_disentanglement_loss(component_nodes):
    nodes = F.normalize(component_nodes.squeeze(0), p=2, dim=-1)
    sim = torch.mm(nodes, nodes.t())
    return torch.norm(sim - torch.eye(nodes.shape[0], device=nodes.device), p="fro")
