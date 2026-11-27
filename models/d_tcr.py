"""Dynamic Text-Conditioned Routing (D-TCR).

1. The pooled text embedding predicts a per-head soft adjacency over M latent
   component nodes (sigmoid output, added to the node self-attention logits).
2. Nodes attend to each other (graph self-attention), then to the image tokens.
3. Text tokens read the routed nodes back (cross-attention) and pass an FFN.
"""
import torch
import torch.nn as nn


class DynamicTextConditionedRouting(nn.Module):
    def __init__(self, embed_dim, num_heads=8, num_components=16):
        super().__init__()
        self.num_components, self.num_heads = num_components, num_heads
        self.component_nodes = nn.Parameter(torch.randn(1, num_components, embed_dim))
        nn.init.xavier_uniform_(self.component_nodes)
        self.text_to_graph = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(),
            nn.Linear(embed_dim // 2, num_heads * num_components ** 2), nn.Sigmoid())
        self.graph_self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_image_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_text_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(embed_dim * 2, embed_dim))

    def forward(self, text_embeds, image_embeds):
        batch_size = image_embeds.shape[0]
        nodes = self.component_nodes.expand(batch_size, -1, -1)
        dynamic_adj = self.text_to_graph(text_embeds.mean(dim=1)).view(
            batch_size * self.num_heads, self.num_components, self.num_components)
        n_graph, _ = self.graph_self_attn(nodes, nodes, nodes, attn_mask=dynamic_adj)
        nodes = self.norm1(nodes + n_graph)
        n_img, img_attn = self.cross_image_attn(nodes, image_embeds, image_embeds)
        nodes = self.norm2(nodes + n_img)
        t_routed, _ = self.cross_text_attn(text_embeds, nodes, nodes)
        out = self.ffn(self.norm3(text_embeds + t_routed))
        # img_attn: (B, M, N_img) node-to-image-token attention, averaged over heads
        return out, {"img_attn": img_attn, "nodes": self.component_nodes}

