"""LD-VQN: CLIP ViT-L/14-336 + Multi-Level Adapter + stacked D-TCR blocks + DeBERTa.

Parameter names (``vision_encoder``, ``deberta``, ``proj``, ``ml_adapter``,
``fusion_module``, ``gem``, ``bn``, ``classifier``) are identical to the
training notebook, so released checkpoints load with ``strict=True``.
"""
import torch
import torch.nn as nn
from transformers import CLIPVisionModel

from .d_tcr import DynamicTextConditionedRouting
from .mla import MultiLevelAdapter
from .text_encoders import build_deberta


class GeM(nn.Module):
    """Generalized-mean pooling over the token dimension."""

    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1. / self.p)


class LDVQN(nn.Module):
    def __init__(self, num_classes, cfg_model):
        super().__init__()
        m = cfg_model

        # ---------------- vision encoder (CLIP) ----------------
        self.vision_encoder = CLIPVisionModel.from_pretrained(m["VISION_BACKBONE"])
        # transformers compatibility: older versions wrap the encoder in `vision_model`
        vision_core = getattr(self.vision_encoder, "vision_model", self.vision_encoder)

        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for layer in vision_core.encoder.layers[m["VISION_TRAINABLE_FROM"]:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in vision_core.post_layernorm.parameters():
            p.requires_grad = True

        cfg_v = self.vision_encoder.config
        self.v_dim = cfg_v.hidden_size if hasattr(cfg_v, "hidden_size") else cfg_v.vision_config.hidden_size
        print(f"[CLIP] hidden_dim={self.v_dim} | layers={len(vision_core.encoder.layers)}")

        # ---------------- text encoder (DeBERTa) ----------------
        self.deberta = build_deberta(m["TEXT_BACKBONE"], m["TEXT_TRAINABLE_FROM"])
        self.proj = nn.Linear(self.deberta.config.hidden_size, self.v_dim)

        # ---------------- LD-VQN modules ----------------
        self.ml_adapter = MultiLevelAdapter(self.v_dim, m["CLIP_LAYERS"])
        self.fusion_module = nn.ModuleList([
            DynamicTextConditionedRouting(self.v_dim, m["NUM_HEADS"], m["NUM_COMPONENTS"])
            for _ in range(m["Q_LAYERS"])
        ])
        self.gem = GeM()
        self.bn = nn.BatchNorm1d(self.v_dim)
        self.classifier = nn.Linear(self.v_dim, num_classes, bias=False)

    def forward(self, img, ids, mask, return_attn=False):
        """
        train mode : (logits, global_feat)            or ((logits, global_feat), attns)
        eval mode  : bn_feat (retrieval embedding)    or (bn_feat, attns)
        """
        with torch.set_grad_enabled(self.training):
            v_out = self.vision_encoder(pixel_values=img.to(next(self.vision_encoder.parameters()).dtype),
                                        output_hidden_states=True)

        img_f = self.ml_adapter(v_out.hidden_states)
        txt_f = self.proj(self.deberta(input_ids=ids, attention_mask=mask).last_hidden_state)

        attns = []
        for layer in self.fusion_module:
            txt_f, a_dict = layer(txt_f, img_f)
            if return_attn:
                attns.append(a_dict)

        g_feat = self.gem(txt_f)
        bn_feat = self.bn(g_feat)

        if self.training:
            out = (self.classifier(bn_feat), g_feat)
        else:
            out = bn_feat
        return (out, attns) if return_attn else out


def load_weights(model, path, device="cpu"):
    """Load either a full training checkpoint (``model_state_dict``) or a plain state dict."""
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    return model


def num_classes_from_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    return state["classifier.weight"].shape[0]
