"""DeBERTa-v3 text encoder with partial fine-tuning."""
from transformers import DebertaV2Model


def build_deberta(name="microsoft/deberta-v3-base", trainable_from=9):
    """Load DeBERTa, freeze everything, then unfreeze encoder layers ``trainable_from`` .. end."""
    model = DebertaV2Model.from_pretrained(name)
    for p in model.parameters():
        p.requires_grad = False
    for layer in model.encoder.layer[trainable_from:]:
        for p in layer.parameters():
            p.requires_grad = True
    return model
