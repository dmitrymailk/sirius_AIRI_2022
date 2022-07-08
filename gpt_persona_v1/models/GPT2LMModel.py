import torch
from torch import nn
from transformers import GPT2PreTrainedModel, GPT2Model, GPT2Config


class GPT2LMModel(GPT2PreTrainedModel):
    def __init__(self, base_model_name=None, n_classes=None, emb_size=None):
        super().__init__(GPT2Config())
        self.transformer = GPT2Model.from_pretrained(base_model_name)
        self.lm_head = nn.Linear(emb_size, n_classes, bias=False)

    def forward(self, x):
        output = self.transformer(x)
        output = output[0]
        logits = self.lm_head(output)
        return logits
