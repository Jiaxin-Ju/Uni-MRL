"""LLM-rule-only ablation model used in the paper's component study."""

import torch
from torch import nn


class LLMOnly(nn.Module):
    def __init__(
        self,
        task="classification",
        feat_dim=512,
        llm4sd_x_dim=0,
        pred_n_layer=2,
        pred_act="softplus",
        **_unused,
    ):
        super().__init__()
        if llm4sd_x_dim <= 0:
            raise ValueError("llm4sd_x_dim must be positive")
        self.llm4sd_proj = nn.Sequential(
            nn.Linear(llm4sd_x_dim, feat_dim * 2),
            nn.ReLU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )
        output_dim = 2 if task == "classification" else 1
        hidden_dim = feat_dim // 2
        activation = nn.ReLU if pred_act == "relu" else nn.Softplus
        layers = [nn.Linear(feat_dim, hidden_dim), activation()]
        for _ in range(max(1, pred_n_layer) - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), activation()))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.pred_head = nn.Sequential(*layers)

    def forward(self, data):
        rule_h = self.llm4sd_proj(data.llm4sd_x)
        return rule_h, self.pred_head(rule_h)
