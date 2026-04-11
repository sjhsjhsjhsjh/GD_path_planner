from typing import Tuple

import torch
import torch.nn as nn


class MultiScaleDuelingQNetwork(nn.Module):
    def __init__(
        self,
        scalar_dim: int,
        patch_channels: int,
        action_feat_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.action_dim = int(action_dim)

        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Shared encoder for both small/large patches.
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(patch_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.action_feat_mlp = nn.Sequential(
            nn.Linear(action_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        fusion_dim = hidden_dim + 64 + 64 + hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def _encode_patch(self, patch: torch.Tensor) -> torch.Tensor:
        x = self.patch_encoder(patch)
        return x.flatten(start_dim=1)

    def forward(
        self,
        scalar: torch.Tensor,
        patch_s: torch.Tensor,
        patch_l: torch.Tensor,
        action_feat: torch.Tensor,
    ) -> torch.Tensor:
        scalar_f = self.scalar_mlp(scalar)
        small_f = self._encode_patch(patch_s)
        large_f = self._encode_patch(patch_l)
        act_f = self.action_feat_mlp(action_feat)

        fused = torch.cat([scalar_f, small_f, large_f, act_f], dim=1)
        fused = self.fusion(fused)

        value = self.value_head(fused)
        adv = self.adv_head(fused)
        q = value + (adv - adv.mean(dim=1, keepdim=True))
        return q
