#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, dim, flip_sin_to_cos=False, downscale_freq_shift=0):
        super().__init__()
        self.dim = dim
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        half = self.dim // 2
        freq = torch.arange(half, dtype=torch.float32, device=t.device)
        emb_scale = math.log(10000) / (half - self.downscale_freq_shift)
        freq = freq * -emb_scale

        emb = t[:, None].float() * freq[None, :]
        if self.flip_sin_to_cos:
            emb = torch.cat([emb.cos(), emb.sin()], dim=1)
        else:
            emb = torch.cat([emb.sin(), emb.cos()], dim=1)

        return self.mlp(emb)

