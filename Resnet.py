#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,out_ch,3,1,1)
        self.norm1 = nn.GroupNorm(min(32,in_ch),in_ch)
        # 改动：输出 out\_ch\*2，分别对应 scale、shift，实现 AdaGN
        self.time = nn.Linear(time_dim, out_ch* 2)
        self.conv2 = nn.Conv2d(out_ch,out_ch,3,1,1)
        self.norm2 = nn.GroupNorm(min(8,out_ch),out_ch)
        self.silu = nn.SiLU()
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch,out_ch,1)
        else:
            self.shortcut = nn.Identity()

    def forward(self,x,t):
        h=self.norm1(x)
        h=self.silu(h)
        h=self.conv1(h)

        # AdaGN 核心改动，替代简单相加
        time_params = self.time(t)
        scale, shift = torch.chunk(time_params, 2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        h=self.norm2(h)
        h=self.silu(h)
        h=self.conv2(h)
        return h+self.shortcut(x)

