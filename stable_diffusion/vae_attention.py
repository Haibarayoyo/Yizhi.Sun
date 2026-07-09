#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class vae_attention(nn.Module):
    def __init__(self,heads,d_model,dropout = 0.2):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.d_head = d_model // heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = False):
        B, T, C = x.shape
        # 标准多头拆分逻辑
        q = self.q_proj(x).view(B, T, self.heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.heads, self.d_head).transpose(1, 2)

        # 缩放点积注意力
        attn_weight = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)
        attn_weight = f.softmax(attn_weight, dim=-1)
        attn_weight = self.dropout(attn_weight)

        out = attn_weight @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.dropout(out)
        return out

