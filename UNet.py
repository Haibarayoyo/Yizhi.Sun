#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class UNetAttention(nn.Module):
    def __init__(self, n_head, n_embd):
        super().__init__()
        channels = n_head * n_embd
        self.gn = nn.GroupNorm(32, channels)
        self.conv_in = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.ln1 = nn.LayerNorm(channels)
        self.att = vae_attention(n_head, channels)

        self.ln2 = nn.LayerNorm(channels)
        self.gelu_linear = nn.Linear(channels, channels * 4 * 2)
        self.gelu_linear2 = nn.Linear(channels * 4, channels)

        self.conv_out = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x):
        fx = self.gn(x)
        fx = self.conv_in(fx)
        B, C, H, W = x.shape
        fx = fx.view(B, C, H*W).transpose(-2, -1)

        ax = fx + self.att(self.ln1(fx))

        sx = ax.clone()
        cx = self.ln2(ax)
        gx, gate = self.gelu_linear(cx).chunk(2, dim = -1)
        gx = gx * f.gelu(gate)
        gx = self.gelu_linear2(gx)
        gx = gx + sx

        gx = gx.transpose(-2, -1).reshape(B, C, H, W)
        out = x + self.conv_out(gx)
        return out


# In[ ]:


class Downsample(nn.Module):
    def __init__(self,in_channels,out_channels = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.conv = nn.Conv2d(self.in_channels,self.out_channels,3,stride = 2,padding = 1)

    def forward(self,x):
        x = self.conv(x)
        return x


# In[ ]:


class Upsample(nn.Module):
    def __init__(self,in_channels,out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(self.in_channels,self.out_channels,3,padding = 1)

    def forward(self,x):
        _,_,h,w = x.shape
        output_size = (h*2,w*2)
        x = f.interpolate(hidden_states, size=output_size, mode="nearest")
        x = self.conv(x)
        return x


# In[ ]:


UNetResidual = ResBlock

class UNetEncoder(nn.Module):
    def __init__(self,channels = 128,time_dim = 256, res_stack_num=2):
        super().__init__()
        self.time_dim = time_dim
        self.res_stack_num = res_stack_num
        self.block1 = nn.Sequential(
            nn.Conv2d(4, channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, channels), channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU()
        )
        res1_layers = []
        for i in range(self.res_stack_num):
            in_c = channels if i==0 else channels
            res1_layers.append(UNetResidual(in_c,channels,time_dim))
        self.res1 = nn.Sequential(*res1_layers)

        self.block2 = nn.Sequential(
            Downsample(channels),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, channels), channels * 2),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1),
            nn.SiLU()
        )
        res2_layers = []
        for i in range(self.res_stack_num):
            in_c = channels*2 if i==0 else channels*2
            res2_layers.append(UNetResidual(in_c,channels*2,time_dim))
        self.res2 = nn.Sequential(*res2_layers)

        self.block3 =  nn.Sequential(
            Downsample(channels*2),,
            nn.Conv2d(channels * 2, channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, channels), channels * 4),
            nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=1),
            nn.SiLU()
        )
        res3_layers = []
        for i in range(self.res_stack_num):
            in_c = channels*4 if i==0 else channels*4
            res3_layers.append(UNetResidual(in_c,channels*4,time_dim))
        self.res3 = nn.Sequential(*res3_layers)
        self.att3 = UNetAttention(8, channels * 4 // 8)

    def forward(self, x, time_emb):
        skips = []
        x = self.block1(x)
        x = self.res1(x,time_emb)
        skips.append(x)

        x = self.block2(x)
        x = self.res2(x,time_emb)
        skips.append(x)

        x = self.block3(x)
        x = self.res3(x,time_emb)
        x = self.att3(x)
        skips.append(x)
        return x, skips


# In[ ]:


class UNetDecoder(nn.Module):
    def __init__(self, channels=128,time_dim = 256,res_stack_num=2):
        super().__init__()
        self.time_dim = time_dim
        self.res_stack_num = res_stack_num
        self.block1 = nn.Sequential(
            nn.Conv2d(channels * 8, channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, channels), channels * 4),
            nn.Conv2d(channels * 4, channels * 2, kernel_size=3, padding=1),
            Upsample(channels*2),
            nn.SiLU()
        )
        res1_layers = [UNetResidual(channels*2, channels*2, time_dim) for _ in range(res_stack_num)]
        self.res1 = nn.Sequential(*res1_layers)

        self.block2 = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, channels), channels * 2),
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            Upsample(channels),
            nn.SiLU()
        )
        res2_layers = [UNetResidual(channels, channels, time_dim) for _ in range(res_stack_num)]
        self.res2 = nn.Sequential(*res2_layers)

        self.block3 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.SiLU()
        )
        res3_layers = [UNetResidual(channels, channels, time_dim) for _ in range(res_stack_num)]
        self.res3 = nn.Sequential(*res3_layers)
        self.att3 = UNetAttention(8, channels // 8)

        self.output_layer = nn.Sequential(
            nn.GroupNorm(min(32, channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, 4, kernel_size=3, padding=1)
        )

    def forward(self, x, skips, emb_time):
        x = torch.cat([x,skips.pop()],dim=1)
        x = self.block1(x)
        x = self.res1(x,emb_time)

        x = torch.cat([x,skips.pop()],dim=1)
        x = self.block2(x)
        x = self.res2(x,emb_time)

        x = torch.cat([x,skips.pop()],dim=1)
        x = self.block3(x)
        x = self.res3(x,emb_time)
        x = self.att3(x)
        x = self.output_layer(x)
        return x


# In[ ]:


class UNet(nn.Module):
    def __init__(self, channels = 128, time_dim = 256, res_stack_num = 2):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        self.encoders = UNetEncoder(channels,time_dim,res_stack_num)
        self.decoders = UNetDecoder(channels, time_dim)

        bottleneck_ch_in = channels * 4
        bottleneck_ch_out = channels * 8

        self.bottleneck_down = Downsample(bottleneck_ch_in)
        self.bottleneck_conv_in = nn.Conv2d(bottleneck_ch_in,bottleneck_ch_out,3,1,1)
        self.bottleneck_res_stack = nn.ModuleList()
        for i in range(res_stack_num):
            in_c = bottleneck_ch_out if i == 0 else bottleneck_ch_out
            self.bottleneck_res_stack.append(UNetResidual(in_c, bottleneck_ch_out, time_dim))

        self.bottleneck_conv_out = nn.Conv2d(bottleneck_ch_out, bottleneck_ch_in, 3, 1, 1)
        self.bottleneck_up = Upsample(bottleneck_ch_in)

    def forward(self, x, t):
        time_emb = self.time_embed(t)
        x, skips = self.encoders(x, time_emb)
        x = self.bottleneck_down(x)
        x = self.bottleneck_conv_in(x)
        for res_block in self.bottleneck_res_stack:
            x = res_block(x,time_emb)
        x = self.bottleneck_att(x)
        x = self.bottleneck_conv_out(x)
        x = self.bottleneck_up(x)
        x = self.decoders(x, skips, time_emb)
        return x


# In[ ]:


class DiffusionModel(nn.Module):
    def __init__(self,in_ch = 128, time_dim=256):
        super().__init__()
        self.unet = UNet(in_ch,time_dim)

    def forward(self,x,t):
        return self.unet(x,t)


model = DiffusionModel(in_ch=128, time_dim=256)

