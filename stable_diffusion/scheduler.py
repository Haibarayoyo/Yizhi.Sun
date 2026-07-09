#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32, device=device)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-4, 0.9999)


# 用原来的 cosine betas 构建 scheduler（与已训 checkpoint 兼容）
trained_betas = cosine_beta_schedule(T).cpu().float().numpy()
noise_scheduler = DDPMScheduler(
    num_train_timesteps=T,
    trained_betas=trained_betas,
    prediction_type="epsilon",
    clip_sample=False,
)
ddim_scheduler = DDIMScheduler.from_config(noise_scheduler.config)

betas = noise_scheduler.betas.to(device)
alphas = noise_scheduler.alphas.to(device)
alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
print("noise_scheduler 已就绪，prediction_type=epsilon")

