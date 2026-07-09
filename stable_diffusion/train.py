#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def forward_diffusion_sample(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    x_t = noise_scheduler.add_noise(x_0, noise, t)
    return x_t, noise


def compute_loss(noise_pred, noise, t, gamma=5.0):
    """Min-SNR 加权 MSE"""
    alpha_hat = alphas_cumprod[t][:, None, None, None].float()
    snr = alpha_hat / (1.0 - alpha_hat)
    weight = torch.minimum(snr, torch.full_like(snr, gamma)) / snr
    loss = weight * (noise_pred.float() - noise.float()).pow(2)
    return loss.mean()


def decode_latents(vae, latents, vae_scale):
    imgs = vae.decode(latents.float() / vae_scale).sample
    return torch.clamp((imgs + 1) / 2, 0, 1)


# In[ ]:


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)
dataloader = dataset()
model, optimizer, dataloader, scheduler = accelerator.prepare(
    model, optimizer, dataloader, scheduler
)
ema = None
if accelerator.is_main_process:
    unwrapped_model = accelerator.unwrap_model(model)
    ema = EMA(unwrapped_model, decay=0.999)


# In[ ]:


for epoch in range(300):
    model.train()
    for step, latent in enumerate(dataloader):
        latent = latent.to(device)
        with accelerator.accumulate(model):
            t = torch.randint(0, T, (latent.shape[0],), device=device)
            latent_noisy, noise = forward_diffusion_sample(latent.float(), t)
            noise_pred = model(latent_noisy, t)
            loss = compute_loss(noise_pred, noise, t)

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if accelerator.is_main_process and accelerator.sync_gradients:
                ema.update()

        if accelerator.is_main_process and step % 100 == 0:
            print(f'Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}')

    scheduler.step()

    # 主进程采样可视化 + 对比诊断
    if accelerator.is_main_process:
        model.eval()
        sample_vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).to(device).float().eval()
        sample_vae.requires_grad_(False)
        sample_vae_scale = sample_vae.config.scaling_factor
        used_ema = False

        with torch.no_grad():
            latent_sample = None
            if epoch >= 20:
                ema.apply_shadow()
                used_ema = True
            eval_model = accelerator.unwrap_model(model)

            # 推荐 DDIM：50 步快速 / 100 步正式评估
            ddim_steps = 50 if epoch < 50 else 100
            latent_sample = sample_ddim(eval_model, n_samples=4, steps=ddim_steps, eta=0.0)

            if latent_sample is None:
                print(f"[Epoch {epoch}] 采样失败（出现 NaN/Inf），跳过画图")
            else:
                print(
                    f"[Epoch {epoch}] 生成latent std: {latent_sample.std():.3f}, "
                    f"min/max: {latent_sample.min():.3f}/{latent_sample.max():.3f}"
                )

                gen_img = decode_latents(sample_vae, latent_sample, sample_vae_scale)

                real_latent = next(iter(dataloader))[:4].to(device).float()
                print(
                    f"[Epoch {epoch}] 真实latent std: {real_latent.std():.3f}, "
                    f"min/max: {real_latent.min():.3f}/{real_latent.max():.3f}"
                )
                real_img = decode_latents(sample_vae, real_latent, sample_vae_scale)

                compare = make_grid(torch.cat([real_img, gen_img], dim=0), nrow=4)
                plt.figure(figsize=(8, 4))
                plt.imshow(compare.cpu().permute(1, 2, 0).numpy())
                plt.title(f"Epoch {epoch} | 上:真实 下:DDIM生成 | steps={ddim_steps} | EMA={used_ema}")
                plt.axis("off")
                plt.savefig(f"compare_epoch_{epoch}.png", bbox_inches="tight")
                plt.close()

                # 仅生成样本（方便快速浏览）
                gen_grid = make_grid(gen_img, nrow=2)
                plt.figure(figsize=(6, 6))
                plt.imshow(gen_grid.cpu().permute(1, 2, 0).numpy())
                plt.axis("off")
                plt.title(f"Samples at Epoch {epoch}")
                plt.savefig(f"epoch_{epoch}.png", bbox_inches="tight")
                plt.close()

            if used_ema:
                ema.restore()

        del sample_vae
        if latent_sample is not None:
            del latent_sample, gen_img, real_img, compare, gen_grid
        gc.collect()
        torch.cuda.empty_cache()

