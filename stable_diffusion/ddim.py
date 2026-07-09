#!/usr/bin/env python
# coding: utf-8

# In[ ]:


@torch.no_grad()
def sample_ddim(model, n_samples=4, steps=50, eta=0.0):
    model.eval()
    ddim = DDIMScheduler.from_config(noise_scheduler.config)
    ddim.set_timesteps(steps, device=device)

    x = torch.randn(n_samples, 4, 64, 64, device=device, dtype=torch.float32)

    for i, t in enumerate(ddim.timesteps):
        t_batch = t.expand(n_samples).long()
        with torch.autocast(device_type=accelerator.device.type, enabled=False):
            pred_noise = model(x, t_batch).float()

        step_out = ddim.step(model_output=pred_noise, timestep=t, sample=x, eta=eta)
        x = step_out.prev_sample

    print(f"DDIM latent: min={x.min():.3f}, max={x.max():.3f}, std={x.std():.3f}, steps={steps}")
    return x

