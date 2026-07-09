#!/usr/bin/env python
# coding: utf-8

# In[ ]:


ROOT_DIR = "/root/PetImages"
CSV_NAME = "output.csv"
CACHE_ROOT = os.path.join(ROOT_DIR, "latent_cache")
VAE_PATH = "./sd-vae-ft-ema"
VAE_SCALE = 0.18215
T = 1000


# In[ ]:


from diffusers import AutoencoderKL, UNet2DModel, DDPMScheduler, DDIMScheduler
accelerator = Accelerator(mixed_precision="fp16",gradient_accumulation_steps=4)
device = accelerator.device      
set_seed(1) 

vae = AutoencoderKL.from_pretrained(
    "./sd-vae-ft-ema",
    local_files_only=True
).to(device).eval()
vae_scale = vae.config.scaling_factor
vae.requires_grad_(False)


# In[ ]:


class CatsLatentDataset(Dataset):
    def __init__(self, csv_file, root_dir, cache_dir):
        self.annotations = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        valid_idx = []
        for idx in range(len(self.annotations)):
            rel_img_path = self.annotations.iloc[idx, 0]
            cache_path = rel_img_path.replace(".jpg", "_latent.pt").replace(".png", "_latent.pt")
            full_cache = os.path.join(self.cache_dir, cache_path)
            if os.path.exists(full_cache):
                valid_idx.append(idx)
        self.annotations = self.annotations.iloc[valid_idx].reset_index(drop=True)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        rel_img_path = self.annotations.iloc[idx, 0]
        cache_path = rel_img_path.replace(".jpg", "_latent.pt").replace(".png", "_latent.pt")
        full_cache = os.path.join(self.cache_dir, cache_path)
        latent = torch.load(full_cache, map_location="cpu")
        return latent


# In[ ]:


def dataset():
    train_ds = CatsLatentDataset(csv_file=CSV_NAME,root_dir=ROOT_DIR,cache_dir=CACHE_ROOT)
    loader = DataLoader(train_ds,batch_size=32,shuffle=True,drop_last=True,num_workers=0,pin_memory=False)
    return loader

