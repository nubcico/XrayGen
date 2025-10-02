import os
import random
import torch
import numpy as np
from pathlib import Path
import torchvision.transforms.functional as F_tv
from omegaconf import OmegaConf
from ldm.models.diffusion.ddim import DDIMSampler
from utils import utils_inpainting

def inpaint_xray(
    xray_path,
    mask_path,
    checkpoint_path,
    config_path,
    prompt,
    num_inference_steps,
    guidance_scale,
    resolution,
    seed
):

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    if not Path(xray_path).exists():
        print(f"Error: X-ray file not found at {xray_path}")
        return None
    if not Path(mask_path).exists():
        print(f"Error: Mask file not found at {mask_path}")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        config = OmegaConf.load(config_path)
        model = utils_inpainting._load_model_from_config(config, checkpoint_path, device)
        sampler = DDIMSampler(model)
        
        original_image_tensor, mask_tensor = utils_inpainting._preprocess_image_and_mask(
            xray_path, mask_path, resolution
        )

        original_image_tensor = original_image_tensor.unsqueeze(0).to(model.device)
        mask_tensor = mask_tensor.unsqueeze(0).to(model.device)

        with torch.no_grad():
            z = model.encode_first_stage(original_image_tensor)
            z = model.get_first_stage_encoding(z)

            cond = model.get_learned_conditioning([prompt])

            latent_resolution = z.shape[-1]
            mask_for_sampler = F_tv.resize(mask_tensor, (latent_resolution, latent_resolution), antialias=True)
            mask_for_sampler = mask_for_sampler.mean(dim=1, keepdim=True).clamp(0, 1)
            
            print(f"Starting DDIM sampling for inpainting ({num_inference_steps} steps) with seed {seed}...")
            unconditional_conditioning = model.get_learned_conditioning([""])

            samples, _ = sampler.sample(
                S=num_inference_steps,
                conditioning=cond,
                batch_size=original_image_tensor.shape[0],
                shape=z.shape[1:], 
                verbose=False,
                unconditional_guidance_scale=guidance_scale,
                eta=0.0,
                x0=z, 
                mask=mask_for_sampler, 
                pneumonia_mask=mask_tensor, 
            )
            
            x_samples = model.decode_first_stage(samples)
            print("Inpainting complete.")
            
            return x_samples
            
    except Exception as e:
        print(f"An error occurred during inpainting: {e}")
        return None

def main():
    
    xray_path = r"resources\original_xray.png"
    mask_path = r"resources\path_mask.png"
    checkpoint_path = r"inpainting.ckpt"
    config_path = r"v1-inference_inpaint.yaml"
    prompt = "pneumonia chest xray"
    num_inference_steps = 125
    guidance_scale = 10.0
    resolution = 512
    seed = 42

    output_dir = r"generated_xrays"
    os.makedirs(output_dir, exist_ok=True)

    print("\nStarting inpainting process...")
    inpainted_tensor = inpaint_xray(
        xray_path=xray_path,
        mask_path=mask_path,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        resolution=resolution,
        seed=seed
    )

    if inpainted_tensor is not None:
        utils_inpainting._save_inpainted_image(inpainted_tensor, xray_path, output_dir)
    else:
        print("\nInpainting failed.")

if __name__ == "__main__":
    main()
