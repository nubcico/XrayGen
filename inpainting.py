import torch
import numpy as np
import random
import argparse
from pathlib import Path
import torchvision.transforms.functional as F_tv
from omegaconf import OmegaConf

from ldm.models.diffusion.ddim import DDIMSampler
from utils import utils_inpainting

def perform_inpainting(
    model,
    sampler,
    xray_path,
    mask_path,
    prompt,
    steps,
    guidance,
    resolution,
    seed
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    try:
        image_tensor, mask_tensor = utils_inpainting._preprocess_image_and_mask(
            xray_path, mask_path, resolution
        )

        image_tensor = image_tensor.unsqueeze(0).to(model.device)
        mask_tensor = mask_tensor.unsqueeze(0).to(model.device)

        with torch.no_grad():
            initial_latent = model.get_first_stage_encoding(model.encode_first_stage(image_tensor))
            conditioning = model.get_learned_conditioning([prompt])
            unconditional_conditioning = model.get_learned_conditioning([""])

            latent_res = initial_latent.shape[-1]
            sampler_mask = F_tv.resize(mask_tensor, (latent_res, latent_res), antialias=True)
            sampler_mask = sampler_mask.mean(dim=1, keepdim=True).clamp(0, 1)
            
            print(f"Beginning the inpainting process with {steps} steps...")

            latent_samples, _ = sampler.sample(
                S=steps,
                conditioning=conditioning,
                batch_size=image_tensor.shape[0],
                shape=initial_latent.shape[1:],
                verbose=False,
                unconditional_guidance_scale=guidance,
                unconditional_conditioning=unconditional_conditioning,
                eta=0.0,
                x0=initial_latent,
                mask=sampler_mask,
                pneumonia_mask=mask_tensor,
            )
            
            inpainted_image = model.decode_first_stage(latent_samples)
            print("Inpainting finished successfully.")
            return inpainted_image

    except Exception as e:
        print(f"An error occurred during the inpainting process: {e}")
        return None

def main(args):
    xray_path = Path(args.xray_path)
    mask_path = Path(args.mask_path)
    output_dir = Path(args.output_dir)

    checkpoint_path = "inpainting.ckpt"
    config_path = "v1-inference_inpaint.yaml"
    prompt = "pneumonia chest xray"
    steps = 125
    guidance = 10.0
    resolution = 512
    seed = 42

    if not xray_path.exists():
        print(f"Error: The specified X-ray file could not be found at {xray_path}")
        return
    if not mask_path.exists():
        print(f"Error: The specified mask file could not be found at {mask_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading the inpainting model and configuration...")
    config = OmegaConf.load(config_path)
    model = utils_inpainting._load_model_from_config(config, checkpoint_path, device)
    sampler = DDIMSampler(model)
    print("Model loaded.")

    inpainted_tensor = perform_inpainting(
        model=model,
        sampler=sampler,
        xray_path=str(xray_path),
        mask_path=str(mask_path),
        prompt=prompt,
        steps=steps,
        guidance=guidance,
        resolution=resolution,
        seed=seed
    )

    if inpainted_tensor is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        utils_inpainting._save_inpainted_image(inpainted_tensor, str(xray_path), str(output_dir))
    else:
        print("The inpainting process failed and no image was saved.")

def setup_arg_parser():
    parser = argparse.ArgumentParser(description="Perform inpainting on a chest X-ray image.")
    
    parser.add_argument("--xray_path", type=str, required=True, help="Path to the original X-ray image file.")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the mask image file for inpainting.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the output image will be saved.")
    
    return parser.parse_args()

if __name__ == "__main__":
    arguments = setup_arg_parser()
    main(arguments)

