import torch
import numpy as np
from PIL import Image
from pathlib import Path

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from utils import utils

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def generate_xray(model, mask, prompt, neg_prompt, steps, guidance):

    sampler = DDIMSampler(model)
    
    with torch.inference_mode():
        conditioning = model.get_learned_conditioning([prompt])
        unconditional_conditioning = model.get_learned_conditioning([neg_prompt])
        shape = [4, 64, 64] 

        control_features = model.organ_config(mask)

        latent_samples, _ = sampler.sample(
            S=steps,
            conditioning=conditioning,
            batch_size=1,
            shape=shape,
            unconditional_guidance_scale=guidance,
            unconditional_conditioning=unconditional_conditioning,
            eta=1.0,
            control_features=control_features
        )
        
        image_tensor = model.decode_first_stage(latent_samples)
        image_tensor = torch.clamp((image_tensor + 1.0) / 2.0, min=0.0, max=1.0)
        
        image_array = (255. * image_tensor[0].cpu().permute(1, 2, 0).numpy()).astype(np.uint8)
        return Image.fromarray(image_array)

def main():
    model_config_path = r"v1-inference.yaml"
    checkpoint_path = r"base_checkpoint.ckpt"
    feature_extractor_ckpt = r"FE_checkpoint.ckpt"
    
    organ_mask_path = r"resources/anatomical_map.png"
    bone_mask_path = r"resources/bone_mask.png"
    output_path = Path(r"generated_xrays/generated_xray.png")

    prompt = "normal chest xray, high resolution, diagnostic quality"
    neg_prompt = "blurry, low quality, artifact, noise, bad anatomy"
    num_steps = 150
    guidance_scale = 12.5
    
    print("Loading the model...")
    model = utils.load_model(model_config_path, checkpoint_path, feature_extractor_ckpt)
    if model is None:
        print("Failed to load the model. Exiting.")
        return
    model.to(DEVICE).eval()
    print("Model loaded successfully!")

    print("Processing input masks...")
    mask_tensor = utils.merge_and_preprocess_masks(
        str(organ_mask_path), 
        str(bone_mask_path), 
        target_size=512
    )
    if mask_tensor is None:
        print("Mask processing failed. Exiting.")
        return

    print(f"Generating image with {num_steps} steps...")
    try:
        generated_image = generate_xray(
            model, 
            mask_tensor, 
            prompt, 
            neg_prompt, 
            num_steps, 
            guidance_scale
        )
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generated_image.save(output_path)
        print(f"\nDone! Image saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred during generation: {e}")
        if "out of memory" in str(e).lower() and torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()