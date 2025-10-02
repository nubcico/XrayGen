import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse

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

def main(args):
    model_config_path = r"v1-inference.yaml"
    checkpoint_path = r"base_checkpoint.ckpt"
    feature_extractor_ckpt = r"FE_checkpoint.ckpt"
    
    output_path = Path(args.output_dir) / "generated_xray.png"

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

    if args.no_control:
        print("Generating without control, using a zero mask.")
        mask_tensor = torch.zeros(1, 2, 512, 512, device=DEVICE)
    else:
        print("Processing input masks for controlled generation...")
        mask_tensor = utils.merge_and_preprocess_masks(
            args.anatomic_map, 
            args.bone_map, 
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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate a chest X-ray using a diffusion model.")
    
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the generated X-ray image.")
    
    control_group = parser.add_mutually_exclusive_group(required=True)
    control_group.add_argument('--no_control', action='store_true', help="Generate without anatomical control (uses a zero mask).")
    control_group.add_argument('--control', action='store_true', help="Generate with anatomical control maps.")
    
    parser.add_argument('--anatomic_map', type=str, help="Path to the anatomical map PNG file.")
    parser.add_argument('--bone_map', type=str, help="Path to the bone map PNG file.")
    
    args = parser.parse_args()
    
    if args.control and (not args.anatomic_map or not args.bone_map):
        parser.error("--anatomic_map and --bone_map are required when using --control.")
        
    return args

if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
