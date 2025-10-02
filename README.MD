# XrayGen: A latent-aligned diffusion framework for chest X-ray (CXR) synthesis

## Inference

The entire codes for training pipeline will be added soon. This page provide inference of the model.

## Overview

> **Abstract:** *We introduce Latent-Aligned Diffusion (LADiff), a controllable chest X-ray generation framework that integrates semantic and spatial conditioning. The model learns a structured latent space representing organ locations, lesion attributes, and appearance features, which are decoded into spatial maps to guide a diffusion generator.*  

LADiff supports three controllable pipelines:

Text-to-Image: synthesis from descriptive clinical prompts

Coordinates-to-Image: generation from anatomical and pathological coordinates

Image-to-Image: synthesis aligned with a reference CXR

Trained on MIMIC-CXR, LADiff produces anatomically consistent and pathology-aware radiographs, and improves diagnostic classification when used for data augmentation. The framework enables realistic synthesis with precise control, supporting applications in augmentation, evaluation, and counterfactual analysis.

## Setup

1. Clone the repository:

```bash
git clone https://github.com/nubcico/XrayGen.git
cd XrayGen
```

2. Set Up Your Environment
Install all the necessary Python packages using the requirements.txt file.

```
pip install -r requirements.txt
```

### Controlled Generation (Using Maps)
To create an X-ray based on specific anatomical and bone structures, use the --control flag. You will need to provide paths to your two map images.

```
python generate_xray.py --control \
                        --anatomic_map "resources/anatomical_map_1.png" \
                        --bone_map "resources/bone_mask_1.png" \
                        --output_dir "generated_xrays"
```

### Uncontrolled Generation
If you don't need to follow a specific structure, use the --no_control flag. The model will generate an CXR image without specific organ/bone structures.

```
python generate_xray.py --no_control \
                        --output_dir "generated_xrays"
```

### Inpainting pneumonia pathology
To inpaint heathy xray with pneumonia, you need to provide paths to the xray, the binary mask defining the area to change (white is background, black is pathology area), and a directory for the output.
```
python inpaint_xray.py --xray_path "resources/original_xray_1.png" \
                       --mask_path "resources/path_mask_1.png" \
                       --output_dir "generated_xrays"
```

## References

This work builds upon the following repositories:

- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [stable-diffusion](https://github.com/CompVis/stable-diffusion/)
