# XrayGen: A latent-aligned diffusion framework for chest X-ray (CXR) synthesis

## Inference

The entire codes for training pipeline will be added soon. This page provide inference of the model.

## Overview

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

3. Controlled Generation (Using Maps)
To create an X-ray based on specific anatomical and bone structures, use the --control flag. You will need to provide paths to your two map images.

```
python generate_xray.py --control \
                        --anatomic_map "resources/anatomical_map_1.png" \
                        --bone_map "resources/bone_mask_1.png" \
                        --output_dir "generated_xrays"
```

4. Uncontrolled Generation
If you don't need to follow a specific structure, use the --no_control flag. The model will generate an CXR image without specific organ/bone structures.

```
python generate_xray.py --no_control \
                        --output_dir "generated_xrays"
```

## References

This work builds upon the following repositories:

- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [stable-diffusion](https://github.com/CompVis/stable-diffusion/)
