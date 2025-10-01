# XrayGen: A latent-aligned diffusion framework for chest X-ray (CXR) synthesis

> **Abstract:** *Controllable text-to-image generation in medical imaging remains challenging due to the ambiguity and spatial imprecision of nat-
ural language. We propose a latent-aligned diffusion framework for chest X-ray (CXR) synthesis that integrates semantic prompts
with spatially precise conditioning across anatomical and pathological structures. The model learns a structured latent space that en-
codes organ positions, lesion attributes, and appearance features, which are decoded into spatial maps guiding a diffusion generator.
This enables versatile pipelines, text-to-image, image-to-image, and latent-to-image, with fine-grained anatomical and pathologi-
cal control. On the MIMIC-CXR dataset, our method achieves state-of-the-art performance (MS-SSIM 0.3298, FID 56.78, Dice
0.6536). For downstream diagnosis, accuracy improves by 17.5% when augmenting 1k real images with synthetic data. These re-
sults highlight the ability of our framework to generate clinically realistic CXRs with precise controllability, enabling applications
in data augmentation, model evaluation, and counterfactual analysis.* 

## Inference

The entire codes for training pipeline will be added soon. This page provide inference of the model.

## Overview

## Setup

1. Clone the repository:

```bash
git clone https://github.com/nubcico/XrayGen.git
cd XrayGen
python inference.py

## References

This work builds upon the following repositories:

- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [stable-diffusion](https://github.com/CompVis/stable-diffusion/)
