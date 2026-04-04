# GA18B.py CS6078 cheng 2026
# Chapter 8 of Hands-on GenAI
# Usage: python GA18B.py

import torch
from diffusers import StableDiffusionXLPipeline

sdxl_base_pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
)

sdxl_base_pipeline.load_ip_adapter(
    "h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin"
)

sdxl_base_pipeline.set_ip_adapter_scale(0.8)

import matplotlib.pyplot as plt
from genaibook.core import SampleURL, load_image
image = load_image(SampleURL.ItemsVariation)
original_image = image.resize((1024, 1024))
plt.imshow(original_image)
plt.xticks([])
plt.yticks([])
plt.show()

variation_image = sdxl_base_pipeline(
    prompt="",
    ip_adapter_image=original_image,
    num_inference_steps=25,
).images[0]
plt.imshow(variation_image)
plt.xticks([])
plt.yticks([])
plt.show()
