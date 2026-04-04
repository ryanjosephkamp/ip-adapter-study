# GA18C.py CS6078 cheng 2026
# Chapter 8 of Hands-on GenAI
# Usage: python GA18C.py

import torch
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
)

pipeline.load_ip_adapter(
    "h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin"
)

scale = {"up": {"block_0": [0.0, 1.0, 0.0]}}
pipeline.set_ip_adapter_scale(scale)

import matplotlib.pyplot as plt
from genaibook.core import SampleURL, load_image
image = load_image(SampleURL.Mamoeiro)
original_image = image.resize((1024, 1024))
plt.imshow(original_image)
plt.xticks([])
plt.yticks([])
plt.show()

variation_image = pipeline(
    prompt="a cat inside of a box",
    ip_adapter_image=original_image,
    num_inference_steps=25,
).images[0]
plt.imshow(variation_image)
plt.xticks([])
plt.yticks([])
plt.show()

