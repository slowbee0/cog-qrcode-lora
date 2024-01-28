from diffusers import AutoencoderKL, ControlNetModel
from diffusers import StableDiffusionControlNetPipeline

CACHE_DIR = "./hf_cache"

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
controlnet = ControlNetModel.from_pretrained(
    "monster-labs/control_v1p_sd15_qrcode_monster",
    cache_dir=CACHE_DIR
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    vae=vae,
    safety_checker=None,
    cache_dir=CACHE_DIR
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "prompthero/openjourney",
    controlnet=controlnet,
    vae=vae,
    safety_checker=None,
)
