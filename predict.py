from typing import List
import os

from PIL.Image import LANCZOS
from PIL import Image
import qrcode
import torch
from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionControlNetPipeline, EulerDiscreteScheduler, ControlNetModel

CACHE_DIR = "hf_cache"
lora_model_dict = {
    "beasts cat": os.path.join(CACHE_DIR, "幻兽-哥斯喵咔系列-极氪白 _幻兽-哥斯喵咔V2.safetensors"),
    "ink and wash": os.path.join(CACHE_DIR, "国风山水绘卷_Guofeng Shanshui Emaki_V1.0.safetensors"),
    "3ddianshang": os.path.join(CACHE_DIR, "3D电商模型_v1.1.safetensors")
}


def resize_for_condition_image(input_image, width, height):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(min(width, height)) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=LANCZOS)
    return img


class Predictor(BasePredictor):
    def setup(self, model_id: str = 'runwayml/stable-diffusion-v1-5'):
        self.meta = {
            "model_id": model_id,
            "fused_lora": False,
            "load_lora": False
        }
        """Load the model into memory to make running multiple predictions efficient"""
        self.controlnet = ControlNetModel.from_pretrained(
            "monster-labs/control_v1p_sd15_qrcode_monster", torch_dtype=torch.float16, cache_dir=CACHE_DIR
        )
        # torch.backends.cuda.matmul.allow_tf32 = True
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            controlnet=self.controlnet,
            cache_dir=CACHE_DIR
        ).to("cuda")
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.enable_xformers_memory_efficient_attention()

    def generate_qrcode(self, qr_code_content, background, border, width, height):
        print("Generating QR Code from content")
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=border,
        )
        qr.add_data(qr_code_content)
        qr.make(fit=True)

        qrcode_image = qr.make_image(fill_color="black", back_color=background)
        qrcode_image = resize_for_condition_image(qrcode_image, width, height)
        return qrcode_image

    # Define the arguments and types the model takes as input
    def predict(
            self,
            base_model: str = Input(
                description="SD Base Model",
                choices=["runwayml/stable-diffusion-v1-5", "prompthero/openjourney"],
                default="runwayml/stable-diffusion-v1-5"),
            adapt_first_lora: str = Input(
                description="you can select the first lora model",
                choices=["None", "beasts cat", "ink and wash", "3ddianshang"],
                default="None",
            ),
            lora_1_weight: float = Input(
                description="weight of the lora",
                default=0.5,
                ge=0.1,
                le=1.0,
            ),
            adapt_second_lora: str = Input(
                description="you can select the second lora model",
                choices=["None", "beasts cat", "ink and wash", "3ddianshang"],
                default="None",
            ),
            lora_2_weight: float = Input(
                description="weight of the lora",
                default=0.5,
                ge=0.1,
                le=1.0,
            ),
            prompt: str = Input(description="The prompt to guide QR Code generation."),
            qr_code_content: str = Input(
                description="The website/content your QR Code will point to."
            ),
            negative_prompt: str = Input(
                description="The negative prompt to guide image generation.",
                default="ugly, disfigured, low quality, blurry, nsfw",
            ),
            num_inference_steps: int = Input(
                description="Number of diffusion steps", ge=20, le=100, default=40
            ),
            guidance_scale: float = Input(
                description="Scale for classifier-free guidance",
                default=7.5,
                ge=0.1,
                le=30.0,
            ),
            seed: int = Input(description="Seed", default=-1),
            width: int = Input(description="Width out the output image", default=768),
            height: int = Input(description="Height out the output image", default=768),
            num_outputs: int = Input(
                description="Number of outputs", ge=1, le=4, default=1
            ),
            image: Path = Input(
                description="Input image. If none is provided, a QR code will be generated",
                default=None,
            ),
            controlnet_conditioning_scale: float = Input(
                description="The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added to the residual in the original unet.",
                ge=0.0,
                le=4.0,
                default=2.2,
            ),
            border: int = Input(description="QR code border size", ge=0, le=4, default=1),
            qrcode_background: str = Input(
                description="Background color of raw QR code",
                choices=["gray", "white"],
                default="gray",
            ),
    ) -> List[Path]:
        if base_model != self.meta.get("model_id"):
            self.setup(model_id=base_model)
            self.meta["model_id"] = base_model

        # todo: do not reload for same params
        if self.meta["fused_lora"]:
            self.setup(model_id=base_model)

        lora_scale = 1.0

        lora_status = [adapt_first_lora, adapt_second_lora]
        lora_num = lora_status.count("None")
        if lora_num == 2:
            if self.meta["load_lora"]:
                self.pipe.unload_lora_weights()
        elif lora_num == 1:
            # todo: do not reload for same params
            if adapt_first_lora != "None":
                self.pipe.load_lora_weights(lora_model_dict[adapt_first_lora])
                lora_scale = lora_1_weight
            else:
                self.pipe.load_lora_weights(lora_model_dict[adapt_second_lora])
                lora_scale = lora_2_weight
            self.meta["load_lora"] = True
        else:
            self.pipe.load_lora_weights(lora_model_dict[adapt_first_lora])
            self.pipe.fuse_lora(lora_scale=lora_1_weight)

            self.pipe.load_lora_weights(lora_model_dict[adapt_second_lora])
            self.pipe.fuse_lora(lora_scale=lora_2_weight)
            self.meta["fused_lora"] = True

        seed = torch.randint(0, 2 ** 32, (1,)).item() if seed == -1 else seed
        if image is None:
            if qrcode_background == "gray":
                qrcode_background = "#808080"
            image = self.generate_qrcode(
                qr_code_content, background=qrcode_background, border=border, width=width, height=height,
            )
        else:
            image = Image.open(str(image))
        out = self.pipe(
            prompt=[prompt] * num_outputs,
            negative_prompt=[negative_prompt] * num_outputs,
            image=[image] * num_outputs,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=torch.Generator().manual_seed(seed),
            num_inference_steps=num_inference_steps,
            cross_attention_kwargs={"scale": lora_scale}
        )

        outputs = []
        for i, image in enumerate(out.images):
            fname = f"output-{i}.png"
            image.save(fname)
            outputs.append(Path(fname))

        return outputs
