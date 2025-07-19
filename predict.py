# predict.py
from cog import BasePredictor, Input, Path
from PIL import Image
import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
import base64

class Predictor(BasePredictor):
    def setup(self):
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

    def predict(
        self,
        prompt: str = Input(description="Mô tả cổ trang/fantasy"),
        image: Path = Input(description="Ảnh chân dung", default=None),
        width: int = Input(default=720),
        height: int = Input(default=1280),
        num_inference_steps: int = Input(default=30),
        guidance_scale: float = Input(default=7.5),
        seed: int = Input(default=42)
    ) -> Path:
        generator = torch.manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        output_path = "/tmp/output.png"
        image.save(output_path)
        return Path(output_path)
