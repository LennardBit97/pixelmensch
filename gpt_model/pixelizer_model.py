import base64
import os
from util.image_operations import load_and_resize
from dotenv import load_dotenv
from openai import OpenAI


class Pixelizer:
    def __init__(
        self,
        ref_dir="input",
        ref_prefix="ref",
        ref_count=7,
        model="gpt-image-1",
        quality="auto",
        size="1024x1536",
    ):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.quality = quality
        self.size = size
        self.ref_images = []
        for i in range(ref_count):
            image_path = f"{ref_dir}/{ref_prefix}{i + 1}.png"
            self.ref_images.append(load_and_resize(image_path))
        self.prompt = (
            "Convert the person from the target image into the visual style of the reference images.\n\n"
            "Style description (from ref images):\n"
            "Pixel art format, the person is exactly 32 large pixels wide × 64 large pixels high\n"
            "Blocky, unshaded pixels, no gradients or textures\n"
            "Neutral pose: frontal view, shoulder‑width stance, arms hanging straight down, hands in front of body\n"
            "Light grey background (#d3d3d3), solid color\n"
            "Facial features stylized: eyes as short lines or dots, simplified mouth, blocky nose\n"
            "Clothing and hair simplified to flat colored shapes\n"
            "No outlines, no detail shading, everything flat and geometric\n\n"
            "Apply this style to the person in the target image:\n"
            "Retain their clothing type, hair color/style, facial structure, facial expression, mouth expression, eye color, and accessories\n"
            "Convert everything into the described pixel style\n"
            "Adjust posture and background to match ref images exactly"
        )

    def pixelize(self, target_image, output_path="output.png"):
        """
        Pixelizes the target image using the reference images and prompt.
        :param target_image: loaded image with size <= 1024p.
        :param output_path: path to save the pixelized image.
        :return: bytes of the pixelized image.
        """
        all_images = self.ref_images + [target_image]
        result = self.client.images.edit(
            model=self.model,
            image=all_images,
            prompt=self.prompt,
            quality=self.quality,
            size=self.size,
        )
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        with open(output_path, "wb") as f:
            f.write(image_bytes)
        return image_bytes
