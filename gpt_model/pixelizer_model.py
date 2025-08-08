import base64
import os
from util.image_operations import load_and_resize, concatenate_images
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI


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
        self.client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version="2025-04-01-preview",
            azure_endpoint="https://cidd-aifoundry-pl.openai.azure.com",
        )
        self.model = model
        self.quality = quality
        self.size = size
        self.ref_images = []
        for i in range(ref_count):
            image_path = f"{ref_dir}/{ref_prefix}{i + 1}.png"
            self.ref_images.append(load_and_resize(image_path))

        self.prompt = f"""In the image, {ref_count} pixel characters appear next to a real person.
            Convert the real person from the target image into the visual style of the pixel reference images.
            Style description, transfer the style of the reference images as much as possible:
            Pixel art format, the person is exactly 40 large pixels wide × 80 large pixels high
            Use half pixels for details like slight facial hair and eyebrows.
            Blocky, unshaded pixels, no gradients or textures
            Light grey background (#d3d3d3), solid color
            No outlines, no detail shading, everything flat and geometric
            Neutral pose: frontal view, shoulder‑width stance, arms hanging straight down, with thumbs in front od the body, feet parallel
            Facial features stylized, no noses: eyes as short colored lines or colored dots, slight eyebrows,tiny gap between eyes and eyebrows, simplified mouth, NO nose, flat shaded like the reference images
            Clothing and hair slightly simplified to flat colored shapes
            
            Apply this style to the person in the target image:
            Retain their clothing type, clothing details, hair color/style, facial structure, facial expression, mouth expression, eye color, body type, and accessories
            Convert everything into the described pixel style
            Output must contain only one person in the center of the image
            Adjust posture and background to match ref images exactly
            """

    def pixelize(self, target_image, output_path="output.png"):
        """
        Pixelizes the target image using the reference images and prompt.
        :param target_image: loaded image with size <= 1024p.
        :param output_path: path to save the pixelized image.
        :return: bytes of the pixelized image.
        """
        target_image.name = "target.png"
        all_images = [target_image] + self.ref_images
        concat_images = concatenate_images(all_images)
        concat_images.seek(0)

        # TODO debug entry
        with open("test.png", "wb") as f:
            f.write(concat_images.read())

        stream = self.client.images.edit(
            model=self.model,
            image=concat_images,
            prompt=self.prompt,
            quality=self.quality,
            size=self.size,
            stream=True,
            partial_images=3,
        )

        for event in stream:
            print(f"Event: {event.type}")
            # print([attr for attr in dir(event) if not attr.startswith("__")])
            if event.type == "image_edit.completed":
                with open("event_log.txt", "w") as f:
                    f.write("Event attributes:\n")
                    f.write(
                        "\n".join(
                            [
                                f"{attr}: {getattr(event, attr)}"
                                for attr in dir(event)
                                if not attr.startswith("__")
                            ]
                        )
                    )
            # if event.type == "image_edit.partial_image":
            image_base64 = event.b64_json
            image_bytes = base64.b64decode(image_base64)
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            yield image_bytes

        # image_base64 = result.data[0].b64_json
        # image_bytes = base64.b64decode(image_base64)
        # with open(output_path, "wb") as f:
        #     f.write(image_bytes)
        # return image_bytes
