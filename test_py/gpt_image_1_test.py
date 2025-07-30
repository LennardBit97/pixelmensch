import base64
from openai import OpenAI
import os
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


prompt = """
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
    "Retain their clothing type, hair color/style, facial structure, facial expression, eye color, and accessories\n"
    "Convert everything into the described pixel style\n"
    "Adjust posture and background to match ref images exactly"
"""


def load_and_resize(image_path, max_size=1024):
    with open(image_path, "rb") as f:
        img = Image.open(f)
        img = img.convert("RGBA")  # falls Transparenz vorhanden ist
        img.thumbnail((max_size, max_size), Image.LANCZOS)  # skaliert proportional
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf


# Beispiel:
img_ref1 = load_and_resize("input/ref1.png")
img_ref2 = load_and_resize("input/ref2.png")
img_ref3 = load_and_resize("input/ref3.png")
img_ref4 = load_and_resize("input/ref4.png")
img_ref5 = load_and_resize("input/ref5.png")
img_ref6 = load_and_resize("input/ref6.png")
img_ref7 = load_and_resize("input/ref7.png")
target = load_and_resize("input/target2.jpeg")

# img_ref1 = open("input/ref1.png", "rb")
# img_ref2 = open("input/ref2.png", "rb")
# img_ref3 = open("input/ref3.png", "rb")
# img_ref4 = open("input/ref4.png", "rb")
# img_ref5 = open("input/ref5.png", "rb")
# img_ref6 = open("input/ref6.png", "rb")
# img_ref7 = open("input/ref7.png", "rb")
# target = open("input/target2.jpeg", "rb")

result = client.images.edit(
    model="gpt-image-1",
    image=[
        img_ref1,
        img_ref2,
        img_ref3,
        img_ref4,
        img_ref5,
        img_ref6,
        img_ref7,
        target,
    ],
    prompt=prompt,
    quality="low",
)

image_base64 = result.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

# Save the image to a file
with open("output.png", "wb") as f:
    f.write(image_bytes)
