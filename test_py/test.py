from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import base64
import os
import config
import litellm

# Load environment variables from .env file
load_dotenv()
LLM_KEY = os.getenv("LLM_KEY")

litellm.api_base = config.LITELLM_API_BASE
litellm.api_key = LLM_KEY
os.environ["OPENAI_API_KEY"] = LLM_KEY
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# def create_file(file_path):
#     with open(file_path, "rb") as file_content:
#         result = client.files.create(file=file_content, purpose="vision")
#     return result.id


def encode_image(file_path):
    with open(file_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    return base64_image


def encode_image_resized(path, size=(128, 256)):
    with Image.open(path) as img:
        img = img.convert("RGB").resize(size)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"


prompt = (
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
)


# github files
files = [
    "./input/ref1.png",
    "./input/ref2.png",
    "./input/ref3.png",
    "./input/ref4.png",
    "./input/ref5.png",
    "./input/ref6.png",
    "./input/ref7.png",
    "./input/target.png",
]
# base64_imgs = [encode_image(p) for p in files]
base64_imgs = [encode_image_resized(p) for p in files]
# # file_ids = [create_file(p) for p in files]

# ACHTUNG: In Responses API ist file_id in input_image NICHT unterstützt – nur image_url
content = [{"type": "text", "text": prompt}]

# Optional: Base64 Data-URI (funktioniert in Responses API)
# for file in files:
#     content.append({"type": "image_url", "image_url": f"{file}"})

for b64 in base64_imgs:
    content.append({"type": "image_url", "image_url": {"url": b64}})
# for fid in file_ids:
#     content.append(
#         {"type": "image_file", "image_file": {"file_id": fid, "detail": "low"}}
#     )
response = litellm.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": content}],
    tools=[
        {
            "type": "image_generation",
            "image_generation": {
                "size": "128x256",
                "quality": "standard",
            },
        }
    ],
    tool_choice="auto",
)

image_generation_calls = [
    out for out in response.output if out.type == "image_generation_call"
]
if image_generation_calls:
    img_b64 = image_generation_calls[0].result
    img_bytes = base64.b64decode(img_b64)
    with open("output.png", "wb") as f:
        f.write(img_bytes)
else:
    print("No image output", response)
