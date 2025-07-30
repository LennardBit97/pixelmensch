import uuid
from pathlib import Path
from PIL import Image
import io

import gradio as gr
from gpt_model.pixelizer_model import Pixelizer
from util.image_operations import load_and_resize

# Instantiate pixelizer
pixelizer = Pixelizer()

# Output dir
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def process_image(image_file):
    image = Image.open(image_file).convert("RGBA")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    resized = load_and_resize(buf)

    output_name = f"pixelized_{uuid.uuid4().hex[:8]}.png"
    output_path = OUTPUT_DIR / output_name

    image_bytes = pixelizer.pixelize(resized, output_path=str(output_path))
    pixelized = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    return pixelized


# Theme
dark_theme = gr.themes.Base(primary_hue="indigo", secondary_hue="slate").set(
    body_background_fill="#1e1e1e",
    body_text_color="#ffffff",
    button_primary_background_fill="#3b82f6",
    button_primary_text_color="#ffffff",
    input_background_fill="#2e2e2e",
    input_border_color="#444",
)

with gr.Blocks(
    theme=dark_theme,
    css="""
        body { max-width: 1280px; margin: auto; }
        .centered-row { justify-content: center !important; }
        .gr-column { align-items: center !important; }
    """,
) as pixelator:
    gr.Markdown("""<h1 style='text-align: center;'>🧱 Pixelizer</h1>""")

    with gr.Row(elem_classes="centered-row"):
        with gr.Column():
            orig_display = gr.Image(
                label="Original",
                type="filepath",
                interactive=True,
                height=630,
                width=430,
            )
            generate_btn = gr.Button("Generate Pixelized Image")

        with gr.Column():
            pixel_display = gr.Image(
                label="Generated",
                interactive=False,
                height=630,
                width=430,
            )

    generate_btn.click(fn=process_image, inputs=[orig_display], outputs=[pixel_display])

pixelator.launch()
