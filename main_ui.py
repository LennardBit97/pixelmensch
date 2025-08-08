import uuid
from pathlib import Path
from PIL import Image
import io

import gradio as gr
from gpt_model.pixelizer_model import Pixelizer
from util.image_operations import load_and_resize

# --- Pixelizer unverändert ---
pixelizer = Pixelizer(ref_count=7, quality="medium")

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

    for image_bytes in pixelizer.pixelize(resized, output_path=str(output_path)):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        yield img


# --------------------------------

dark_theme = gr.themes.Base(primary_hue="indigo", secondary_hue="slate").set(
    body_background_fill="#0b0f19",
    body_text_color="#e5e7eb",
    button_primary_background_fill="#5F575A",
    button_primary_text_color="#FFE900",
    input_background_fill="#101826",
    input_border_color="#253146",
)

ci_text = "#5F575A"
ci_accent = "#FFE900"
ci_bg = "#FFFFFF"
ci_border = "#E6E6E6"

# ... (alles wie bei dir oben)

CSS = """
html, body { margin:0; padding:0; }
.gradio-container { width:1920px !important; height:1024px !important; margin:0 auto; overflow:hidden; }

/* Bereiche */
#hdr { height:60px; display:flex; align-items:center; justify-content:center; border-bottom:1px solid #1f2a3a; }
#ctrls { height:60px; display:flex; align-items:center; justify-content:center; gap:10px; border-bottom:1px solid #1f2a3a; }
#stage { height:800px; display:grid; grid-template-columns: 1fr 1fr; gap:10px; padding:2px 24px; }

/* Panels */
.panel { background:#FFFFFF; border:1px solid #E6E6E6; border-radius:16px; padding:14px; display:flex; flex-direction:column; }
.panel h3 { margin:0 0 10px 0; font-weight:600; color:#5F575A; }

.equal { 
    flex:1; 
    display:flex; 
    align-items:center; 
    justify-content:center; 
    width: 460px;
    margin: auto;
}

/* Fixe Bildfläche 466x700 – auch wenn leer */
#img_in .gradio-image,
#img_out .gradio-image { width:466px !important; height:700px !important; }

/* Inhalt stets einpassen, niemals größer werden */
#img_in img, #img_in canvas,
#img_out img, #img_out canvas {
  width:100% !important;
  height:100% !important;
  object-fit: contain !important;
}

/* Footer ausblenden */
footer { display:none !important; }
"""

with gr.Blocks(theme=dark_theme, css=CSS, analytics_enabled=False) as pixelator:
    with gr.Row(elem_id="hdr"):
        with gr.Column(scale=1, elem_classes=["brand-title"]):
            gr.Markdown("<h1>🧱 Pixelizer</h1>")
        # with gr.Column(scale=0, elem_classes=["brand-logo"]):
        #     gr.HTML(
        #         '<img alt="Cologne Intelligence" '
        #         'src="https://media.licdn.com/dms/image/v2/C560BAQGF4SMQKTBqtg/company-logo_200_200/company-logo_200_200/0/1631309925885?e=2147483647&v=beta&t=eoyxGRc88wIeSZIx65g9tq24C9GNbK06bGRcYIQk1_E" />'
        #     )
    # Stage
    with gr.Row(elem_id="stage"):
        # INPUT
        with gr.Column(elem_classes=["panel"]):
            gr.HTML("<h3>Input</h3>")
            with gr.Row(elem_classes=["equal"]):
                orig_display = gr.Image(
                    elem_id="img_in",
                    type="filepath",
                    interactive=True,
                    sources=["upload", "clipboard", "webcam"],
                    height=700,
                    width=466,  # feste Fläche, auch leer
                )

        # OUTPUT
        with gr.Column(elem_classes=["panel"]):
            gr.HTML("<h3>Output</h3>")
            with gr.Row(elem_classes=["equal"]):
                pixel_display = gr.Image(
                    elem_id="img_out",
                    interactive=False,
                    height=700,
                    width=466,  # feste Fläche, auch leer
                )

        # Controls
        with gr.Row(elem_id="ctrls"):
            pixelize_btn = gr.Button("Pixelize", variant="primary")
            reset_btn = gr.Button("Reset", variant="secondary")

    # Wiring
    pixelize_btn.click(fn=process_image, inputs=orig_display, outputs=pixel_display)
    reset_btn.click(fn=lambda: (None, None), outputs=[orig_display, pixel_display])

if __name__ == "__main__":
    pixelator.launch(share=False, server_name="0.0.0.0", server_port=7860)
