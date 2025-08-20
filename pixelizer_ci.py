"""
Pixelizer application adapted to Cologne Intelligence corporate design.

This version modifies the original Pixelizer UI to align with the light, clean
corporate style used on Cologne‑Intelligence’s website.  The colour palette
is derived from their logo: a neutral grey tone (#5F575A) and a bright
yellow accent (#FFE900) are used for text and primary actions, while panels
rest on a white background (#FFFFFF) with subtle grey borders (#E6E6E6)【33911874793402†L6-L8】.  The
application’s header includes the company’s logo (sourced from their public
favicon) and the Pixelizer title.  Otherwise, the functionality remains
unchanged.

To run this app, install the required dependencies (gradio, Pillow and
pixelizer_model) and execute the script.  When launched, open the provided
URL in your browser to interact with the Pixelizer.
"""

import uuid
from pathlib import Path
from PIL import Image
import io

import gradio as gr
from gpt_model.pixelizer_model import Pixelizer
from util.image_operations import load_and_resize

# Instantiate Pixelizer with the same settings as the original app.
pixelizer = Pixelizer(ref_count=7, quality="medium")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def process_image(image_file):
    """Generate a pixelized version of an uploaded image.

    The incoming file is read as RGBA, resized using util.image_operations
    and passed through the pixelizer model.  The function yields PIL Images
    incrementally, allowing Gradio to update the UI as pixels are processed.
    """
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


# Corporate colour definitions derived from Cologne‑Intelligence branding
CI_TEXT = "#5F575A"  # neutral dark grey for copy and headings【33911874793402†L6-L8】
CI_ACCENT = "#FFE900"  # bright yellow accent colour matching the bracket in the logo【33911874793402†L6-L8】
CI_BG = "#FFFFFF"  # white background for panels and overall page
CI_BORDER = "#E6E6E6"  # light grey borders separating panels

# Define a light Gradio theme reflecting CI’s look & feel
ci_theme = gr.themes.Base(primary_hue="yellow", secondary_hue="slate").set(
    body_background_fill=CI_BG,
    body_text_color=CI_TEXT,
    button_primary_background_fill=CI_ACCENT,
    button_primary_text_color=CI_TEXT,
    button_secondary_background_fill=CI_BG,
    button_secondary_text_color=CI_TEXT,
    input_background_fill=CI_BG,
    input_border_color=CI_BORDER,
)

# Custom CSS to mirror Cologne‑Intelligence’s clean layouts
CSS = f"""
html, body {{ margin:0; padding:0; font-family: sans-serif; }}
.gradio-container {{ width:1920px !important; height:1024px !important; margin:0 auto; overflow:hidden; background:{CI_BG}; color:{CI_TEXT}; }}
main {{ padding-top:0 !important; }}
/* Header section */
#hdr {{ height:60px; display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid {CI_BORDER}; padding:0 24px; }}
#hdr .logo-wrapper {{ display:flex; align-items:center; gap:12px; }}
#hdr .logo-wrapper img {{ height:40px; width:40px; }}
#hdr h1 {{ margin:0; font-size:28px; color:{CI_TEXT}; }}

/* Stage grid */
#stage {{ height:800px; display:grid; grid-template-columns: 1fr 1fr; gap:10px; padding:2px 24px; }}

/* Panels */
.panel {{ background:{CI_BG}; border:1px solid {CI_BORDER}; border-radius:16px; padding:14px; display:flex; flex-direction:column; }}
.panel h3 {{ margin:0 0 10px 0; font-weight:600; color:{CI_TEXT}; font-size:16px; }}

.equal {{ 
    flex:1; 
    display:flex; 
    align-items:center; 
    justify-content:center; 
    width:460px;
    margin:auto;
}}

/* Fix the image areas to a constant size, even when empty */
#img_in .gradio-image,
#img_out .gradio-image {{ width:466px !important; height:700px !important; }}

/* Always fit contents inside image containers */
#img_in img, #img_in canvas,
#img_out img, #img_out canvas {{
  width:100% !important;
  height:100% !important;
  object-fit: contain !important;
}}

/* Hide the default Gradio footer */
footer {{ display:none !important; }}

/* Override primary and secondary button styles */
button.bg-primary {{ background:{CI_ACCENT} !important; color:{CI_TEXT} !important; border:none !important; }}
button.bg-secondary {{ background:{CI_BG} !important; color:{CI_TEXT} !important; border:1px solid {CI_BORDER} !important; }}
"""

# Build the Gradio interface
with gr.Blocks(theme=ci_theme, css=CSS, analytics_enabled=False) as pixelator:
    # Header: CI logo and application title
    with gr.Row(elem_id="hdr"):
        with gr.Column(scale=0, elem_classes=["logo-wrapper"]):
            # Use the public favicon as the logo.  If an alternative PNG is available,
            # replace the src attribute accordingly.
            gr.HTML(
                f'<div class="logo-wrapper"><img alt="CI Logo" src="https://www.cologne-intelligence.de/frontend/favicons/apple-touch-icon.png" />'
                f"<h1>Pixelizer</h1></div>"
            )
    # Stage: input, output and controls
    with gr.Row(elem_id="stage"):
        # INPUT panel
        with gr.Column(elem_classes=["panel"]):
            gr.HTML("<h3>Originalbild</h3>")
            with gr.Row(elem_classes=["equal"]):
                orig_display = gr.Image(
                    elem_id="img_in",
                    type="filepath",
                    interactive=True,
                    sources=["upload", "clipboard", "webcam"],
                    height=700,
                    width=466,
                )

        # OUTPUT panel
        with gr.Column(elem_classes=["panel"]):
            gr.HTML("<h3>Pixelbild</h3>")
            with gr.Row(elem_classes=["equal"]):
                pixel_display = gr.Image(
                    elem_id="img_out",
                    interactive=False,
                    height=700,
                    width=466,
                )

    # Controls row beneath the stage (outside of the grid)
    with gr.Row(elem_id="ctrls", elem_classes=["panel"]):
        pixelize_btn = gr.Button("Pixelize", variant="primary")
        reset_btn = gr.Button("Reset", variant="secondary")

    # Bind actions
    pixelize_btn.click(fn=process_image, inputs=orig_display, outputs=pixel_display)
    reset_btn.click(fn=lambda: (None, None), outputs=[orig_display, pixel_display])

# Launch the application when run directly
if __name__ == "__main__":
    pixelator.launch(share=False, server_name="0.0.0.0", server_port=7860)
