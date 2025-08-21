"""
Pixelizer application adapted to Cologne Intelligence corporate design.
(Hardening/Resilience edition)
"""

import uuid
from pathlib import Path
from typing import Generator, Optional, Tuple
from PIL import Image, UnidentifiedImageError
import io
import os

import gradio as gr
from gpt_model.pixelizer_model import Pixelizer
from util.image_operations import load_and_resize

# Instantiate Pixelizer with the same settings as the original app.
# Defensive: Falls Konstruktor scheitert, später im Handler behandeln.
try:
    pixelizer = Pixelizer(ref_count=7, quality="medium")
except Exception as e:
    pixelizer = None  # Wird im Handler geprüft

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Resilience / Validation configuration (anpassbar) ---
ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
MAX_INPUT_BYTES = 25 * 1024 * 1024  # 25 MB, optional
MAX_OUTPUT_BYTES = 50 * 1024 * 1024  # 50 MB, optional


def _is_pathlike_image(path_str: str) -> bool:
    if not path_str or not isinstance(path_str, str):
        return False
    ext = Path(path_str).suffix.lower()
    return ext in ALLOWED_EXTS and Path(path_str).exists()


def _safe_open_image_as_rgba(path_str: str) -> Image.Image:
    """
    Öffnet Bild robust, verifiziert es und konvertiert nach RGBA.
    Raises Exception bei Problemen.
    """
    # Optional: Größe prüfen (nur bei lokalen Pfaden sinnvoll)
    try:
        if os.path.isfile(path_str):
            size = os.path.getsize(path_str)
            if size > MAX_INPUT_BYTES:
                raise ValueError(
                    f"Die Datei ist zu groß ({size // (1024 * 1024)} MB). "
                    f"Maximal erlaubt sind {MAX_INPUT_BYTES // (1024 * 1024)} MB."
                )
    except Exception:
        # Bei Zugriffsfeldern nicht hart abbrechen – wir versuchen trotzdem zu öffnen.
        pass

    try:
        with Image.open(path_str) as img_probe:
            # Korrupte Dateien früh erkennen
            img_probe.verify()
        # verify() schließt die Datei; neu öffnen zum eigentlichen Laden
        img = Image.open(path_str).convert("RGBA")
        return img
    except UnidentifiedImageError:
        raise ValueError("Die angegebene Datei ist kein gültiges Bild.")
    except OSError:
        raise ValueError("Das Bild konnte nicht gelesen werden (I/O-Fehler).")


def _safe_save_bytes_to_rgba_image(image_bytes: bytes) -> Image.Image:
    """
    Bytes -> PIL Image (RGBA) mit defensiver Prüfung.
    """
    try:
        if image_bytes is None:
            raise ValueError("Leerer Bild-Chunk vom Modell erhalten.")
        if len(image_bytes) > MAX_OUTPUT_BYTES:
            raise ValueError(
                "Generiertes Bild ist unerwartet groß. Vorgang wird abgebrochen."
            )
        bio = io.BytesIO(image_bytes)
        img = Image.open(bio).convert("RGBA")
        return img
    except UnidentifiedImageError:
        raise ValueError("Ungültige Bilddaten vom Modell erhalten.")
    except OSError:
        raise ValueError("Fehler beim Dekodieren der generierten Bilddaten.")


def _prepare_resized_png_bytes(pil_img_rgba: Image.Image) -> io.BytesIO:
    """
    Speichert ein PIL‑Bild als PNG in BytesIO (z. B. für load_and_resize).
    """
    buf = io.BytesIO()
    pil_img_rgba.save(buf, format="PNG")
    buf.seek(0)
    return buf


def process_image(
    image_file: Optional[str],
) -> Generator[Optional[Image.Image], None, None]:
    """
    Generate a pixelized version of an uploaded image.

    Defensive version:
    - Validiert input
    - Fängt Fehler in load_and_resize und im Pixelizer ab
    - Gibt Nutzer‑Meldungen über Gradio‑Toasts aus
    - Liefert bei Fehlern kein Bild (None), sodass die UI konsistent bleibt
    """
    # 1) Basic input checks
    if not image_file:
        gr.Warning("Bitte ein Bild auswählen oder hochladen.")
        yield None
        return

    if not _is_pathlike_image(image_file):
        gr.Warning("Die ausgewählte Datei scheint kein unterstütztes Bild zu sein.")
        # Versuche dennoch zu öffnen – ggf. handelt es sich um eine temporäre Webcam‑Datei ohne Endung
        # Bei Fehler bricht _safe_open_image_as_rgba mit klarer Meldung ab.
    try:
        image_rgba = _safe_open_image_as_rgba(image_file)
    except Exception as e:
        gr.Error(f"Eingabefehler: {e}")
        yield None
        return

    # 2) Resize / Pre‑process defensively
    try:
        buf_png = _prepare_resized_png_bytes(image_rgba)
        resized = load_and_resize(buf_png)
        if resized is None:
            raise ValueError("Bild konnte nicht skaliert/verarbeitet werden.")
    except Exception as e:
        gr.Error(f"Vorverarbeitung fehlgeschlagen: {e}")
        yield None
        return

    # 3) Output Pfad vorbereiten
    try:
        output_name = f"pixelized_{uuid.uuid4().hex[:8]}.png"
        output_path = OUTPUT_DIR / output_name
        # Schreibprobe optional:
        with open(output_path, "wb") as fp:
            pass
        # Datei wieder entfernen – Modell wird sie gleich befüllen.
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass
    except Exception:
        # Fallback: in RAM arbeiten, kein persistentes Ziel
        output_path = None

    # 4) Pixelizer ausführen (robust)
    if pixelizer is None:
        gr.Error("Das Pixelizer‑Modell konnte nicht initialisiert werden.")
        yield None
        return

    try:
        iterator = pixelizer.pixelize(
            resized,
            output_path=str(output_path) if output_path else None,
        )

        # Falls der Pixelizer wider Erwarten nichts liefert, Nutzer informieren
        got_any = False
        for image_bytes in iterator:
            got_any = True
            try:
                img = _safe_save_bytes_to_rgba_image(image_bytes)
            except Exception as chunk_err:
                # Einzelne fehlerhafte Chunks überspringen; weiter versuchen
                gr.Warning(f"Ein Zwischenschritt war ungültig: {chunk_err}")
                continue
            yield img

        if not got_any:
            gr.Error("Das Modell hat keine Ausgabe erzeugt.")
            yield None
            return

    except FileNotFoundError as e:
        gr.Error(f"Dateifehler während der Pixelisierung: {e}")
        yield None
        return
    except MemoryError:
        gr.Error("Nicht genügend Speicher während der Verarbeitung.")
        yield None
        return
    except Exception as e:
        gr.Error(f"Unerwarteter Fehler bei der Pixelisierung: {e}")
        yield None
        return


def safe_reset() -> Tuple[None, None]:
    """
    Defensive Reset‑Funktion, die unabhängig vom Zustand immer ein leeres UI herstellt.
    """
    try:
        # Hier könnten temporäre Dateien gelöscht werden (optional).
        pass
    except Exception:
        pass
    return None, None


# Corporate colour definitions derived from Cologne‑Intelligence branding
CI_TEXT = "#5F575A"  # neutral dark grey for copy and headings
CI_ACCENT = "#FFE900"  # bright yellow accent
CI_BG = "#FFFFFF"  # white background
CI_BORDER = "#E6E6E6"  # light grey borders

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
html, body {{ margin:0; padding:0; font-family: sans-serif; overflow:hidden; }}
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
with gr.Blocks(
    theme=ci_theme, css=CSS, analytics_enabled=False, title="CI-Pixelizer"
) as pixelator:
    # Header: CI logo and application title
    with gr.Row(elem_id="hdr"):
        with gr.Column(scale=0, elem_classes=["logo-wrapper"]):
            gr.HTML(
                '<div class="logo-wrapper">'
                '<img alt="CI Logo" src="https://www.cologne-intelligence.de/frontend/favicons/apple-touch-icon.png" />'
                "<h1>Pixelizer</h1></div>"
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

    # Bind actions (robust)
    pixelize_btn.click(fn=process_image, inputs=orig_display, outputs=pixel_display)
    reset_btn.click(fn=safe_reset, outputs=[orig_display, pixel_display])

# Launch the application when run directly
if __name__ == "__main__":
    try:
        pixelator.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=int(os.environ.get("PORT", 7860)),
            favicon_path="https://www.cologne-intelligence.de/frontend/favicons/apple-touch-icon.png",
        )
    except OSError as e:
        # Falls Port belegt ist – automatischer Fallback
        gr.Warning(f"Standardport belegt, weiche auf Port 0 aus: {e}")
        pixelator.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=0,
            favicon_path="https://www.cologne-intelligence.de/frontend/favicons/apple-touch-icon.png",
        )
