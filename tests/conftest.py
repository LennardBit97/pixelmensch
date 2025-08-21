import sys
from pathlib import Path

# Repo-Root auf sys.path legen (eine Ebene über test_py/)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# shared fixtures & utilities for tests
import io
from PIL import Image
import pytest


@pytest.fixture
def tiny_rgba_image():
    img = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
    return img


@pytest.fixture
def tiny_png_bytes(tiny_rgba_image):
    buf = io.BytesIO()
    tiny_rgba_image.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture
def warnings_sink(monkeypatch):
    """
    Fängt gr.Warning / gr.Error ab, damit wir sie prüfen können,
    ohne echte UI‑Toasts zu erzeugen.
    """
    import gradio as gr

    msgs = []
    monkeypatch.setattr(
        gr, "Warning", lambda m: msgs.append(("warning", str(m))), raising=False
    )
    monkeypatch.setattr(
        gr, "Error", lambda m: msgs.append(("error", str(m))), raising=False
    )
    return msgs
