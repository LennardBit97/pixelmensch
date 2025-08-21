import io
from pathlib import Path
import os
import pytest
from PIL import Image

# Passe den Modulnamen hier an:
import pixelizer_ci as mod

# inline-snapshot API
from inline_snapshot import snapshot

# ---------- Hilfsfunktionen ----------


def test_is_pathlike_image_true(tmp_path):
    p = tmp_path / "ok.png"
    Image.new("RGBA", (2, 2), (0, 0, 0, 0)).save(p, format="PNG")
    assert mod._is_pathlike_image(str(p)) == snapshot(True)


def test_is_pathlike_image_false(tmp_path):
    p = tmp_path / "not_image.txt"
    p.write_text("hello")
    assert mod._is_pathlike_image(str(p)) == snapshot(False)


def test_safe_open_image_as_rgba_ok(tmp_path):
    p = tmp_path / "img.webp"
    Image.new("RGB", (3, 4), (10, 20, 30)).save(p, format="WEBP")
    img = mod._safe_open_image_as_rgba(str(p))
    assert (img.mode, img.size) == snapshot(("RGBA", (3, 4)))


def test_safe_open_image_as_rgba_corrupt(tmp_path):
    p = tmp_path / "bad.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x00BAD")
    with pytest.raises(ValueError) as exc:
        mod._safe_open_image_as_rgba(str(p))
    msg = str(exc.value)
    # Statt snapshot([...]) -> Liste: stabiler Boolean-Snapshot
    assert (
        ("kein gültiges Bild" in msg) or ("konnte nicht gelesen" in msg)
    ) == snapshot(True)


def test_prepare_resized_png_bytes_roundtrip(tiny_rgba_image):
    buf = mod._prepare_resized_png_bytes(tiny_rgba_image)
    head = list(buf.getvalue()[:8])
    # Nur die PNG-Signatur snappen (stabil), Größe NICHT!
    assert head == snapshot([137, 80, 78, 71, 13, 10, 26, 10])
    # Roundtrip zurück in PIL, Größe/Mode klassisch prüfen
    img2 = Image.open(io.BytesIO(buf.getvalue()))
    assert img2.size == tiny_rgba_image.size
    assert img2.mode in ("RGBA", "RGB", "LA", "L")


def test_safe_save_bytes_to_rgba_image_ok(tiny_png_bytes):
    img = mod._safe_save_bytes_to_rgba_image(tiny_png_bytes)
    assert (img.mode, img.size) == snapshot(("RGBA", (10, 10)))


def test_safe_save_bytes_to_rgba_image_invalid():
    with pytest.raises(ValueError) as exc:
        mod._safe_save_bytes_to_rgba_image(b"")
    msg = str(exc.value)
    assert any(s in msg for s in ("A", "B")) == snapshot(True)


def test_safe_reset():
    assert mod.safe_reset() == snapshot((None, None))


# ---------- process_image (mit Mocks) ----------


def test_process_image_none_input_yields_none_and_warns(warnings_sink):
    out = list(mod.process_image(None))
    assert out == snapshot([None])
    # mind. eine Warning
    assert any(k == "warning" for k, _ in warnings_sink) == snapshot(True)


def test_process_image_valid_flow(
    tmp_path, monkeypatch, tiny_rgba_image, warnings_sink
):
    src = tmp_path / "in.jpg"
    tiny_rgba_image.convert("RGB").save(src, format="JPEG")

    def fake_load_and_resize(buf):
        Image.open(io.BytesIO(buf.getvalue()))  # Validierung
        return tiny_rgba_image

    monkeypatch.setattr(mod, "load_and_resize", fake_load_and_resize, raising=True)

    class FakePixelizer:
        def pixelize(self, pil_img, output_path=None):
            frames = []
            for color in [(255, 0, 0, 255), (0, 255, 0, 255)]:
                im = Image.new("RGBA", (10, 10), color)
                b = io.BytesIO()
                im.save(b, format="PNG")
                frames.append(b.getvalue())
            for f in frames:
                yield f

    monkeypatch.setattr(mod, "pixelizer", FakePixelizer(), raising=True)

    results = list(mod.process_image(str(src)))
    # zwei Bilder, beide 10x10
    assert len(results) == snapshot(2)
    assert [im.size for im in results] == snapshot([(10, 10), (10, 10)])
    # keine Errors
    assert any(k == "error" for k, _ in warnings_sink) == snapshot(False)


def test_process_image_no_yield_from_model(
    tmp_path, monkeypatch, tiny_rgba_image, warnings_sink
):
    src = tmp_path / "in.png"
    tiny_rgba_image.save(src, format="PNG")

    monkeypatch.setattr(
        mod, "load_and_resize", lambda buf: tiny_rgba_image, raising=True
    )

    class FakePixelizerEmpty:
        def pixelize(self, pil_img, output_path=None):
            if False:
                yield b""

    monkeypatch.setattr(mod, "pixelizer", FakePixelizerEmpty(), raising=True)

    out = list(mod.process_image(str(src)))
    assert out == snapshot([None])
    # Fehlermeldung wurde ausgegeben
    assert any(
        k == "error" and "keine Ausgabe" in m for k, m in warnings_sink
    ) == snapshot(True)


def test_process_image_bad_chunk_then_good(
    tmp_path, monkeypatch, tiny_rgba_image, warnings_sink
):
    src = tmp_path / "in.png"
    tiny_rgba_image.save(src, format="PNG")

    monkeypatch.setattr(
        mod, "load_and_resize", lambda buf: tiny_rgba_image, raising=True
    )

    class FakePixelizerMixed:
        def pixelize(self, pil_img, output_path=None):
            yield b"not-a-png"  # kaputt
            b2 = io.BytesIO()
            Image.new("RGBA", (10, 10), (0, 0, 255, 255)).save(b2, format="PNG")
            yield b2.getvalue()

    monkeypatch.setattr(mod, "pixelizer", FakePixelizerMixed(), raising=True)

    out = list(mod.process_image(str(src)))
    # am Ende genau ein valides Bild
    assert len(out) == snapshot(1)
    assert isinstance(out[0], Image.Image) == snapshot(True)
    # es gab eine Warning für den schlechten Chunk
    assert any(
        k == "warning" and "Zwischenschritt" in m for k, m in warnings_sink
    ) == snapshot(True)


def test_process_image_model_not_initialized(
    tmp_path, monkeypatch, tiny_rgba_image, warnings_sink
):
    src = tmp_path / "in.png"
    tiny_rgba_image.save(src, format="PNG")

    monkeypatch.setattr(
        mod, "load_and_resize", lambda buf: tiny_rgba_image, raising=True
    )
    monkeypatch.setattr(mod, "pixelizer", None, raising=True)

    out = list(mod.process_image(str(src)))
    assert out == snapshot([None])
    assert any(
        k == "error" and "nicht initialisiert" in m for k, m in warnings_sink
    ) == snapshot(True)
