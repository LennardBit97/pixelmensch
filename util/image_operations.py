from PIL import Image
import io


def load_and_resize(image_input, max_size=1024):
    # Öffne Pfad oder BytesIO
    if isinstance(image_input, (str, bytes)):
        img = Image.open(image_input)
    else:
        img = Image.open(
            io.BytesIO(image_input.read())
            if hasattr(image_input, "read")
            else image_input
        )

    img = img.convert("RGBA")
    img.thumbnail((max_size, max_size), Image.LANCZOS)

    # In korrektes PNG schreiben
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # Setze notwendige Attribute für OpenAI Upload
    buf.name = "image.png"
    buf.content_type = "image/png"

    return buf
