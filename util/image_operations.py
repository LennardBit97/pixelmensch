from PIL import Image
import io


def load_and_resize(image_input, max_width=400, max_height=765):
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
    img.thumbnail((max_width, max_height), Image.LANCZOS)

    # In korrektes PNG schreiben
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)

    # Setze notwendige Attribute für OpenAI Upload
    buf.name = f"{image_input}"
    buf.content_type = "image/png"

    return buf


def concatenate_images(images, direction="horizontal"):
    """
    Concatenate multiple PIL images either horizontally or vertically.

    Args:
        images: List of PIL Images
        direction: "horizontal" or "vertical"

    Returns:
        PIL Image: Concatenated image
    """
    if not images:
        return None

    # Filter out None images
    valid_images = [Image.open(img) for img in images if img is not None]

    if not valid_images:
        return None

    if len(valid_images) == 1:
        return valid_images[0].convert("RGB")

    # Convert all images to RGB
    valid_images = [img.convert("RGB") for img in valid_images]

    if direction == "horizontal":
        # Calculate total width and max height
        total_width = sum(img.width for img in valid_images)
        max_height = max(img.height for img in valid_images)

        # Create new image
        concatenated = Image.new("RGB", (total_width, max_height), (255, 255, 255))

        # Paste images
        x_offset = 0
        for img in valid_images:
            # Center image vertically if heights differ
            y_offset = (max_height - img.height) // 2
            concatenated.paste(img, (x_offset, y_offset))
            x_offset += img.width

    else:  # vertical
        # Calculate max width and total height
        max_width = max(img.width for img in valid_images)
        total_height = sum(img.height for img in valid_images)

        # Create new image
        concatenated = Image.new("RGB", (max_width, total_height), (255, 255, 255))

    # In korrektes PNG schreiben
    buf = io.BytesIO()
    concatenated.save(buf, format="PNG")
    buf.seek(0)

    # Setze notwendige Attribute für OpenAI Upload
    buf.name = "input.png"
    buf.content_type = "image/png"

    return buf
