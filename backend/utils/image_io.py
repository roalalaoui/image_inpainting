"""Decoding, validation and encoding helpers.

Common convention used everywhere in the backend:
  image : numpy uint8, shape (H, W, 3), RGB
  mask  : numpy uint8, shape (H, W),    values 0 or 255
          255 (white) = missing / damaged pixels, 0 (black) = valid pixels
"""
import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 2048 * 2048
ALLOWED_FORMATS = {"PNG", "JPEG"}


class InputError(ValueError):
    """The uploaded image or mask is invalid (reported to the client as HTTP 400)."""


def _open(data: bytes, label: str) -> Image.Image:
    if not data:
        raise InputError(f"The {label} file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise InputError(f"The {label} is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise InputError(f"The {label} is not a readable image.")
    if img.format not in ALLOWED_FORMATS:
        raise InputError(f"The {label} must be PNG or JPEG (got {img.format}).")
    return ImageOps.exif_transpose(img)  # same orientation the browser shows


def decode_image(data: bytes) -> np.ndarray:
    img = _open(data, "image")
    if img.width * img.height > MAX_PIXELS:
        raise InputError(
            f"The image is {img.width}x{img.height}; the limit is {MAX_PIXELS:,} pixels (about 2048x2048)."
        )
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    return np.array(img.convert("RGB"), dtype=np.uint8)


def decode_mask(data: bytes, expected_hw: tuple) -> np.ndarray:
    img = _open(data, "mask").convert("L")
    if (img.height, img.width) != tuple(expected_hw):
        raise InputError(
            f"The mask is {img.width}x{img.height} but the image is {expected_hw[1]}x{expected_hw[0]}. "
            "They must match."
        )
    mask = np.array(img, dtype=np.uint8)
    mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    white = int(np.count_nonzero(mask))
    if white == 0:
        raise InputError("The mask is empty. Paint the damaged area in white.")
    if white == mask.size:
        raise InputError("The mask covers the whole image. Leave some valid (black) pixels.")
    return mask


def encode_png(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return buf.getvalue()
