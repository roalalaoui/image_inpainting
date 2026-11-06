"""Shared helpers: input validation and format conversion used by all five algorithms."""
import numpy as np


def validate_inputs(image, mask):
    """Check types/shapes and return (image, binary_mask).

    image: HxWx3 uint8 RGB.  mask: HxW uint8, 255 = missing, 0 = valid.
    Returns the image unchanged and a strict 0/255 uint8 mask (values > 127 are holes).
    """
    if not isinstance(image, np.ndarray) or not isinstance(mask, np.ndarray):
        raise TypeError("image and mask must be NumPy arrays")
    if image.dtype != np.uint8:
        raise TypeError(f"image must be uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3) RGB, got {image.shape}")
    if mask.dtype != np.uint8:
        raise TypeError(f"mask must be uint8, got {mask.dtype}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D grayscale, got shape {mask.shape}")
    if mask.shape != image.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match image {image.shape[:2]}")
    if min(image.shape[:2]) < 16:
        raise ValueError("image must be at least 16x16 pixels")
    binary = np.where(mask > 127, 255, 0).astype(np.uint8)
    if not binary.any():
        return image, binary          # nothing to inpaint (callers return image as is)
    if binary.all():
        raise ValueError("mask covers the whole image: no valid pixels to inpaint from")
    return image, binary


def keep_known_pixels(original, result, mask):
    """Return `result` with every valid (mask == 0) pixel restored from `original`."""
    out = original.copy()
    hole = mask > 0
    out[hole] = result[hole]
    return out
