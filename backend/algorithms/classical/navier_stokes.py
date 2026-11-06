"""Navier-Stokes (Classical) - integration point.

HOW TO CONNECT YOUR EXISTING IMPLEMENTATION
  1. Copy your code next to this file (or import it from here).
  2. Make `inpaint` below call it.
  3. Set CONNECTED = True and restart the server.

INPUT (identical for every algorithm)
  image : numpy uint8, (H, W, 3), RGB
  mask  : numpy uint8, (H, W), 255 = missing pixels, 0 = valid pixels
OUTPUT
  numpy uint8, (H, W, 3), RGB, same size as the input image.

If your code needs another format (BGR, float, 0/1 mask, ...), convert INSIDE this
file and convert the result back to RGB uint8 before returning.

Until CONNECTED is True the API reports "Navier-Stokes" as unavailable and never returns an
image for it. Raise any exception to report a failure; other algorithms are not
affected.
"""
import cv2
from .._common import validate_inputs

CONNECTED = True

def inpaint(image, mask, radius=3, **kwargs):
    """radius: neighbourhood radius (pixels) considered for each filled pixel."""
    image, mask = validate_inputs(image, mask)
    if radius <= 0:
        raise ValueError("radius must be positive")
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)             # OpenCV works in BGR
    out = cv2.inpaint(bgr, mask, float(radius), cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
