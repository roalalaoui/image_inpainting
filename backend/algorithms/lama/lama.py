"""LaMa (Deep learning) - integration point.

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

Until CONNECTED is True the API reports "LaMa" as unavailable and never returns an
image for it. Raise any exception to report a failure; other algorithms are not
affected.
"""

import os
import tempfile
import warnings

from pathlib import Path

import numpy as np

from .._common import validate_inputs, keep_known_pixels

CONNECTED = True

# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_CHECKPOINT = Path(
    os.environ.get(
        "LAMA_CHECKPOINT",
        str(Path.home() / ".cache" / "lama" / "big-lama.pt")
    )
).expanduser()

WEIGHTS_URL = (
    "https://github.com/enesmsahin/simple-lama-inpainting/"
    "releases/download/v0.1.0/big-lama.pt"
)

# Cache loaded models.
# Key: checkpoint path + device + file modification information.
_MODEL_CACHE = {}

def load():
  load_model()
# ============================================================
# DEVICE SELECTION
# ============================================================

def get_device(device=None):
    """Automatically select CUDA or CPU."""

    import torch

    if device is None or device == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            "CUDA is unavailable. Falling back to CPU."
        )
        return torch.device("cpu")

    return device


# ============================================================
# CHECKPOINT VALIDATION
# ============================================================

def validate_checkpoint(checkpoint):
    """Check that the checkpoint exists and is not obviously incomplete."""

    checkpoint = Path(checkpoint)

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"LaMa checkpoint does not exist:\n{checkpoint}"
        )

    size = checkpoint.stat().st_size

    if size < 1024 * 1024:
        raise RuntimeError(
            f"LaMa checkpoint is too small: "
            f"{size / (1024 * 1024):.2f} MB.\n"
            "The download may be incomplete or invalid."
        )

    return checkpoint


# ============================================================
# DOWNLOAD PRETRAINED WEIGHTS
# ============================================================

def download_weights(destination=DEFAULT_CHECKPOINT, force=False):
    """
    Download the official big-lama TorchScript checkpoint.

    The downloaded file is validated before replacing
    the destination checkpoint.
    """

    import torch

    destination = Path(destination).expanduser()

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if destination.exists() and not force:

        try:
            validate_checkpoint(destination)
            torch.jit.load(
                str(destination),
                map_location="cpu"
            )
            print(f"LaMa checkpoint already exists: {destination}")
            return destination

        except Exception:
            print("Existing checkpoint is invalid. Downloading again...")

    temp_path = None

    try:

        # Create temporary file in destination directory
        fd, temp_name = tempfile.mkstemp(
            prefix="lama_download_",
            suffix=".pt",
            dir=str(destination.parent)
        )

        os.close(fd)
        temp_path = Path(temp_name)

        print("Downloading LaMa pretrained weights...")
        print(f"URL: {WEIGHTS_URL}")

        torch.hub.download_url_to_file(
            WEIGHTS_URL,
            str(temp_path),
            progress=True
        )

        # Validate downloaded size
        if temp_path.stat().st_size < 1024 * 1024:
            raise RuntimeError(
                "Downloaded checkpoint is too small. "
                "Check your internet connection or download URL."
            )

        # Verify that the file is a valid TorchScript archive
        print("Verifying downloaded checkpoint...")

        test_model = torch.jit.load(
            str(temp_path),
            map_location="cpu"
        )

        del test_model

        # Replace old checkpoint only after successful validation
        os.replace(temp_path, destination)

        # Remove cached models for this checkpoint
        clear_model_cache(destination)

        print("LaMa weights downloaded successfully!")
        print(f"Saved to: {destination}")

        return destination

    except Exception as e:

        raise RuntimeError(
            "Failed to download or validate LaMa weights.\n"
            f"Destination: {destination}\n"
            f"Reason: {e}"
        ) from e

    finally:

        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


# ============================================================
# MODEL CACHE
# ============================================================

def clear_model_cache(checkpoint=None):
    """Clear all cached models or models for one checkpoint."""

    if checkpoint is None:
        _MODEL_CACHE.clear()
        return

    checkpoint = Path(checkpoint).expanduser().resolve()

    keys = [
        key for key in _MODEL_CACHE
        if key[0] == str(checkpoint)
    ]

    for key in keys:
        del _MODEL_CACHE[key]


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(checkpoint=None, device=None):
    """
    Load the pretrained LaMa TorchScript model.

    Uses caching to avoid loading the model repeatedly.
    """

    import torch

    if checkpoint is None:
        checkpoint = DEFAULT_CHECKPOINT

    checkpoint = validate_checkpoint(checkpoint)
    checkpoint = checkpoint.resolve()

    device = get_device(device)

    # Include file modification information in cache key.
    stat = checkpoint.stat()

    cache_key = (
        str(checkpoint),
        str(device),
        stat.st_size,
        stat.st_mtime_ns
    )

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    print(f"Loading LaMa model on {device}...")

    try:

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                module="torch.jit._serialization"
            )

            model = torch.jit.load(
                str(checkpoint),
                map_location=device
            )

        model = model.to(device)
        model.eval()

        _MODEL_CACHE[cache_key] = model

        print("LaMa model loaded successfully.")

        return model

    except Exception as e:

        raise RuntimeError(
            "\nUnable to load LaMa checkpoint.\n"
            f"Checkpoint: {checkpoint}\n"
            f"Size: {stat.st_size / (1024 * 1024):.2f} MB\n"
            f"Device: {device}\n"
            f"Error: {e}\n\n"
            "The checkpoint may be corrupted or may not be "
            "a compatible TorchScript model.\n"
            "Try download_weights(force=True)."
        ) from e


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def prepare_image(image, mask):
    """
    Prepare image and mask for LaMa.

    Image: H x W x 3, uint8.
    Mask: H x W, nonzero = missing pixels.

    Pads dimensions to multiples of 8.
    """

    h, w = mask.shape

    pad_h = (-h) % 8
    pad_w = (-w) % 8

    image_p = np.pad(
        image,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="symmetric"
    )

    mask_p = np.pad(
        mask,
        ((0, pad_h), (0, pad_w)),
        mode="symmetric"
    )

    return image_p, mask_p, h, w


# ============================================================
# LAMA INPAINTING
# ============================================================

def inpaint(
    image,
    mask,
    checkpoint=None,
    device=None,
    **kwargs
):
    """
    LaMa pretrained inpainting.

    Parameters
    ----------
    image : numpy.ndarray
        RGB image (H, W, 3).

    mask : numpy.ndarray
        Binary mask (H, W).
        Nonzero = missing pixels.

    checkpoint : str or Path, optional
        Path to big-lama.pt.

    device : str, optional
        'cpu', 'cuda', or 'auto'.

    Returns
    -------
    numpy.ndarray
        Reconstructed RGB image (H, W, 3), uint8.
    """

    import torch

    # Validate inputs
    image, mask = validate_inputs(image, mask)

    # No missing pixels
    if not mask.any():
        return image.copy()

    # Load model
    model = load_model(checkpoint, device)

    device = next(model.parameters()).device

    # Prepare image and mask
    image_p, mask_p, h, w = prepare_image(image, mask)

    # Convert image to float32 tensor
    img_t = (
        torch.from_numpy(
            np.ascontiguousarray(image_p)
        )
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )

    # Mask: 1 = hole, 0 = known
    mask_t = (
        torch.from_numpy(
            np.ascontiguousarray(mask_p > 0)
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )

    # Inference
    with torch.inference_mode():

        output = model(img_t, mask_t)

        # Support models returning tuples or lists
        if isinstance(output, (tuple, list)):
            output = output[0]

        if not isinstance(output, torch.Tensor):
            raise RuntimeError(
                "LaMa returned an unsupported output type: "
                f"{type(output)}"
            )

        if output.ndim != 4:
            raise RuntimeError(
                f"Unexpected LaMa output shape: {output.shape}"
            )

        if output.shape[0] != 1 or output.shape[1] != 3:
            raise RuntimeError(
                f"Expected output shape (1, 3, H, W), "
                f"received {tuple(output.shape)}"
            )

        # Convert to H x W x 3
        output = (
            output[0]
            .permute(1, 2, 0)
            .float()
            .cpu()
            .numpy()
        )

    # Remove padding
    output = output[:h, :w, :]

    # Convert to uint8
    result = np.clip(
        np.rint(output * 255.0),
        0,
        255
    ).astype(np.uint8)

    # Preserve original known pixels
    result = keep_known_pixels(
        image,
        result,
        mask
    )

    return result

