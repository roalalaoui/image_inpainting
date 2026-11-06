"""HTTP endpoints."""
import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from services import algorithm_service as svc
from utils.image_io import InputError, decode_image, decode_mask

router = APIRouter(prefix="/api")


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/algorithms")
def algorithms():
    return {"algorithms": svc.list_algorithms()}


@router.post("/inpaint")
async def inpaint(
    image: UploadFile = File(..., description="PNG or JPEG image"),
    mask: UploadFile = File(..., description="PNG/JPEG mask, white = damaged, black = valid, same size as image"),
    algorithms: str = Form("", description="Comma-separated keys, e.g. 'telea,lama'"),
    params: str = Form("{}", description="Optional JSON: {'telea': {'radius': 3}}"),
):
    keys = [k.strip() for k in algorithms.split(",") if k.strip()]
    keys = list(dict.fromkeys(keys))  # drop duplicates, keep order
    if not keys:
        raise HTTPException(400, "Select at least one algorithm.")
    unknown = [k for k in keys if k not in svc.BY_KEY]
    if unknown:
        raise HTTPException(400, f"Unknown algorithm(s): {', '.join(unknown)}. Valid: {', '.join(svc.BY_KEY)}.")
    try:
        parsed = json.loads(params or "{}")
        if not isinstance(parsed, dict):
            raise ValueError
    except ValueError:
        raise HTTPException(400, "'params' must be a JSON object.")

    try:
        img = decode_image(await image.read())
        msk = decode_mask(await mask.read(), img.shape[:2])
    except InputError as exc:
        raise HTTPException(400, str(exc))

    results = svc.run_algorithms(keys, img, msk, parsed)
    return {
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "masked_pixels": int((msk == 255).sum()),
        "results": results,
    }
