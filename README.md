# Image Inpainting Lab

React + Vite frontend, FastAPI backend, and five integration points for your own inpainting
implementations (Telea, Navier-Stokes, PatchMatch, K-SVD, LaMa).

**No algorithm is implemented here.** Until you connect one, the API reports it as
*unavailable* and returns no image for it. Nothing in the app produces fake results.

## Install

Requirements: Python 3.10+, Node.js 18+.

```bash
# backend
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# frontend
cd ../frontend
npm install
```

## Run (two terminals)

```bash
cd backend  && uvicorn main:app --port 8000 --reload
cd frontend && npm run dev            # open http://localhost:5173
```

The dev server forwards `/api` to `127.0.0.1:8000`. To call another backend, set
`VITE_API_URL` (for example `VITE_API_URL=http://myhost:8000 npm run dev`).
Interactive API docs: http://localhost:8000/docs

## Connect your five implementations

Each algorithm has one file. Open it, read the docstring at the top, and edit it:

| Algorithm     | File                                            | Registry key    |
|---------------|-------------------------------------------------|-----------------|
| Telea         | `backend/algorithms/classical/telea.py`         | `telea`         |
| Navier-Stokes | `backend/algorithms/classical/navier_stokes.py` | `navier_stokes` |
| PatchMatch    | `backend/algorithms/patchmatch/patchmatch.py`   | `patchmatch`    |
| K-SVD         | `backend/algorithms/ksvd/ksvd.py`               | `ksvd`          |
| LaMa          | `backend/algorithms/lama/lama.py`               | `lama`          |

For every file:

1. Copy your code into the same folder (or import it from where it lives).
2. Make `inpaint(image, mask, **kwargs)` call it and return the result.
3. Set `CONNECTED = True`, then restart the server.

```python
# backend/algorithms/classical/telea.py  (pattern, with your own function name)
import numpy as np
from .my_telea import telea_inpaint        # your existing code

CONNECTED = True

def inpaint(image: np.ndarray, mask: np.ndarray, **kwargs) -> np.ndarray:
    # image: RGB uint8 (H, W, 3)   mask: uint8 (H, W), 255 = missing, 0 = valid
    out = telea_inpaint(image, mask, **kwargs)   # convert formats here if yours differ
    return out                                    # must be RGB uint8, same shape as image
```

The backend requires a numpy `uint8` array with the same shape as the input image. Anything
else is reported as an error for that algorithm only.

**LaMa weights:** put them in `backend/algorithms/lama/weights/` (or wherever you like) and
load them inside `load()` into a module variable. `load()` runs once (at server start, or lazily
on first use), so every request reuses the same model. A failed load is shown in the UI.

**Extra parameters:** `POST /api/inpaint` accepts an optional `params` JSON, for example
`{"telea": {"radius": 5}}`, passed to that algorithm as keyword arguments.

## API

| Method | Path              | Purpose |
|--------|-------------------|---------|
| GET    | `/api/health`     | `{"status": "ok"}` |
| GET    | `/api/algorithms` | List of `{key, name, category, description, available, reason}` |
| POST   | `/api/inpaint`    | Run one or more algorithms (multipart form) |

`POST /api/inpaint` form fields:

- `image`: PNG or JPEG file (up to 25 MB, up to 2048x2048 pixels)
- `mask`: PNG or JPEG, same size as the image; white (>127) = damaged, black = valid
- `algorithms`: comma-separated keys, e.g. `telea,lama`
- `params`: optional JSON object

Response:

```json
{
  "width": 640, "height": 480, "masked_pixels": 21092,
  "results": [
    {"algorithm": "telea", "name": "Telea", "category": "Classical",
     "status": "success", "time_ms": 412.3, "message": null, "image_base64": "<PNG, base64>"},
    {"algorithm": "lama", "status": "unavailable", "message": "Not connected yet...", "image_base64": null}
  ]
}
```

`status` is `success`, `error` (the algorithm raised or returned an invalid array) or
`unavailable` (`CONNECTED = False`). Each algorithm gets its own copy of the same image and mask,
runs independently, and one failure never stops the others. Invalid input returns HTTP 400
with a readable `detail` message.

## Testing: two different things

**Interface tests** check the plumbing (validation, routing, error isolation, response format).
They do not test any algorithm:

```bash
cd backend && python -m pytest -q tests
```

They use throwaway stand-in functions that exist only inside `tests/test_api.py`.

**Algorithm tests** can only happen once you connect your code. Then open the app, load a
sample (or your own image and mask), run the algorithm, and judge the real output.
Samples are in `samples/` and `frontend/public/samples/` (regenerate with
`python samples/make_samples.py`). The sample masks are inputs, not results.

## Layout

```
backend/
  main.py                       FastAPI app
  api/routes.py                 endpoints
  services/algorithm_service.py registry, timing, error isolation, one-time load
  algorithms/<group>/<name>.py  YOUR integration points
  utils/image_io.py             decoding and validation
  tests/                        interface tests
frontend/src/                   App.jsx, api.js, components/ (4 steps + before/after slider)
samples/                        sample images and masks
```
