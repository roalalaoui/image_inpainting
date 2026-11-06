"""INTERFACE tests - these check the API plumbing, not the algorithms.

The stand-in functions below exist only inside this test file (via monkeypatch).
They are NOT shipped as algorithms and nothing in the app can reach them.
"""
import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import app  # noqa: E402
from services import algorithm_service as svc  # noqa: E402

client = TestClient(app)


def png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


IMG = np.full((40, 60, 3), 120, np.uint8)
MASK = np.zeros((40, 60), np.uint8)
MASK[10:20, 10:20] = 255


def post(algos, image=IMG, mask=MASK):
    return client.post("/api/inpaint",
                       files={"image": ("i.png", png(image), "image/png"), "mask": ("m.png", png(mask), "image/png")},
                       data={"algorithms": algos})


def test_all_unavailable_by_default():
    algos = client.get("/api/algorithms").json()["algorithms"]
    assert [a["key"] for a in algos] == ["telea", "navier_stokes", "patchmatch", "ksvd", "lama"]
    assert not any(a["available"] for a in algos)
    body = post("telea,lama").json()
    assert [r["status"] for r in body["results"]] == ["unavailable", "unavailable"]
    assert all(r["image_base64"] is None for r in body["results"])


def test_validation():
    assert post("").status_code == 400
    assert post("nope").status_code == 400
    assert post("telea", mask=np.zeros((40, 60), np.uint8)).status_code == 400          # empty mask
    assert post("telea", mask=np.zeros((10, 10), np.uint8)).status_code == 400          # size mismatch
    bad = client.post("/api/inpaint", files={"image": ("i.txt", b"hello", "text/plain"), "mask": ("m.png", png(MASK), "image/png")},
                      data={"algorithms": "telea"})
    assert bad.status_code == 400


def test_failure_isolation_and_same_input(monkeypatch):
    seen = {}
    ok = svc._import(svc.BY_KEY["telea"])
    boom = svc._import(svc.BY_KEY["ksvd"])
    bad_dtype = svc._import(svc.BY_KEY["lama"])

    def fake_ok(image, mask, **kw):
        seen["image"], seen["mask"] = image.copy(), mask.copy()
        return image
    def fake_boom(image, mask, **kw):
        image[:] = 0  # must not leak into other algorithms
        raise RuntimeError("deliberate failure")
    def fake_float(image, mask, **kw):
        return image.astype(float)

    for mod, fn in ((ok, fake_ok), (boom, fake_boom), (bad_dtype, fake_float)):
        monkeypatch.setattr(mod, "CONNECTED", True)
        monkeypatch.setattr(mod, "inpaint", fn)
    body = post("ksvd,telea,lama,patchmatch").json()
    by = {r["algorithm"]: r for r in body["results"]}
    assert by["ksvd"]["status"] == "error" and "deliberate" in by["ksvd"]["message"]
    assert by["lama"]["status"] == "error" and "uint8" in by["lama"]["message"]
    assert by["patchmatch"]["status"] == "unavailable"
    assert by["telea"]["status"] == "success" and by["telea"]["time_ms"] >= 0
    out = np.array(Image.open(io.BytesIO(base64.b64decode(by["telea"]["image_base64"]))))
    assert out.shape == IMG.shape
    assert np.array_equal(seen["image"], IMG) and np.array_equal(seen["mask"], MASK)
