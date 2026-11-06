
"""
Optimized pure NumPy PatchMatch image inpainting.

Based on:
- Barnes et al. (2009): PatchMatch
- Wexler et al. (2007): Image completion

Features:
- Coarse-to-fine image pyramid
- Random nearest-neighbour initialization
- Neighbour propagation
- Exponentially shrinking random search
- Weighted patch SSD
- Overlap-based patch voting
- Chunked distance computation to reduce RAM usage
- Deterministic results with a fixed random seed

Dependencies: NumPy and OpenCV.
No C++ extensions or GPU required.
"""

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .._common import validate_inputs, keep_known_pixels

CONNECTED = True

# ---------------------------------------------------------
# 1. PATCH WINDOWS
# ---------------------------------------------------------

def _windows(arr, p):
    """Return sliding p x p windows without copying the image."""
    return sliding_window_view(
        arr,
        (p, p),
        axis=(0, 1)
    )


# ---------------------------------------------------------
# 2. CHUNKED PATCH DISTANCE
# ---------------------------------------------------------

def _patch_distance(img_w, wgt_w, tgt, cand, chunk_size=2048):
    """
    Calculate weighted SSD in chunks.

    Chunking limits temporary memory consumption when
    processing large images.
    """

    n = len(tgt)

    distances = np.empty(n, dtype=np.float32)

    for start in range(0, n, chunk_size):

        end = min(start + chunk_size, n)

        t = tgt[start:end]
        c = cand[start:end]

        a = img_w[t[:, 0], t[:, 1]]
        b = img_w[c[:, 0], c[:, 1]]

        weights = wgt_w[t[:, 0], t[:, 1]]

        diff = a - b

        numerator = (
            (diff * diff).sum(axis=1) * weights
        ).sum(axis=(1, 2))

        denominator = weights.sum(axis=(1, 2))

        distances[start:end] = (
            numerator / np.maximum(denominator, 1e-8)
        )

    return distances


# ---------------------------------------------------------
# 3. PATCHMATCH SEARCH
# ---------------------------------------------------------

def _patchmatch(
    img,
    hole,
    p,
    iters,
    rng,
    init_nnf=None,
    hole_weight=0.1,
    chunk_size=2048
):
    """
    Compute nearest-neighbour matches for patches
    touching the hole.
    """

    h, w = hole.shape

    gh = h - p + 1
    gw = w - p + 1

    if gh <= 0 or gw <= 0:
        raise ValueError("Patch size is larger than the image.")

    # Identify target and source patches
    hole_any = _windows(
        hole.astype(np.uint8), p
    ).any(axis=(2, 3))

    valid_src = ~hole_any

    targets = np.argwhere(hole_any)
    sources = np.argwhere(valid_src)

    if len(sources) == 0:
        raise ValueError("No valid source patches available.")

    if len(targets) == 0:
        return targets, targets.copy(), np.zeros(
            (gh, gw, 2), dtype=np.int64
        )

    img_w = _windows(img, p)

    wgt_w = _windows(
        np.where(hole, hole_weight, 1.0).astype(np.float32),
        p
    )

    # Initialize NNF with identity matches
    yy, xx = np.meshgrid(
        np.arange(gh),
        np.arange(gw),
        indexing="ij"
    )

    nnf = np.stack((yy, xx), axis=-1).astype(np.int64)

    # Random initialization
    random_indices = rng.integers(
        0, len(sources), size=len(targets)
    )

    nnf[
        targets[:, 0],
        targets[:, 1]
    ] = sources[random_indices]

    # Warm start from coarser level
    if init_nnf is not None:

        cand = init_nnf[
            targets[:, 0],
            targets[:, 1]
        ]

        inside = (
            (cand[:, 0] >= 0) &
            (cand[:, 0] < gh) &
            (cand[:, 1] >= 0) &
            (cand[:, 1] < gw)
        )

        valid = np.zeros(len(targets), dtype=bool)

        ids = np.flatnonzero(inside)

        if len(ids):
            valid[ids] = valid_src[
                cand[ids, 0],
                cand[ids, 1]
            ]

        nnf[
            targets[valid, 0],
            targets[valid, 1]
        ] = cand[valid]

    # Calculate initial distances
    cur = nnf[targets[:, 0], targets[:, 1]]

    best = _patch_distance(
        img_w, wgt_w, targets, cur, chunk_size
    )

    # Candidate evaluation
    def try_update(tgt, cand, best):

        inside = (
            (cand[:, 0] >= 0) &
            (cand[:, 0] < gh) &
            (cand[:, 1] >= 0) &
            (cand[:, 1] < gw)
        )

        cand = np.clip(
            cand,
            [0, 0],
            [gh - 1, gw - 1]
        )

        valid = np.zeros(len(tgt), dtype=bool)

        ids = np.flatnonzero(inside)

        if len(ids):
            valid[ids] = valid_src[
                cand[ids, 0],
                cand[ids, 1]
            ]

        ids = np.flatnonzero(valid)

        if len(ids) == 0:
            return best

        distances = _patch_distance(
            img_w,
            wgt_w,
            tgt[ids],
            cand[ids],
            chunk_size
        )

        better_local = distances < best[ids]

        better_ids = ids[better_local]

        if len(better_ids):

            nnf[
                tgt[better_ids, 0],
                tgt[better_ids, 1]
            ] = cand[better_ids]

            best[better_ids] = distances[better_local]

        return best

    # PatchMatch iterations
    for iteration in range(iters):

        # Propagation
        if iteration % 2 == 0:
            steps = [(-1, 0), (0, -1)]
        else:
            steps = [(1, 0), (0, 1)]

        for dy, dx in steps:

            ny = np.clip(
                targets[:, 0] + dy, 0, gh - 1
            )

            nx = np.clip(
                targets[:, 1] + dx, 0, gw - 1
            )

            cand = nnf[ny, nx] - np.array([dy, dx])

            best = try_update(
                targets, cand, best
            )

        # Exponentially shrinking random search
        radius = max(gh, gw)

        while radius >= 1:

            cur = nnf[
                targets[:, 0],
                targets[:, 1]
            ]

            offsets = rng.integers(
                -radius,
                radius + 1,
                size=cur.shape
            )

            cand = cur + offsets

            best = try_update(
                targets, cand, best
            )

            radius //= 2

    matches = nnf[
        targets[:, 0],
        targets[:, 1]
    ]

    return targets, matches, nnf


# ---------------------------------------------------------
# 4. PATCH VOTING
# ---------------------------------------------------------

def _vote(img, hole, targets, matches, p):
    """
    Reconstruct missing pixels by averaging matched
    source patches covering each target pixel.
    """

    h, w = hole.shape

    acc = np.zeros((h * w, 3), dtype=np.float64)
    cnt = np.zeros(h * w, dtype=np.float64)

    for dy in range(p):
        for dx in range(p):

            idx = (
                (targets[:, 0] + dy) * w
                + targets[:, 1] + dx
            )

            values = img[
                matches[:, 0] + dy,
                matches[:, 1] + dx
            ]

            cnt += np.bincount(
                idx,
                minlength=h * w
            )

            for channel in range(3):

                acc[:, channel] += np.bincount(
                    idx,
                    weights=values[:, channel],
                    minlength=h * w
                )

    out = img.copy()

    count_map = cnt.reshape(h, w)

    update = hole & (count_map > 0)

    voted = (
        acc / np.maximum(cnt, 1)[:, None]
    ).reshape(h, w, 3)

    out[update] = voted[update]

    return out


# ---------------------------------------------------------
# 5. IMAGE PYRAMID
# ---------------------------------------------------------

def _build_pyramid(image, hole, levels):

    pyramid = [(image, hole)]

    for _ in range(levels - 1):

        im, ho = pyramid[-1]

        height, width = im.shape[:2]

        if min(height, width) < 2:
            break

        size = (
            max(1, width // 2),
            max(1, height // 2)
        )

        small = cv2.resize(
            im,
            size,
            interpolation=cv2.INTER_AREA
        )

        small_hole = cv2.resize(
            ho.astype(np.uint8),
            size,
            interpolation=cv2.INTER_AREA
        ) > 0

        pyramid.append((small, small_hole))

    return pyramid


# ---------------------------------------------------------
# 6. MAIN INPAINTING FUNCTION
# ---------------------------------------------------------

def inpaint(
    image,
    mask,
    patch_size=5,
    levels=None,
    em_iters=2,
    pm_iters=2,
    seed=0,
    chunk_size=2048,
    verbose=False,
    **kwargs
):
    """
    Optimized pure NumPy PatchMatch inpainting.

    Parameters
    ----------
    image : RGB uint8 image
    mask : binary mask (255 = hole, 0 = known)
    patch_size : odd patch dimension
    levels : pyramid levels (None = automatic)
    em_iters : NNF search and voting repetitions
    pm_iters : PatchMatch iterations per NNF search
    seed : random seed
    chunk_size : maximum patches processed per distance batch
    verbose : display progress

    Returns
    -------
    Reconstructed RGB image.
    """

    image, mask = validate_inputs(image, mask)

    if patch_size < 3 or patch_size % 2 == 0:
        raise ValueError(
            "patch_size must be an odd integer >= 3"
        )

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    if not mask.any():
        return image.copy()

    if patch_size > min(image.shape[:2]):
        raise ValueError("Patch size exceeds image dimensions.")

    rng = np.random.default_rng(seed)

    hole = mask > 0

    # Never use hidden ground-truth pixels
    img = image.astype(np.float32).copy()
    img[hole] = 0

    # Automatically select pyramid depth
    if levels is None:

        levels = int(np.clip(
            1 + np.log2(
                min(img.shape[:2]) / (8 * patch_size)
            ),
            1,
            4
        ))

    if levels < 1:
        raise ValueError("levels must be >= 1")

    pyramid = _build_pyramid(
        img, hole, levels
    )

    # Remove levels without valid source patches
    while len(pyramid) > 1:

        coarse_hole = pyramid[-1][1]

        if not _windows(
            coarse_hole.astype(np.uint8),
            patch_size
        ).any((2, 3)).all():
            break

        valid = ~_windows(
            coarse_hole.astype(np.uint8),
            patch_size
        ).any((2, 3))

        if valid.any():
            break

        pyramid.pop()

    # Coarse-to-fine processing
    est = None
    nnf = None

    for level, (lvl_img, lvl_hole) in reversed(
        list(enumerate(pyramid))
    ):

        if verbose:
            print(
                f"PatchMatch level {level + 1}/{len(pyramid)}: "
                f"{lvl_img.shape[1]}x{lvl_img.shape[0]}",
                flush=True
            )

        if est is None:

            # Initial hole filling
            ring = cv2.dilate(
                lvl_hole.astype(np.uint8),
                np.ones((5, 5), np.uint8)
            ) > 0

            ring &= ~lvl_hole

            known = ~lvl_hole

            if not known.any():
                raise ValueError(
                    "Image contains no known pixels."
                )

            fill = (
                lvl_img[ring].mean(axis=0)
                if ring.any()
                else lvl_img[known].mean(axis=0)
            )

            est = lvl_img.copy()
            est[lvl_hole] = fill

        else:

            # Upsample estimate
            up = cv2.resize(
                est,
                (lvl_img.shape[1], lvl_img.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

            est = np.where(
                lvl_hole[..., None],
                up,
                lvl_img
            )

            # Upsample NNF
            if nnf is not None:

                gh = lvl_hole.shape[0] - patch_size + 1
                gw = lvl_hole.shape[1] - patch_size + 1

                big = np.repeat(
                    np.repeat(nnf * 2, 2, axis=0),
                    2,
                    axis=1
                )

                pad = np.zeros(
                    (gh, gw, 2),
                    dtype=np.int64
                )

                hh = min(gh, big.shape[0])
                ww = min(gw, big.shape[1])

                pad[:hh, :ww] = big[:hh, :ww]

                nnf = np.clip(
                    pad,
                    0,
                    [gh - 1, gw - 1]
                )

        # EM iterations
        for em in range(em_iters):

            if verbose:
                print(
                    f"  EM iteration {em + 1}/{em_iters}",
                    flush=True
                )

            targets, matches, nnf = _patchmatch(
                est,
                lvl_hole,
                patch_size,
                pm_iters,
                rng,
                nnf,
                chunk_size=chunk_size
            )

            est = _vote(
                est,
                lvl_hole,
                targets,
                matches,
                patch_size
            )

    result = np.clip(
        np.rint(est),
        0,
        255
    ).astype(np.uint8)

    return keep_known_pixels(
        image,
        result,
        mask
    )
