"""K-SVD (Machine learning) - integration point.

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

Until CONNECTED is True the API reports "K-SVD" as unavailable and never returns an
image for it. Raise any exception to report a failure; other algorithms are not
affected.
"""
import warnings
import numpy as np
from sklearn.linear_model import orthogonal_mp
from numpy.lib.stride_tricks import sliding_window_view
from .._common import validate_inputs, keep_known_pixels

CONNECTED = True

def learn_dictionary(patches, n_atoms, sparsity, n_iter, rng):
    """K-SVD. patches: (n_features, n_samples) mean-removed signals. Returns dictionary (n_features, n_atoms)."""
    n_atoms = min(n_atoms, patches.shape[1])
    # initialise atoms with real training patches (not random noise), unit-normalised
    D = patches[:, rng.choice(patches.shape[1], n_atoms, replace=False)].copy()
    D /= np.maximum(np.linalg.norm(D, axis=0), 1e-8)
    for _ in range(n_iter):
        with warnings.catch_warnings():
            # Atoms are initialised from training patches, so those patches are reproduced
            # exactly by one atom; OMP then stops early (zero residual) and sklearn warns.
            # This is expected and harmless, so only this specific warning is silenced.
            warnings.filterwarnings("ignore", message="Orthogonal matching pursuit ended prematurely")
            codes = orthogonal_mp(D, patches, n_nonzero_coefs=sparsity)      # (n_atoms, n_samples)
        for k in range(n_atoms):
            users = np.flatnonzero(codes[k])
            if len(users) == 0:                                              # unused atom: re-seed
                err = ((patches - D @ codes) ** 2).sum(0)
                D[:, k] = patches[:, err.argmax()] / max(np.linalg.norm(patches[:, err.argmax()]), 1e-8)
                continue
            # residual without atom k, restricted to the signals that use it
            residual = patches[:, users] - D @ codes[:, users] + np.outer(D[:, k], codes[k, users])
            u, s, vt = np.linalg.svd(residual, full_matrices=False)          # best rank-1 fit
            D[:, k] = u[:, 0]
            codes[k, users] = s[0] * vt[0]
    return D


def _masked_omp(D, y, known, sparsity):
    """OMP using only the known entries of y. Returns full-length sparse reconstruction D @ a."""
    Dk, yk = D[known], y[known]
    norms = np.maximum(np.linalg.norm(Dk, axis=0), 1e-8)
    residual, chosen, coef = yk.copy(), [], np.zeros(0)
    for _ in range(min(sparsity, int(known.sum()))):
        k = int(np.argmax(np.abs(Dk.T @ residual) / norms))
        if k in chosen:
            break
        chosen.append(k)
        coef, *_ = np.linalg.lstsq(Dk[:, chosen], yk, rcond=None)
        residual = yk - Dk[:, chosen] @ coef
    return D[:, chosen] @ coef if chosen else np.zeros_like(y)


def _to_vectors(windows):
    """(N, 3, p, p) windows -> (N, 3*p*p) vectors (channel-major)."""
    return windows.reshape(len(windows), -1)


def inpaint(image, mask, patch_size=8, n_atoms=256, sparsity=5, n_iter=10, n_train=6000,
            stride=2, min_known=0.4, seed=0, **kwargs):
    """K-SVD inpainting.

    patch_size: patch side. n_atoms: dictionary size. sparsity: non-zeros per patch (OMP).
    n_iter: K-SVD iterations. n_train: max training patches (sampled from known patches).
    stride: step between reconstructed patches. min_known: min. fraction of known pixels a
    patch needs to be reconstructed in a round. seed: random seed.
    """
    image, mask = validate_inputs(image, mask)
    if patch_size < 3 or sparsity < 1 or n_atoms < 2 or n_iter < 1 or stride < 1 or not 0 < min_known <= 1:
        raise ValueError("invalid K-SVD parameters")
    if not mask.any():
        return image.copy()
    rng = np.random.default_rng(seed)
    p = patch_size
    hole = mask > 0
    img = image.astype(np.float64) / 255.0
    img[hole] = 0                                  # hidden pixels are never used (no leakage)

    # ---- 1. learn the dictionary from fully-known patches of this image -------------------
    chans = np.moveaxis(img, 2, 0)                                           # (3, H, W)
    wins = sliding_window_view(chans, (p, p), axis=(1, 2))                   # (3, gh, gw, p, p)
    wins = np.moveaxis(wins, 0, 2)                                           # (gh, gw, 3, p, p)
    clean = ~sliding_window_view(hole, (p, p)).any((2, 3))                   # patch fully known
    pos = np.argwhere(clean)
    if len(pos) < 20:
        raise ValueError("too few fully-known patches to learn a dictionary; use a smaller patch_size")
    pos = pos[rng.choice(len(pos), min(n_train, len(pos)), replace=False)]
    train = _to_vectors(wins[pos[:, 0], pos[:, 1]]).T                        # (3p², N)
    means = train.reshape(3, p * p, -1).mean(1, keepdims=True)
    train = (train.reshape(3, p * p, -1) - means).reshape(3 * p * p, -1)
    # perfectly flat patches become all-zero signals after mean removal: they carry no
    # structure and would create zero-norm atoms, so they are excluded from training
    train = train[:, np.linalg.norm(train, axis=0) > 1e-3]
    if train.shape[1] < 20:
        raise ValueError("too few textured known patches to learn a dictionary")
    D = learn_dictionary(train, n_atoms, sparsity, n_iter, rng)

    # ---- 2. progressive reconstruction of the hole ---------------------------------------
    known = ~hole
    h, w = hole.shape
    for _ in range(4 * max(h, w)):                                           # safety bound
        if known.all():
            break
        cover = sliding_window_view(known, (p, p)).mean((2, 3))              # known fraction per patch
        needs_fill = cover < 1.0
        eligible = np.argwhere(needs_fill & (cover >= min_known))
        if len(eligible) == 0:                                               # relax threshold
            best = cover[needs_fill].max()
            eligible = np.argwhere(needs_fill & (cover >= best - 1e-9))
        eligible = eligible[(eligible[:, 0] % stride == 0) & (eligible[:, 1] % stride == 0)] \
            if len(eligible) > stride * stride else eligible
        acc, cnt = np.zeros((h, w, 3)), np.zeros((h, w))
        for y0, x0 in eligible:
            patch = img[y0:y0 + p, x0:x0 + p]                                # (p, p, 3)
            kn = known[y0:y0 + p, x0:x0 + p]
            vec = np.moveaxis(patch, 2, 0).reshape(3, p * p)
            kvec = np.tile(kn.reshape(1, -1), (3, 1))
            mean = (vec * kvec).sum(1) / kvec.sum(1)                         # mean of known pixels
            y = (vec - mean[:, None]).reshape(-1)
            rec = _masked_omp(D, y, kvec.reshape(-1), sparsity).reshape(3, p * p) + mean[:, None]
            acc[y0:y0 + p, x0:x0 + p] += np.moveaxis(rec.reshape(3, p, p), 0, 2)
            cnt[y0:y0 + p, x0:x0 + p] += 1
        fill = (~known) & (cnt > 0)
        if not fill.any():                                                   # cannot progress
            break
        img[fill] = acc[fill] / cnt[fill][:, None]
        known = known | fill
    if not known.all():
        raise RuntimeError("K-SVD could not fill the whole hole")

    result = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
    return keep_known_pixels(image, result, mask)
