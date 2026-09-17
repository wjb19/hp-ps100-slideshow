#!/usr/bin/env python3
"""Scan old photos on the HP PS100 into a fixed slideshow portal.

Every output frame is the same size (the portal). Photos of different physical
sizes are auto-cropped, aspect-corrected, then letterboxed into that portal so
a slideshow player can advance without layout jumps.

With --crop-to-portal (or --fit cover), cropping prefers keeping as many faces
in frame as possible when OpenCV + the bundled YuNet model are available.
If no faces are found, salient subjects (boats, buildings, etc.) are used instead.
If an old-fashioned white print border is detected, aspect cropping is skipped
and the print is upscaled to fit the portal while keeping its native aspect ratio.
Small prints are enlarged with stepwise Lanczos interpolation plus a light unsharp.
By default frames are rendered at 2× portal density (e.g. 3840×2160 for 1080p) so
a slideshow can zoom in without immediately running out of pixels.
Pass --smooth-grid to soften scanner/paper mesh (opt-in).

Recommended for vintage prints:
  sudo python3 photo_slideshow.py --old-photos --out ~/Pictures/album
  sudo python3 photo_slideshow.py --old-photos --out ~/Pictures/album --count 12

Other examples:
  python3 photo_slideshow.py --from-raw /tmp/ps100.raw --out /tmp/album --old-photos
  sudo python3 photo_slideshow.py --out ~/Pictures/album --portal 1920x1080 --crop-to-portal

Keys during interactive scanning: Enter = scan next, q = quit.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Prefer project venv when present (ships OpenCV for face-aware cropping).
_VENV_PY = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
if _VENV_PY.exists() and Path(sys.executable).resolve() != _VENV_PY.resolve():
    try:
        import cv2  # noqa: F401
    except ImportError:
        os.execv(str(_VENV_PY), [str(_VENV_PY), *sys.argv])

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore
    _HAS_CV2 = False

from bot import Bot, hx
from protocol import (
    EXPECTED_IMAGE_BYTES,
    PARAM_WIDTH,
    content_bbox,
    decode_rowsbs_rgb,
    ping,
    prepare_scan,
    read_image,
    resolve_yscale,
    stop_scan,
    stretch_contrast,
)


DEFAULT_PORTAL = (1920, 1080)
# Soft charcoal matte — neutral for B&W and color prints without harsh black bars.
DEFAULT_MATTE = (28, 26, 24)
DEFAULT_DESKEW_LIMIT = 4.0
# Render above display portal so slideshow zoom still has pixels to spare.
DEFAULT_DENSITY = 2.0
# Soften scanner/paper grid lines (Radon + spatial morph). Opt-in via --smooth-grid only.
DEFAULT_SMOOTH_GRID = False

# Recommended treatment bundle for faded / bordered / group prints.
OLD_PHOTO_PRESET = {
    "portal": "1920x1080",
    "matte": "28,26,24",
    "fit": "contain",
    "crop_to_portal": True,
    "crop_threshold": 0.03,
    "enhance": 1.15,
    "rotate": "auto",
    "yscale": "auto",
    "deskew": DEFAULT_DESKEW_LIMIT,
    "face_aware": True,
    "keep_white_border": True,
    "density": DEFAULT_DENSITY,
}


def parse_portal(spec: str) -> tuple[int, int]:
    spec = spec.lower().strip()
    presets = {
        "1080p": (1920, 1080),
        "720p": (1280, 720),
        "4k": (3840, 2160),
        "4:3": (1600, 1200),
        "5:4": (1500, 1200),
        "3:2": (1920, 1280),
        "1:1": (1600, 1600),
        "square": (1600, 1600),
    }
    if spec in presets:
        return presets[spec]
    if "x" in spec:
        a, b = spec.split("x", 1)
        return int(a), int(b)
    raise argparse.ArgumentTypeError(f"bad portal {spec!r}; try 1920x1080 or 4:3")


def parse_rgb(spec: str) -> tuple[int, int, int]:
    parts = [int(x) for x in spec.replace(",", " ").split()]
    if len(parts) != 3 or any(c < 0 or c > 255 for c in parts):
        raise argparse.ArgumentTypeError("matte must be R G B (0-255)")
    return parts[0], parts[1], parts[2]


def render_portal_size(portal: tuple[int, int], density: float) -> tuple[int, int]:
    """Scale portal by density for zoom headroom (e.g. 2× → 4K from 1080p)."""
    d = max(1.0, float(density))
    return (
        max(1, int(round(portal[0] * d))),
        max(1, int(round(portal[1] * d))),
    )


def raw_to_rgb_image(raw: bytes, width: int = PARAM_WIDTH, y_scale: float | str = "auto") -> Image.Image:
    y_scale = resolve_yscale(raw, y_scale, width=width)
    w, h, rgb = decode_rowsbs_rgb(raw, width=width, y_scale=y_scale)
    rgb = stretch_contrast(rgb)
    return Image.frombytes("RGB", (w, h), rgb)


def _longest_run(mask_1d: np.ndarray) -> tuple[int, int]:
    """Return [start, end) of the longest True run; full span if none."""
    n = len(mask_1d)
    best = (0, 0)
    i = 0
    while i < n:
        if not mask_1d[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask_1d[j]:
            j += 1
        if j - i > best[1] - best[0]:
            best = (i, j)
        i = j
    if best[1] <= best[0]:
        return 0, n
    return best


def _grow_box_to_content(
    gray: np.ndarray,
    box: tuple[int, int, int, int],
    thr: int = 28,
    strip_frac: float = 0.12,
    max_steps: int = 800,
) -> tuple[int, int, int, int]:
    """Expand a seed box while bordering strips still look like print content."""
    h, w = gray.shape
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, max(x0 + 1, x1)), min(h, max(y0 + 1, y1))
    mask = gray > thr
    for _ in range(max_steps):
        grew = False
        if x0 > 0 and float(mask[y0:y1, x0 - 1].mean()) >= strip_frac:
            x0 -= 1
            grew = True
        if x1 < w and float(mask[y0:y1, x1].mean()) >= strip_frac:
            x1 += 1
            grew = True
        if y0 > 0 and float(mask[y0 - 1, x0:x1].mean()) >= strip_frac:
            y0 -= 1
            grew = True
        if y1 < h and float(mask[y1, x0:x1].mean()) >= strip_frac:
            y1 += 1
            grew = True
        if not grew:
            break
    return x0, y0, x1, y1


def dense_content_bbox(
    gray: np.ndarray,
    thr: int | None = None,
    min_frac: float = 0.18,
    pad: int = 8,
    chroma: np.ndarray | None = None,
) -> tuple[int, int, int, int]:
    """BBox of the densest bright region (photo print), not sparse scanner noise."""
    h, w = gray.shape
    work = gray.astype(np.float32, copy=True)
    # Rainbow decode stripes are high-chroma; zero them so they don't win the bbox.
    if chroma is not None:
        work = work.copy()
        work[chroma > 40] = 0

    med = float(np.median(gray))
    if thr is None:
        thr_list = [22, 28, 35] if med < 20 else [int(max(28, min(60, med + 12))), 28, 35]
    else:
        thr_list = [thr]

    # Allow tiny prints on a large bed (absolute floor, not fraction of bed width).
    min_side = 48
    best: tuple[float, tuple[int, int, int, int]] | None = None
    for t in thr_list:
        mask = work > t
        fill = float(mask.mean())
        col_frac = mask.mean(axis=0)
        row_frac = mask.mean(axis=1)
        for frac in (min_frac, 0.15, 0.12, 0.08, 0.05):
            cx0, cx1 = _longest_run(col_frac >= frac)
            cy0, cy1 = _longest_run(row_frac >= frac)
            bw, bh = cx1 - cx0, cy1 - cy0
            if bw < min_side or bh < min_side:
                continue
            ar = bw / max(1, bh)
            if ar < 0.35 or ar > 2.8:
                continue
            # When the bed is mostly empty, reject near-full-width voids.
            if fill < 0.45 and (bw > 0.85 * w or bh > 0.85 * h) and frac < 0.15:
                continue
            if bw * bh > 0.92 * w * h and frac < 0.15:
                continue
            # Prefer denser, then prefer compact (small prints over bed+noise).
            compactness = 1.0 / (1.0 + (bw * bh) / max(1.0, 0.15 * w * h))
            score = frac * 1e9 + compactness * 1e6 + bw * bh
            box = (cx0, cy0, cx1, cy1)
            if best is None or score > best[0]:
                best = (score, box)

    if best is None:
        return 0, 0, w, h
    x0, y0, x1, y1 = best[1]
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    return x0, y0, x1, y1


def _face_union_box(
    faces: list[tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
    margin_scale: float = 2.2,
) -> tuple[int, int, int, int]:
    x0 = min(x for x, y, w, h in faces)
    y0 = min(y for x, y, w, h in faces)
    x1 = max(x + w for x, y, w, h in faces)
    y1 = max(y + h for x, y, w, h in faces)
    # Pad from typical face size — not from the union span (group shots span wide).
    face_span = max(max(w, h) for x, y, w, h in faces)
    pad = int(max(face_span * margin_scale, 48))
    pad_top = int(pad * 1.15)
    pad_bot = int(pad * 0.95)
    return (
        max(0, x0 - pad),
        max(0, y0 - pad_top),
        min(img_w, x1 + pad),
        min(img_h, y1 + pad_bot),
    )


def trim_scanner_bed(
    gray: np.ndarray,
    thr: int = 30,
    dark_frac: float = 0.82,
    mean_cap: float = 14.0,
    strip: int = 1,
    pad: int = 1,
) -> tuple[int, int, int, int]:
    """Shrink bbox by peeling edge lines that are still mostly scanner-black.

    Requires a low edge mean so dark photo content (foliage, clothing) is kept.
    Peels one row/column at a time so a 1–2px bed hairline isn't blocked by
    neighboring photo pixels in a wider strip.
    """
    h, w = gray.shape
    x0, y0, x1, y1 = 0, 0, w, h
    strip = max(1, strip)

    def mostly_bed(region: np.ndarray) -> bool:
        if region.size < 1:
            return False
        return float(region.mean()) <= mean_cap and float((region <= thr).mean()) >= dark_frac

    guard = 0
    while guard < max(w, h) and (x1 - x0) > 64 and (y1 - y0) > 64:
        guard += 1
        moved = False
        if mostly_bed(gray[y0:y1, x0 : x0 + strip]):
            x0 += strip
            moved = True
        if mostly_bed(gray[y0:y1, x1 - strip : x1]):
            x1 -= strip
            moved = True
        if mostly_bed(gray[y0 : y0 + strip, x0:x1]):
            y0 += strip
            moved = True
        if mostly_bed(gray[y1 - strip : y1, x0:x1]):
            y1 -= strip
            moved = True
        if not moved:
            break
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(w, x1 + pad),
        min(h, y1 + pad),
    )


def auto_crop(
    img: Image.Image,
    pad: int = 8,
    thr: int = 22,
    faces: list[tuple[int, int, int, int]] | None = None,
) -> Image.Image:
    """Trim scanner void / black margins around the photo print.

    When faces are provided (or detected), prefers a face-seeded crop so tiny
    prints on a large bed still get isolated and upscaled.
    """
    rgb = np.asarray(img.convert("RGB"))
    gray = np.asarray(img.convert("L"))
    h, w = gray.shape
    chroma = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)

    face_list = list(faces) if faces is not None else detect_faces(img)
    x0, y0, x1, y1 = dense_content_bbox(gray, thr=None, pad=pad, chroma=chroma)
    area = (x1 - x0) * (y1 - y0)
    bed_area = w * h
    dense_ok = area >= 32 * 32 and area <= 0.85 * bed_area

    if face_list:
        fx0, fy0, fx1, fy1 = _face_union_box(face_list, w, h)
        contains = x0 <= fx0 and y0 <= fy0 and x1 >= fx1 and y1 >= fy1
        if (not dense_ok) or (not contains) or area > 0.5 * bed_area:
            # Seed from faces, then grow to the print edges (skipping chroma noise).
            clean = gray.astype(np.float32).copy()
            clean[chroma > 40] = 0
            x0, y0, x1, y1 = _grow_box_to_content(clean, (fx0, fy0, fx1, fy1), thr=max(thr, 26))
            x0 = max(0, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(w, x1 + pad)
            y1 = min(h, y1 + pad)
            dense_ok = (x1 - x0) >= 32 and (y1 - y0) >= 32

    if not dense_ok:
        data = gray.tobytes()
        x0, y0, x1, y1 = content_bbox(data, w, h, thr=thr)
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(w, x1 + pad)
        y1 = min(h, y1 + pad)

    if x1 - x0 < 32 or y1 - y0 < 32:
        return img

    cropped = img.crop((x0, y0, x1, y1))
    # Final peel of residual black bed (and deskew-style void) at the edges.
    g2 = np.asarray(cropped.convert("L"))
    tx0, ty0, tx1, ty1 = trim_scanner_bed(g2, thr=max(thr + 4, 26), pad=1)
    if (tx1 - tx0) >= 32 and (ty1 - ty0) >= 32:
        cropped = cropped.crop((tx0, ty0, tx1, ty1))
    return cropped


def auto_orient(img: Image.Image, prefer_landscape: bool = True) -> Image.Image:
    """Orient using faces when possible; otherwise optional landscape preference.

    Portrait prints are no longer blindly rotated for the portal — that sideways'd
    bordered snapshots. Prefer the rotation with the most upright faces.
    """
    w, h = img.size
    if max(w, h) < 64:
        return img

    candidates = [img]
    if h > w * 1.05 or w > h * 1.05:
        candidates.append(img.transpose(Image.Transpose.ROTATE_270))
        candidates.append(img.transpose(Image.Transpose.ROTATE_90))

    best = img
    best_score = (-1, -1, -1)

    for im in candidates:
        faces = detect_faces(im)
        n = len(faces)
        # Upright heads are typically taller than they are wide.
        tall = sum(1 for _x, _y, fw, fh in faces if fh >= fw * 0.95)
        iw, ih = im.size
        land = 1 if (prefer_landscape and iw >= ih) else (1 if iw >= ih else 0)
        # Face count wins; then upright-ish boxes; then landscape preference.
        score = (n, tall, land if prefer_landscape else -land)
        if score > best_score:
            best_score = score
            best = im
    return best


def _edge_map(gray: np.ndarray) -> np.ndarray:
    """Simple Sobel magnitude; favors print borders and strong photo edges."""
    g = gray.astype(np.float32)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.hypot(gx, gy)
    # Keep the strongest edges only so texture doesn't dominate skew score.
    thr = np.percentile(mag, 90)
    return (mag >= thr).astype(np.float32)


def _projection_score(edges: np.ndarray) -> float:
    """Higher when horizontal edges align to rows (deskewed)."""
    row = edges.sum(axis=1)
    col = edges.sum(axis=0)
    # Favor strong row alignment; column term stabilizes mostly-vertical content.
    return float(row.var() + 0.35 * col.var())


def estimate_skew_angle(
    img: Image.Image,
    limit: float = DEFAULT_DESKEW_LIMIT,
    coarse: float = 0.5,
    fine: float = 0.1,
) -> float:
    """Estimate small skew in degrees (positive = CCW) via edge projection search."""
    if limit <= 0:
        return 0.0
    # Work on a modest preview for speed.
    preview = img.convert("L")
    max_side = 900
    w, h = preview.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        preview = preview.resize(
            (max(32, int(w * scale)), max(32, int(h * scale))),
            Image.Resampling.BILINEAR,
        )
    # Light blur reduces film grain noise in old prints.
    preview = preview.filter(ImageFilter.GaussianBlur(radius=0.8))
    gray = np.asarray(preview, dtype=np.float32)
    edges = _edge_map(gray)
    if edges.mean() < 1e-6:
        return 0.0

    def best_in(angles: list[float]) -> tuple[float, float]:
        best_a, best_s = 0.0, -1.0
        for a in angles:
            if abs(a) < 1e-6:
                sample = edges
            else:
                rot = Image.fromarray((edges * 255).astype(np.uint8)).rotate(
                    a, resample=Image.Resampling.BILINEAR, fillcolor=0
                )
                sample = (np.asarray(rot) > 128).astype(np.float32)
            s = _projection_score(sample)
            if s > best_s:
                best_s, best_a = s, a
        return best_a, best_s

    coarse_angles = [i * coarse for i in range(int(-limit / coarse), int(limit / coarse) + 1)]
    a0, _ = best_in(coarse_angles)
    fine_angles = [
        a0 + i * fine
        for i in range(int(-coarse / fine), int(coarse / fine) + 1)
        if abs(a0 + i * fine) <= limit + 1e-6
    ]
    angle, _ = best_in(fine_angles or [a0])

    # Ignore tiny jitter.
    if abs(angle) < 0.15:
        return 0.0
    return round(angle, 2)


def deskew(
    img: Image.Image,
    limit: float = DEFAULT_DESKEW_LIMIT,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> tuple[Image.Image, float]:
    """Rotate by the estimated skew; returns (image, angle_degrees)."""
    angle = estimate_skew_angle(img, limit=limit)
    if abs(angle) < 0.15:
        return img, 0.0
    # PIL rotate is CCW for positive angles; expand then re-crop void.
    rotated = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=fill)
    return auto_crop(rotated, pad=4, thr=18), angle


def enhance_old_photo(img: Image.Image, strength: float = 1.0) -> Image.Image:
    """Gentle lift for faded prints; keep strength ≤ 1.2 to avoid crunchiness."""
    if strength <= 0:
        return img
    img = ImageOps.autocontrast(img, cutoff=1 * strength)
    img = ImageEnhance.Contrast(img).enhance(1.0 + 0.08 * strength)
    img = ImageEnhance.Color(img).enhance(1.0 + 0.05 * strength)
    img = ImageEnhance.Sharpness(img).enhance(1.0 + 0.15 * strength)
    return img


def _estimate_grid_angle(
    gray: np.ndarray, lim: float = 6.0, step: float = 0.5
) -> tuple[float, float, float]:
    """Radon-like projections: angle of strongest long-line energy (0° = vertical)."""
    g = gray.astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.2)
    scale = min(1.0, 400.0 / max(hp.shape[0], hp.shape[1]))
    small = cv2.resize(hp, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    best_a, best_e = 0.0, -1.0
    cy, cx = small.shape[0] / 2.0, small.shape[1] / 2.0
    for a in np.arange(-lim, lim + 1e-9, step):
        M = cv2.getRotationMatrix2D((cx, cy), float(a), 1.0)
        rot = cv2.warpAffine(small, M, (small.shape[1], small.shape[0]))
        e = float(rot.mean(axis=0).std())
        if e > best_e:
            best_e, best_a = e, float(a)
    e_h = float(small.mean(axis=1).std())
    return best_a, best_e, e_h


def _destripe_rgb(
    rgb: np.ndarray,
    strength_v: float = 0.45,
    strength_h: float = 0.28,
    smooth_frac: float = 0.04,
    edge_guard: float = 0.06,
) -> np.ndarray:
    """Subtract slow-varying column/row bias (classic scanner destriping).

    Profiles are estimated from the interior only, and the correction is tapered
    to zero near the frame edges so dark margins / print borders don't turn into
    vertical spectral bands.
    """
    out = rgb.astype(np.float32)
    h, w = rgb.shape[:2]
    gx = max(4, int(round(w * edge_guard)))
    gy = max(4, int(round(h * edge_guard)))
    gx = min(gx, max(4, w // 5))
    gy = min(gy, max(4, h // 5))

    def _edge_taper(n: int, guard: int) -> np.ndarray:
        t = np.ones(n, np.float32)
        if guard > 0 and 2 * guard < n:
            ramp = np.linspace(0.0, 1.0, guard, dtype=np.float32)
            t[:guard] = ramp
            t[-guard:] = ramp[::-1]
        return t

    taper_x = _edge_taper(w, gx)
    taper_y = _edge_taper(h, gy)

    for c in range(3):
        ch = out[:, :, c]
        if strength_v > 0 and h > 2 * gy + 8:
            col = ch[gy : h - gy, :].mean(axis=0)
            trend = cv2.GaussianBlur(
                col.reshape(1, -1), (0, 0), max(5.0, w * smooth_frac)
            ).ravel()
            stripe = (col - trend) * taper_x
            ch = ch - strength_v * stripe[None, :]
        if strength_h > 0 and w > 2 * gx + 8:
            row = ch[:, gx : w - gx].mean(axis=1)
            trend = cv2.GaussianBlur(
                row.reshape(1, -1), (0, 0), max(5.0, h * smooth_frac)
            ).ravel()
            stripe = (row - trend) * taper_y
            ch = ch - strength_h * stripe[:, None]
        out[:, :, c] = ch
    return np.clip(out, 0, 255).astype(np.uint8)


def _detect_grid_periods(
    gray: np.ndarray, min_ac: float = 0.28, max_lag: int = 40
) -> tuple[list[int], list[int]]:
    """Vote on local autocorr periods for fine mesh / screen texture."""
    h, w = gray.shape
    boxes = [
        (h // 6, h // 3, w // 6, w // 3),
        (h // 2, 2 * h // 3, w // 2, 2 * w // 3),
        (h // 3, h // 2, w // 3, w // 2),
    ]
    votes_v: dict[int, float] = {}
    votes_h: dict[int, float] = {}
    for y0, y1, x0, x1 in boxes:
        if y1 - y0 < max_lag + 8 or x1 - x0 < max_lag + 8:
            continue
        patch = gray[y0:y1, x0:x1].astype(np.float32)
        for votes, prof in ((votes_v, patch.mean(axis=0)), (votes_h, patch.mean(axis=1))):
            hp = prof - cv2.GaussianBlur(prof.reshape(1, -1), (0, 0), 4).ravel()
            s = hp - hp.mean()
            ac = np.correlate(s, s, mode="full")[len(s) - 1 :]
            ac = ac / (ac[0] + 1e-9)
            for i in range(3, min(max_lag, len(ac) - 1)):
                if ac[i] > ac[i - 1] and ac[i] > ac[i + 1] and ac[i] >= min_ac:
                    votes[i] = votes.get(i, 0.0) + float(ac[i])

    def _top(votes: dict[int, float]) -> list[int]:
        kept: list[int] = []
        for period, _score in sorted(votes.items(), key=lambda kv: -kv[1]):
            if any(
                abs(period - k * q) < 2 or abs(k * period - q) < 2
                for q in kept
                for k in (2, 3)
            ):
                continue
            kept.append(int(period))
            if len(kept) >= 3:
                break
        return kept

    return _top(votes_v), _top(votes_h)


def _fft_notch_periods(
    rgb: np.ndarray,
    periods_v: list[int],
    periods_h: list[int],
    harmonics: int = 3,
    strength: float = 0.65,
    pad: int | None = None,
) -> np.ndarray:
    """Notch detected mesh periods in FFT domain.

    Reflect-pads before the transform so wraparound from print edges doesn't
    ring into vertical/horizontal spectral bands near the border.
    """
    if not periods_v and not periods_h:
        return rgb
    h0, w0 = rgb.shape[:2]
    if pad is None:
        longest = max([1, *periods_v, *periods_h])
        pad = int(max(48, min(160, longest * 3)))
    work = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    h, w = work.shape[:2]
    cy, cx = h // 2, w // 2
    mask = np.ones((h, w), np.float32)

    def carve(axis: str, period: int) -> None:
        nonlocal mask
        span = w if axis == "v" else h
        center = cx if axis == "v" else cy
        for k in range(1, harmonics + 1):
            freq = span / float(period) * k
            if freq >= span * 0.48:
                break
            for sign in (-1.0, 1.0):
                pos = int(round(center + sign * freq))
                if axis == "v" and 2 <= pos < w - 2:
                    x0, x1 = max(0, pos - 3), min(w, pos + 4)
                    y0, y1 = max(0, cy - 10), min(h, cy + 11)
                    xs = np.arange(x0, x1, dtype=np.float32)[None, :]
                    ys = np.arange(y0, y1, dtype=np.float32)[:, None]
                    g = np.exp(-0.5 * (((xs - pos) / 1.3) ** 2 + ((ys - cy) / 4.5) ** 2))
                    mask[y0:y1, x0:x1] *= 1.0 - strength * g
                elif axis == "h" and 2 <= pos < h - 2:
                    y0, y1 = max(0, pos - 3), min(h, pos + 4)
                    x0, x1 = max(0, cx - 10), min(w, cx + 11)
                    xs = np.arange(x0, x1, dtype=np.float32)[None, :]
                    ys = np.arange(y0, y1, dtype=np.float32)[:, None]
                    g = np.exp(-0.5 * (((ys - pos) / 1.3) ** 2 + ((xs - cx) / 4.5) ** 2))
                    mask[y0:y1, x0:x1] *= 1.0 - strength * g

    for p in periods_v:
        carve("v", p)
    for p in periods_h:
        carve("h", p)

    out = np.empty_like(work, dtype=np.float32)
    for c in range(3):
        spectrum = np.fft.fftshift(np.fft.fft2(work[:, :, c].astype(np.float32)))
        out[:, :, c] = np.fft.ifft2(np.fft.ifftshift(spectrum * mask)).real
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out[pad : pad + h0, pad : pad + w0]


def _fft_notch_detail(
    rgb: np.ndarray,
    periods_v: list[int],
    periods_h: list[int],
    sigma: float = 1.8,
    harmonics: int = 3,
    strength: float = 0.65,
) -> np.ndarray:
    """Notch mesh periods on the detail layer only.

    Hard edges / print borders live in the Gaussian base, so FFT notches can't
    Gibbs-ring them into spectral bands. The fine scanner/paper mesh stays in
    the residual and is what we remove.
    """
    if not periods_v and not periods_h:
        return rgb
    base = cv2.GaussianBlur(rgb, (0, 0), sigma)
    detail = rgb.astype(np.float32) - base.astype(np.float32)
    # Bias into uint8 range for the shared notch helper, then restore.
    shifted = np.clip(detail + 128.0, 0, 255).astype(np.uint8)
    filtered = _fft_notch_periods(
        shifted, periods_v, periods_h, harmonics=harmonics, strength=strength
    )
    detail2 = filtered.astype(np.float32) - 128.0
    return np.clip(base.astype(np.float32) + detail2, 0, 255).astype(np.uint8)


def _morph_smooth_grid(rgb: np.ndarray, k: int = 1, strength: float = 0.45) -> np.ndarray:
    """Lift thin dark lines / push thin bright lines via directional top/black-hat."""
    out = rgb.astype(np.float32)
    se_h = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * k + 1, 1))
    se_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2 * k + 1))
    for c in range(3):
        ch = rgb[:, :, c]
        corr = (
            cv2.morphologyEx(ch, cv2.MORPH_BLACKHAT, se_h).astype(np.float32)
            + cv2.morphologyEx(ch, cv2.MORPH_BLACKHAT, se_v).astype(np.float32)
            - cv2.morphologyEx(ch, cv2.MORPH_TOPHAT, se_h).astype(np.float32)
            - cv2.morphologyEx(ch, cv2.MORPH_TOPHAT, se_v).astype(np.float32)
        )
        out[:, :, c] = np.clip(ch.astype(np.float32) + strength * np.clip(corr, -30, 30), 0, 255)
    return out.astype(np.uint8)


def _feather_edges(original: np.ndarray, processed: np.ndarray, guard: float = 0.05) -> np.ndarray:
    """Blend processed result back to the original near frame borders."""
    h, w = original.shape[:2]
    gx = max(3, int(round(w * guard)))
    gy = max(3, int(round(h * guard)))
    gx = min(gx, max(3, w // 6))
    gy = min(gy, max(3, h // 6))
    weight = np.ones((h, w), np.float32)
    if gx > 0:
        ramp = np.linspace(0.0, 1.0, gx, dtype=np.float32)
        weight[:, :gx] *= ramp[None, :]
        weight[:, -gx:] *= ramp[None, ::-1]
    if gy > 0:
        ramp = np.linspace(0.0, 1.0, gy, dtype=np.float32)
        weight[:gy, :] *= ramp[:, None]
        weight[-gy:, :] *= ramp[::-1, None]
    w3 = weight[..., None]
    out = original.astype(np.float32) * (1.0 - w3) + processed.astype(np.float32) * w3
    return np.clip(out, 0, 255).astype(np.uint8)


def smooth_grid_lines(img: Image.Image) -> tuple[Image.Image, dict]:
    """Remove / soften periodic grid & scanner striping on a print.

    Uses a Radon-like projection to find the dominant line angle, then:
      1) edge-safe column/row destriping
      2) period-aware directional morphological cleanup (spatial; no FFT)
      3) feather back to the original near borders

    FFT notches are avoided here — they introduce Gibbs / spectral bands near
    hard print edges and on flat water. Helpers remain for experimentation.

    No-ops when OpenCV is missing or line energy is weak (clean photos).
    """
    info: dict = {
        "applied": False,
        "angle": 0.0,
        "ev": 0.0,
        "eh": 0.0,
        "periods_v": [],
        "periods_h": [],
    }
    if not _HAS_CV2:
        return img, info

    rgb = np.asarray(img.convert("RGB"))
    if rgb.shape[0] < 48 or rgb.shape[1] < 48:
        return img, info

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    angle, e_v, e_h = _estimate_grid_angle(gray)
    info.update(angle=angle, ev=e_v, eh=e_h)
    # Skip when neither orientation shows structured line energy.
    if max(e_v, e_h) < 0.9:
        return img, info

    original = rgb
    work = rgb
    h, w = work.shape[:2]
    rotated = abs(angle) > 0.35
    if rotated:
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle, 1.0)
        work = cv2.warpAffine(
            work, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        original = work.copy()

    do_v = e_v >= e_h * 0.55
    do_h = e_h >= e_v * 0.55
    periods_v, periods_h = _detect_grid_periods(cv2.cvtColor(work, cv2.COLOR_RGB2GRAY))
    if not do_v:
        periods_v = []
    if not do_h:
        periods_h = []
    info["periods_v"] = periods_v
    info["periods_h"] = periods_h

    work = _destripe_rgb(
        work,
        strength_v=0.28 if do_v else 0.0,
        strength_h=0.18 if do_h else 0.0,
    )
    # Spatial only: FFT notches ring at hard edges into vertical spectral bands.
    work = _morph_smooth_grid(work, k=1, strength=0.50)
    if periods_v or periods_h:
        # Slightly wider SE when a clear fine period was detected.
        fine = min([*periods_v, *periods_h])
        if 3 <= fine <= 12:
            work = _morph_smooth_grid(work, k=2, strength=0.30)
    work = _feather_edges(original, work, guard=0.08)

    if rotated:
        Minv = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        work = cv2.warpAffine(
            work, Minv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )

    info["applied"] = True
    return Image.fromarray(work), info


def _resize_hq(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    tw, th = size
    if _HAS_CV2:
        arr = np.asarray(img.convert("RGB"))
        out = cv2.resize(arr, (tw, th), interpolation=cv2.INTER_LANCZOS4)
        return Image.fromarray(out)
    return img.resize((tw, th), Image.Resampling.LANCZOS)


def upscale_image(
    img: Image.Image,
    size: tuple[int, int],
    sharpen: float | None = None,
) -> Image.Image:
    """High-quality upscale for small prints.

    Uses stepwise ≤2× interpolation (OpenCV Lanczos4 when available, else
    Pillow Lanczos) instead of one big jump, then a mild unsharp pass when
    the enlargement is substantial.
    """
    tw, th = size
    w, h = img.size
    if tw < 1 or th < 1 or w < 1 or h < 1:
        return img
    if (tw, th) == (w, h):
        return img

    scale = max(tw / w, th / h)
    # Downscale: single high-quality step is fine.
    if scale <= 1.0:
        return _resize_hq(img, (tw, th))

    if sharpen is None:
        sharpen = 0.0 if scale < 1.35 else min(1.35, 0.35 + 0.25 * (scale - 1.0))

    work = img.convert("RGB")
    cw, ch = work.size
    while cw * 2 < tw or ch * 2 < th:
        nw = min(tw, max(cw + 1, int(cw * 2)))
        nh = min(th, max(ch + 1, int(ch * 2)))
        work = _resize_hq(work, (nw, nh))
        cw, ch = work.size
    if (cw, ch) != (tw, th):
        work = _resize_hq(work, (tw, th))

    if sharpen > 0.01:
        radius = 1.2 if scale < 2.5 else 1.6
        work = work.filter(
            ImageFilter.UnsharpMask(radius=radius, percent=int(80 * sharpen), threshold=2)
        )
    return work


def _yunet_model_path() -> Path | None:
    path = Path(__file__).resolve().parent / "data" / "face_detection_yunet_2023mar.onnx"
    return path if path.exists() else None


def detect_faces(img: Image.Image) -> list[tuple[int, int, int, int]]:
    """Return face boxes as (x, y, w, h). Uses OpenCV YuNet when available."""
    if not _HAS_CV2 or not hasattr(cv2, "FaceDetectorYN_create"):
        return []
    model = _yunet_model_path()
    if model is None:
        return []

    full = img.convert("RGB")
    fw, fh = full.size
    if fw < 8 or fh < 8:
        return []

    # Cap working size for speed on large scans.
    max_side = 960
    scale = min(1.0, max_side / max(fw, fh))
    work_w = max(32, int(fw * scale))
    work_h = max(32, int(fh * scale))
    work = full if scale >= 0.999 else full.resize((work_w, work_h), Image.Resampling.BILINEAR)
    bgr = cv2.cvtColor(np.asarray(work), cv2.COLOR_RGB2BGR)

    detector = cv2.FaceDetectorYN_create(
        str(model),
        "",
        (work_w, work_h),
        score_threshold=0.55,
        nms_threshold=0.3,
        top_k=50,
    )
    detector.setInputSize((work_w, work_h))
    _, faces = detector.detect(bgr)
    if faces is None or len(faces) == 0:
        return []

    inv = 1.0 / scale
    boxes: list[tuple[int, int, int, int]] = []
    for row in faces:
        x, y, w, h = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
        # Clamp to image bounds before scaling back.
        x = max(0.0, x)
        y = max(0.0, y)
        w = max(1.0, min(w, work_w - x))
        h = max(1.0, min(h, work_h - y))
        boxes.append(
            (
                int(x * inv),
                int(y * inv),
                max(1, int(w * inv)),
                max(1, int(h * inv)),
            )
        )
    return boxes


def detect_subjects(img: Image.Image, max_subjects: int = 6) -> list[tuple[int, int, int, int]]:
    """Find salient subject boxes (boats, buildings, etc.) when faces are absent.

    Uses a blurred edge-energy map and connected components. Returns (x, y, w, h)
    in full-image coordinates, largest/most energetic first.
    """
    full = img.convert("RGB")
    fw, fh = full.size
    if fw < 32 or fh < 32:
        return []

    max_side = 720
    scale = min(1.0, max_side / max(fw, fh))
    work_w = max(32, int(fw * scale))
    work_h = max(32, int(fh * scale))
    small = full if scale >= 0.999 else full.resize((work_w, work_h), Image.Resampling.BILINEAR)
    gray = np.asarray(small.convert("L"), dtype=np.float32)

    if _HAS_CV2:
        blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        energy = cv2.magnitude(gx, gy)
        energy = cv2.GaussianBlur(energy, (0, 0), 3.5)
    else:
        # Numpy fallback: simple gradient magnitude + box blur.
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        energy = np.hypot(gx, gy)
        # crude blur
        k = 9
        pad = k // 2
        padded = np.pad(energy, pad, mode="edge")
        integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        energy = (
            integral[k:, k:]
            - integral[:-k, k:]
            - integral[k:, :-k]
            + integral[:-k, :-k]
        ) / (k * k)

    thr = float(np.percentile(energy, 82))
    if thr <= 1e-3:
        return []
    mask = (energy >= thr).astype(np.uint8) * 255
    # Ignore scanner-edge / water-grain bands that dominate energy maps.
    margin = max(4, int(min(work_w, work_h) * 0.06))
    mask[:margin, :] = 0
    mask[-margin:, :] = 0
    mask[:, :margin] = 0
    mask[:, -margin:] = 0

    boxes: list[tuple[int, int, int, int, float]] = []
    if _HAS_CV2:
        # Close small gaps so a boat+sail stays one blob.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        n, labels, stats, _cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < 0.002 * work_w * work_h:
                continue
            if bw < 10 or bh < 10:
                continue
            if bw * bh > 0.90 * work_w * work_h:
                continue
            score = float(energy[labels == i].mean()) * np.sqrt(area)
            boxes.append((x, y, bw, bh, score))
    else:
        # Flood-fill components without OpenCV.
        visited = np.zeros_like(mask, dtype=bool)
        ys, xs = np.where(mask > 0)
        for y0, x0 in zip(ys, xs):
            if visited[y0, x0]:
                continue
            stack = [(y0, x0)]
            visited[y0, x0] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < work_h and 0 <= nx < work_w and not visited[ny, nx] and mask[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(cells) < 0.002 * work_w * work_h:
                continue
            yy = [c[0] for c in cells]
            xx = [c[1] for c in cells]
            x, y = min(xx), min(yy)
            bw, bh = max(xx) - x + 1, max(yy) - y + 1
            if bw * bh > 0.90 * work_w * work_h:
                continue
            score = float(sum(energy[cy, cx] for cy, cx in cells))
            boxes.append((x, y, bw, bh, score))

    if not boxes:
        # Fallback: single window around energy centroid / peak.
        flat = energy.ravel()
        peak = int(np.argmax(flat))
        py, px = divmod(peak, work_w)
        bw = max(32, work_w // 3)
        bh = max(32, work_h // 3)
        x = max(0, px - bw // 2)
        y = max(0, py - bh // 2)
        bw = min(bw, work_w - x)
        bh = min(bh, work_h - y)
        boxes = [(x, y, bw, bh, float(energy[py, px]))]

    boxes.sort(key=lambda b: b[4], reverse=True)
    # Prefer keeping a fleet/group together: union of the strongest blobs.
    top = boxes[: min(max_subjects, len(boxes))]
    if len(top) >= 2:
        ux0 = min(b[0] for b in top)
        uy0 = min(b[1] for b in top)
        ux1 = max(b[0] + b[2] for b in top)
        uy1 = max(b[1] + b[3] for b in top)
        union_score = sum(b[4] for b in top) * 1.25
        boxes.insert(0, (ux0, uy0, ux1 - ux0, uy1 - uy0, union_score))

    # Horizontal subject band (regattas, skylines): strongest energy ridge,
    # ignoring outer margins where grain/scanner edges spike.
    row_e = energy.mean(axis=1).copy()
    m = max(4, int(work_h * 0.08))
    row_e[:m] *= 0.25
    row_e[-m:] = 0.0
    peak_r = int(np.argmax(row_e))
    # Expand while rows stay relatively energetic.
    band_thr = float(row_e[peak_r] * 0.40)
    r0 = peak_r
    while r0 > 0 and row_e[r0 - 1] >= band_thr:
        r0 -= 1
    r1 = peak_r
    while r1 + 1 < work_h and row_e[r1 + 1] >= band_thr:
        r1 += 1
    band_h = r1 - r0 + 1
    if band_h >= max(12, work_h // 14) and band_h <= int(work_h * 0.70):
        # Find horizontal span of energy inside the band.
        band = energy[r0 : r1 + 1]
        col_e = band.mean(axis=0)
        col_thr = float(np.percentile(col_e, 50))
        cols = np.where(col_e >= col_thr)[0]
        if cols.size >= 8:
            c0, c1 = int(cols[0]), int(cols[-1]) + 1
            pad_y = max(4, band_h // 5)
            y0 = max(0, r0 - pad_y)
            y1 = min(work_h, r1 + 1 + pad_y)
            score = float(band[:, c0:c1].sum())
            boxes.insert(0, (c0, y0, c1 - c0, y1 - y0, score * 1.15))

    inv = 1.0 / scale
    out: list[tuple[int, int, int, int]] = []
    for x, y, bw, bh, _score in boxes[: max_subjects + 1]:
        # Slight expand so masts / hulls aren't clipped.
        pad_x = int(bw * 0.10)
        pad_y = int(bh * 0.10)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(work_w, x + bw + pad_x)
        y1 = min(work_h, y + bh + pad_y)
        out.append(
            (
                int(x0 * inv),
                int(y0 * inv),
                max(1, int((x1 - x0) * inv)),
                max(1, int((y1 - y0) * inv)),
            )
        )
    return out


def detect_crop_anchors(img: Image.Image) -> tuple[list[tuple[int, int, int, int]], str]:
    """Prefer faces; fall back to salient subjects for crop framing."""
    faces = detect_faces(img)
    if faces:
        return faces, "faces"
    subjects = detect_subjects(img)
    if subjects:
        return subjects, "subjects"
    return [], "none"


def _face_crop_score(
    faces: list[tuple[int, int, int, int]],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> tuple[int, float, float]:
    """Score a crop window: (# fully inside, area inside, closeness to center of faces)."""
    if not faces:
        return (0, 0.0, 0.0)
    full = 0
    area = 0.0
    # Prefer windows whose center is near the centroid of all faces.
    fcx = sum(x + w / 2 for x, y, w, h in faces) / len(faces)
    fcy = sum(y + h / 2 for x, y, w, h in faces) / len(faces)
    wcx, wcy = (x0 + x1) / 2, (y0 + y1) / 2
    for x, y, w, h in faces:
        ix0, iy0 = max(x0, x), max(y0, y)
        ix1, iy1 = min(x1, x + w), min(y1, y + h)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        inter = (ix1 - ix0) * (iy1 - iy0)
        face_area = w * h
        area += inter
        if inter >= 0.85 * face_area:
            full += 1
    # Smaller distance is better → negative distance as tertiary key.
    dist = -((wcx - fcx) ** 2 + (wcy - fcy) ** 2)
    return (full, area, dist)


def best_crop_origin(
    img_w: int,
    img_h: int,
    crop_w: int,
    crop_h: int,
    faces: list[tuple[int, int, int, int]],
    axis: str,
) -> int:
    """Pick crop origin along axis ('x' or 'y') maximizing retained faces."""
    if axis == "x":
        max_origin = max(0, img_w - crop_w)
        center = max_origin // 2
        if not faces or max_origin == 0:
            return center

        def window(o: int) -> tuple[int, int, int, int]:
            # When sliding X only, evaluate against full height unless crop_h is tighter.
            y0 = max(0, (img_h - crop_h) // 2)
            return (o, y0, o + crop_w, min(img_h, y0 + crop_h))

    else:
        max_origin = max(0, img_h - crop_h)
        center = max_origin // 2
        if not faces or max_origin == 0:
            return center

        def window(o: int) -> tuple[int, int, int, int]:
            x0 = max(0, (img_w - crop_w) // 2)
            return (x0, o, min(img_w, x0 + crop_w), o + crop_h)

    # Candidate positions: center, face centers, and union bounds of face groups.
    candidates = {center, 0, max_origin}
    for x, y, w, h in faces:
        if axis == "x":
            # Place face near center / keep face fully inside.
            candidates.add(max(0, min(max_origin, x + w // 2 - crop_w // 2)))
            candidates.add(max(0, min(max_origin, x)))
            candidates.add(max(0, min(max_origin, x + w - crop_w)))
        else:
            # Bias slightly upward so foreheads/hairline stay in frame.
            candidates.add(max(0, min(max_origin, y + h // 2 - int(crop_h * 0.45))))
            candidates.add(max(0, min(max_origin, y)))
            candidates.add(max(0, min(max_origin, y + h - crop_h)))

    # Also try covering as many faces as possible by sliding over face-span.
    if len(faces) >= 2:
        if axis == "x":
            left = min(x for x, y, w, h in faces)
            right = max(x + w for x, y, w, h in faces)
            candidates.add(max(0, min(max_origin, left)))
            candidates.add(max(0, min(max_origin, right - crop_w)))
            candidates.add(max(0, min(max_origin, (left + right) // 2 - crop_w // 2)))
        else:
            top = min(y for x, y, w, h in faces)
            bot = max(y + h for x, y, w, h in faces)
            candidates.add(max(0, min(max_origin, top)))
            candidates.add(max(0, min(max_origin, bot - crop_h)))
            candidates.add(max(0, min(max_origin, (top + bot) // 2 - crop_h // 2)))

    best_o = center
    best_score = (-1, -1.0, float("-inf"))
    for o in sorted(candidates):
        o = int(max(0, min(max_origin, o)))
        score = _face_crop_score(faces, *window(o))
        if score > best_score:
            best_score = score
            best_o = o
    return best_o


def best_cover_origin(
    img_w: int,
    img_h: int,
    crop_w: int,
    crop_h: int,
    faces: list[tuple[int, int, int, int]],
) -> tuple[int, int]:
    """Pick a 2D crop origin maximizing faces kept inside the portal window."""
    max_x = max(0, img_w - crop_w)
    max_y = max(0, img_h - crop_h)
    cx, cy = max_x // 2, max_y // 2
    if not faces or (max_x == 0 and max_y == 0):
        return cx, cy

    xs: set[int] = {cx, 0, max_x}
    ys: set[int] = {cy, 0, max_y}
    for x, y, w, h in faces:
        xs.add(max(0, min(max_x, x + w // 2 - crop_w // 2)))
        xs.add(max(0, min(max_x, x)))
        xs.add(max(0, min(max_x, x + w - crop_w)))
        ys.add(max(0, min(max_y, y + h // 2 - int(crop_h * 0.45))))
        ys.add(max(0, min(max_y, y)))
        ys.add(max(0, min(max_y, y + h - crop_h)))
    if len(faces) >= 2:
        left = min(x for x, y, w, h in faces)
        right = max(x + w for x, y, w, h in faces)
        top = min(y for x, y, w, h in faces)
        bot = max(y + h for x, y, w, h in faces)
        xs.add(max(0, min(max_x, left)))
        xs.add(max(0, min(max_x, right - crop_w)))
        xs.add(max(0, min(max_x, (left + right) // 2 - crop_w // 2)))
        ys.add(max(0, min(max_y, top)))
        ys.add(max(0, min(max_y, bot - crop_h)))
        ys.add(max(0, min(max_y, (top + bot) // 2 - crop_h // 2)))

    best = (cx, cy)
    best_score = (-1, -1.0, float("-inf"))
    for xo in sorted(xs):
        for yo in sorted(ys):
            score = _face_crop_score(faces, xo, yo, xo + crop_w, yo + crop_h)
            if score > best_score:
                best_score = score
                best = (xo, yo)
    return best


def _remap_faces(
    faces: list[tuple[int, int, int, int]],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> list[tuple[int, int, int, int]]:
    """Shift face boxes into a crop window; drop faces with no overlap."""
    out: list[tuple[int, int, int, int]] = []
    for x, y, w, h in faces:
        ix0, iy0 = max(x0, x), max(y0, y)
        ix1, iy1 = min(x1, x + w), min(y1, y + h)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        out.append((ix0 - x0, iy0 - y0, ix1 - ix0, iy1 - iy0))
    return out


def crop_to_aspect(
    img: Image.Image,
    target_ar: float,
    threshold: float = 0.03,
    faces: list[tuple[int, int, int, int]] | None = None,
) -> tuple[Image.Image, bool, int, list[tuple[int, int, int, int]]]:
    """Crop to target aspect ratio. Uses faces when provided to keep people in frame.

    Returns (image, did_crop, faces_kept_fully, faces_in_crop_coords).
    """
    w, h = img.size
    face_list = list(faces) if faces else []
    if w < 1 or h < 1 or target_ar <= 0:
        return img, False, 0, face_list
    ar = w / h
    if abs(ar - target_ar) / target_ar <= threshold:
        kept = _face_crop_score(face_list, 0, 0, w, h)[0] if face_list else 0
        return img, False, kept, face_list

    if ar > target_ar:
        nw = max(1, int(round(h * target_ar)))
        x0 = best_crop_origin(w, h, nw, h, face_list, "x")
        cropped = img.crop((x0, 0, x0 + nw, h))
        kept = _face_crop_score(face_list, x0, 0, x0 + nw, h)[0]
        return cropped, True, kept, _remap_faces(face_list, x0, 0, x0 + nw, h)
    nh = max(1, int(round(w / target_ar)))
    y0 = best_crop_origin(w, h, w, nh, face_list, "y")
    cropped = img.crop((0, y0, w, y0 + nh))
    kept = _face_crop_score(face_list, 0, y0, w, y0 + nh)[0]
    return cropped, True, kept, _remap_faces(face_list, 0, y0, w, y0 + nh)


def has_white_border(
    img: Image.Image,
    band_frac: float = 0.07,
    min_band_px: int = 5,
    max_band_frac: float = 0.22,
    luma_thr: int = 155,
    chroma_thr: int = 32,
    min_sides: int = 3,
) -> bool:
    """Detect classic white/cream print borders on most sides of the print.

    Measures bands relative to the print's own non-black bounds so a tilted
    print with scanner void in the AABB corners still counts.
    """
    rgb = np.asarray(img.convert("RGB"))
    h, w = rgb.shape[:2]
    if w < 48 or h < 48:
        return False

    luma = rgb.mean(axis=2)
    chroma = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    # Print = anything clearly above the black scanner bed.
    on_print = luma > 22
    ys, xs = np.where(on_print)
    if xs.size < 64:
        return False
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pw, ph = x1 - x0, y1 - y0
    if pw < 40 or ph < 40:
        return False

    band_x = int(max(min_band_px, min(pw * max_band_frac, pw * band_frac)))
    band_y = int(max(min_band_px, min(ph * max_band_frac, ph * band_frac)))
    if band_x * 2 >= pw or band_y * 2 >= ph:
        return False

    pl = luma[y0:y1, x0:x1]
    pc = chroma[y0:y1, x0:x1]
    inner = pl[band_y : ph - band_y, band_x : pw - band_x]
    if inner.size < 16:
        return False
    inner_med = float(np.median(inner))

    def side_is_border(strip_luma: np.ndarray, strip_chroma: np.ndarray) -> bool:
        if strip_luma.size < 8:
            return False
        # Ignore residual bed pixels inside the print AABB (tilted corners).
        usable = strip_luma > 22
        if float(usable.mean()) < 0.35:
            return False
        bright = usable & (strip_luma >= luma_thr) & (strip_chroma <= chroma_thr)
        frac = float(bright[usable].mean()) if usable.any() else 0.0
        med = float(np.median(strip_luma[usable])) if usable.any() else 0.0
        return frac >= 0.50 and med >= luma_thr - 10 and med >= inner_med + 15

    sides = (
        side_is_border(pl[:band_y, :], pc[:band_y, :]),
        side_is_border(pl[ph - band_y :, :], pc[ph - band_y :, :]),
        side_is_border(pl[:, :band_x], pc[:, :band_x]),
        side_is_border(pl[:, pw - band_x :], pc[:, pw - band_x :]),
    )
    return sum(1 for s in sides if s) >= min_sides


def fit_to_portal(
    img: Image.Image,
    portal: tuple[int, int],
    matte: tuple[int, int, int] = DEFAULT_MATTE,
    mode: str = "contain",
    crop_to_portal: bool = False,
    crop_threshold: float = 0.03,
    face_aware: bool = True,
    keep_white_border: bool | None = True,
) -> tuple[Image.Image, bool, dict]:
    """Place img into a fixed portal.

    Returns (frame, cropped_for_aspect, info).
    If keep_white_border is True (default) and a classic white print border is
    detected, skip aspect cropping and letterbox-upscale to preserve AR + border.
    Pass False to always allow --crop-to-portal; None treats as True.
    """
    pw, ph = portal
    cropped = False
    info = {
        "detected": 0,
        "kept": 0,
        "used": False,
        "white_border": False,
        "kept_aspect": False,
        "anchor": "none",
    }
    anchors: list[tuple[int, int, int, int]] = []
    # Drop residual bed before border/face decisions.
    g0 = np.asarray(img.convert("L"))
    tx0, ty0, tx1, ty1 = trim_scanner_bed(g0)
    if (tx1 - tx0) >= 32 and (ty1 - ty0) >= 32 and (tx0, ty0, tx1, ty1) != (0, 0, g0.shape[1], g0.shape[0]):
        img = img.crop((tx0, ty0, tx1, ty1))

    border = has_white_border(img)
    info["white_border"] = border
    preserve = bool(keep_white_border) and border

    need_anchors = face_aware and (crop_to_portal or mode == "cover") and not preserve
    if need_anchors:
        anchors, kind = detect_crop_anchors(img)
        info["detected"] = len(anchors)
        info["used"] = bool(anchors)
        info["anchor"] = kind

    if preserve and (crop_to_portal or mode == "cover"):
        # Drop scanner-void corners around a tilted bordered print, then
        # upscale to fit portal while keeping native AR + decorative border.
        luma = np.asarray(img.convert("L"))
        on = np.asarray(luma) > 22
        ys, xs = np.where(on)
        if xs.size >= 64:
            bx0, bx1 = int(xs.min()), int(xs.max()) + 1
            by0, by1 = int(ys.min()), int(ys.max()) + 1
            if (bx1 - bx0) >= 32 and (by1 - by0) >= 32:
                img = img.crop((bx0, by0, bx1, by1))
        mode = "contain"
        info["kept_aspect"] = True
        crop_to_portal = False
    elif crop_to_portal and pw > 0 and ph > 0:
        img, cropped, kept, anchors = crop_to_aspect(
            img, pw / ph, threshold=crop_threshold, faces=anchors if face_aware else None
        )
        info["kept"] = kept
        if cropped or abs((img.size[0] / max(1, img.size[1])) - (pw / ph)) <= crop_threshold:
            mode = "cover"

    canvas = Image.new("RGB", (pw, ph), matte)
    w, h = img.size
    if w < 1 or h < 1:
        return canvas, cropped, info

    if mode == "cover":
        scale = max(pw / w, ph / h)
    else:
        scale = min(pw / w, ph / h)

    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = upscale_image(img, (nw, nh))

    if mode == "cover":
        # Prefer face/subject-aware origin when the cover window still crops.
        if face_aware and anchors and (nw > pw or nh > ph):
            sx, sy = nw / w, nh / h
            scaled = [
                (int(x * sx), int(y * sy), max(1, int(aw * sx)), max(1, int(ah * sy)))
                for x, y, aw, ah in anchors
            ]
            x0, y0 = best_cover_origin(nw, nh, pw, ph, scaled)
            info["kept"] = _face_crop_score(scaled, x0, y0, x0 + pw, y0 + ph)[0]
        else:
            x0 = max(0, (nw - pw) // 2)
            y0 = max(0, (nh - ph) // 2)
        frame = resized.crop((x0, y0, x0 + pw, y0 + ph))
        # Peel residual bed hairlines only (low-mean strips), then refill portal.
        fg = np.asarray(frame.convert("L"))
        fx0, fy0, fx1, fy1 = trim_scanner_bed(fg, thr=32, dark_frac=0.80, mean_cap=12.0, strip=1, pad=0)
        peeled_w, peeled_h = fx1 - fx0, fy1 - fy0
        if (fx0, fy0, fx1, fy1) != (0, 0, pw, ph) and peeled_w >= int(pw * 0.92) and peeled_h >= int(ph * 0.92):
            frame = upscale_image(frame.crop((fx0, fy0, fx1, fy1)), (pw, ph), sharpen=0.0)
        return frame, cropped, info

    x = (pw - nw) // 2
    y = (ph - nh) // 2
    canvas.paste(resized, (x, y))
    return canvas, cropped, info


def next_index(out_dir: Path) -> int:
    existing = sorted(out_dir.glob("photo_*.jpg")) + sorted(out_dir.glob("photo_*.jpeg"))
    if not existing:
        return 1
    nums = []
    for p in existing:
        stem = p.stem  # photo_001
        try:
            nums.append(int(stem.split("_")[-1]))
        except ValueError:
            continue
    return (max(nums) + 1) if nums else 1


def process_raw(
    raw: bytes,
    out_path: Path,
    portal: tuple[int, int],
    matte: tuple[int, int, int],
    y_scale: float | str,
    fit: str,
    enhance: float,
    rotate: str,
    deskew_limit: float = DEFAULT_DESKEW_LIMIT,
    crop_to_portal: bool = False,
    crop_threshold: float = 0.03,
    face_aware: bool = True,
    keep_white_border: bool = True,
    smooth_grid: bool = DEFAULT_SMOOTH_GRID,
) -> tuple[int, int, float, bool, dict]:
    img = raw_to_rgb_image(raw, y_scale=y_scale)
    img = auto_crop(img)
    # Deskew while the print still has detectable borders, before 90° orient.
    img, skew = deskew(img, limit=deskew_limit, fill=(0, 0, 0))
    # Peel black corners introduced by rotation fill.
    g = np.asarray(img.convert("L"))
    bx0, by0, bx1, by1 = trim_scanner_bed(g, thr=26, pad=1)
    if (bx1 - bx0) >= 32 and (by1 - by0) >= 32 and (bx0, by0, bx1, by1) != (0, 0, g.shape[1], g.shape[0]):
        img = img.crop((bx0, by0, bx1, by1))
    # Bordered prints keep native AR — don't force landscape for the portal.
    prefer_landscape = not (keep_white_border and has_white_border(img))
    if rotate == "auto":
        img = auto_orient(img, prefer_landscape=prefer_landscape)
    elif rotate == "90":
        img = img.transpose(Image.Transpose.ROTATE_270)
    elif rotate == "180":
        img = img.transpose(Image.Transpose.ROTATE_180)
    elif rotate == "270":
        img = img.transpose(Image.Transpose.ROTATE_90)
    img = enhance_old_photo(img, strength=enhance)
    grid_info: dict = {"applied": False}
    if smooth_grid:
        # Descreen at native print res before portal upscale amplifies the mesh.
        img, grid_info = smooth_grid_lines(img)
    frame, ar_cropped, face_info = fit_to_portal(
        img,
        portal,
        matte=matte,
        mode=fit,
        crop_to_portal=crop_to_portal,
        crop_threshold=crop_threshold,
        face_aware=face_aware,
        keep_white_border=keep_white_border,
    )
    face_info["grid"] = grid_info
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.save(out_path, "JPEG", quality=92, optimize=True, progressive=True)
    return frame.size[0], frame.size[1], skew, ar_cropped, face_info


def scan_one(bot: Bot, verbose: bool = False) -> bytes:
    prepare_scan(bot)
    raw = read_image(bot, total_bytes=EXPECTED_IMAGE_BYTES, start_offset=None)
    stop_scan(bot)
    return raw


def apply_old_photo_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Apply recommended vintage-print settings without clobbering explicit flags."""
    preset = OLD_PHOTO_PRESET
    # Enable portal fill + face framing; white-border logic still overrides crop when needed.
    args.crop_to_portal = True
    if args.enhance == 1.0:
        args.enhance = float(preset["enhance"])
    if args.portal == "1920x1080":
        args.portal = str(preset["portal"])
    if args.matte == "28,26,24":
        args.matte = str(preset["matte"])
    if args.fit == "contain":
        args.fit = str(preset["fit"])
    if args.yscale == "auto":
        args.yscale = str(preset["yscale"])
    if args.rotate == "auto":
        args.rotate = str(preset["rotate"])
    # Deskew stays on unless --no-deskew; bump to preset limit if left at default.
    if not args.no_deskew and float(args.deskew) == DEFAULT_DESKEW_LIMIT:
        args.deskew = float(preset["deskew"])
    if float(args.density) == DEFAULT_DENSITY:
        args.density = float(preset["density"])
    # Preset assumes face-aware + border keep; honor explicit opt-outs.
    return args


def main() -> int:
    ap = argparse.ArgumentParser(
        description="PS100 photo → fixed slideshow portal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
recommended for old photos:
  %(prog)s --old-photos --out ~/Pictures/album
      crop-to-portal + face-aware framing, keep white-border AR, auto deskew,
      gentle contrast/color lift (enhance 1.15), 1080p portal @ 2× density,
      charcoal matte

  useful tweaks:
      --enhance 1.3          stronger lift for badly faded prints
      --yscale 0.85          if auto looks too flat or too tall
      --portal 4:3           classic slideshow ratio
      --density 1            native portal pixels (no zoom headroom)
      --density 3            more zoom headroom (heavier files)
      --no-keep-border       force 16:9 fill even on bordered prints
      --no-face-crop         center crop instead of face-aware
      --smooth-grid          soften scanner/paper mesh (off unless set)
      --deskew 6             allow a bit more skew correction
""",
    )
    ap.add_argument(
        "--old-photos",
        action="store_true",
        help="recommended vintage-print treatment (crop-to-portal, faces, "
        "keep white borders, deskew, enhance 1.15)",
    )
    ap.add_argument("--out", default="~/Pictures/ps100-album", help="output album directory")
    ap.add_argument("--portal", default="1920x1080", help="WxH or 1080p/4:3/3:2/1:1")
    ap.add_argument(
        "--density",
        type=float,
        default=DEFAULT_DENSITY,
        help=f"render scale vs portal for zoom headroom (default {DEFAULT_DENSITY:g}; "
        "2→3840x2160 from 1080p; use 1 for 1:1 portal pixels)",
    )
    ap.add_argument("--matte", default="28,26,24", help="letterbox RGB as R,G,B")
    ap.add_argument("--fit", choices=("contain", "cover"), default="contain", help="contain=pad, cover=crop")
    ap.add_argument(
        "--crop-to-portal",
        action="store_true",
        help="if photo aspect differs from portal, crop to portal AR then fill (no letterbox); "
        "prefers keeping detected faces when OpenCV is available",
    )
    ap.add_argument(
        "--crop-threshold",
        type=float,
        default=0.03,
        help="min relative AR difference to trigger --crop-to-portal (default 0.03 = 3%%)",
    )
    ap.add_argument(
        "--no-face-crop",
        action="store_true",
        help="when cropping, use center crop instead of face/subject-aware framing",
    )
    ap.add_argument(
        "--no-keep-border",
        action="store_true",
        help="allow aspect-crop even when a classic white print border is detected",
    )
    ap.add_argument("--yscale", default="auto", help="auto or numeric scale")
    ap.add_argument("--enhance", type=float, default=1.0, help="0=off, 1=default lift for old prints")
    ap.add_argument("--rotate", choices=("auto", "none", "90", "180", "270"), default="auto")
    ap.add_argument(
        "--deskew",
        type=float,
        nargs="?",
        const=DEFAULT_DESKEW_LIMIT,
        default=DEFAULT_DESKEW_LIMIT,
        help=f"max auto-deskew degrees (default {DEFAULT_DESKEW_LIMIT}; 0 disables)",
    )
    ap.add_argument("--no-deskew", action="store_true", help="disable small-angle deskew")
    ap.add_argument(
        "--smooth-grid",
        action="store_true",
        help="soften scanner/paper grid lines (Radon + spatial morph; off unless set)",
    )
    ap.add_argument("--count", type=int, default=0, help="scan N photos then stop (0=interactive)")
    ap.add_argument("--from-raw", action="append", default=[], help="process existing .raw (repeatable)")
    ap.add_argument("--start", type=int, default=0, help="starting photo_NNN index (0=auto)")
    ap.add_argument("--scan", action="store_true", help="enter live scan mode after --from-raw")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if args.old_photos:
        args = apply_old_photo_preset(args)

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    portal = parse_portal(args.portal)
    density = max(1.0, float(args.density))
    render_portal = render_portal_size(portal, density)
    matte = parse_rgb(args.matte)
    deskew_limit = 0.0 if args.no_deskew else float(args.deskew)
    idx = args.start if args.start > 0 else next_index(out_dir)

    face_aware = not args.no_face_crop
    keep_border = not args.no_keep_border
    smooth_grid = bool(args.smooth_grid)
    print(f"album:  {out_dir}")
    preset_note = "  preset=old-photos" if args.old_photos else ""
    print(
        f"portal: {portal[0]}x{portal[1]}  render={render_portal[0]}x{render_portal[1]}  "
        f"density={density:g}  fit={args.fit}  matte={matte}  "
        f"deskew=±{deskew_limit}°  crop_to_portal={args.crop_to_portal}  "
        f"face_crop={face_aware and _HAS_CV2}  keep_border={keep_border}  "
        f"smooth_grid={smooth_grid and _HAS_CV2}  "
        f"enhance={args.enhance}{preset_note}"
    )

    def _process(raw: bytes, dest: Path) -> str:
        w, h, skew, ar_cropped, face_info = process_raw(
            raw,
            dest,
            render_portal,
            matte,
            args.yscale,
            args.fit,
            args.enhance,
            args.rotate,
            deskew_limit=deskew_limit,
            crop_to_portal=args.crop_to_portal,
            crop_threshold=args.crop_threshold,
            face_aware=face_aware,
            keep_white_border=keep_border,
            smooth_grid=smooth_grid,
        )
        notes = []
        if abs(skew) >= 0.15:
            notes.append(f"deskew {skew:+.2f}°")
        if face_info.get("kept_aspect"):
            notes.append("kept white-border AR")
        elif ar_cropped:
            notes.append("aspect-cropped")
        if face_info.get("used") and (ar_cropped or args.fit == "cover"):
            kind = face_info.get("anchor") or "faces"
            notes.append(f"{kind} {face_info['kept']}/{face_info['detected']}")
        grid = face_info.get("grid") or {}
        if grid.get("applied"):
            pv = grid.get("periods_v") or []
            ph = grid.get("periods_h") or []
            if pv or ph:
                notes.append(f"grid v{pv}/h{ph}")
            else:
                notes.append("grid smooth")
        extra = f" ({', '.join(notes)})" if notes else ""
        return f"{dest.name} ({w}x{h}{extra})"

    scanned = 0
    for raw_path in args.from_raw:
        p = Path(raw_path).expanduser()
        raw = p.read_bytes()
        dest = out_dir / f"photo_{idx:03d}.jpg"
        print(f"wrote {_process(raw, dest)} from {p.name}")
        idx += 1
        scanned += 1

    live = args.scan or args.count > 0 or (not args.from_raw)
    if not live:
        print(f"done — {scanned} frame(s) in {out_dir}")
        return 0

    bot = Bot(verbose=args.verbose)
    try:
        data = ping(bot)
        if not data:
            print("no status response; is the PS100 plugged in?", file=sys.stderr)
            return 1
        print(f"status: {data!r} ({hx(data)})")

        live_count = 0
        while True:
            if args.count and live_count >= args.count:
                break
            if not args.count:
                try:
                    reply = input("Load a photo, then Enter to scan (q=quit): ").strip().lower()
                except EOFError:
                    break
                if reply in ("q", "quit", "exit"):
                    break

            print("scanning…")
            t0 = time.time()
            try:
                raw = scan_one(bot, verbose=args.verbose)
            except Exception as exc:
                print(f"scan failed: {exc}", file=sys.stderr)
                continue
            dest = out_dir / f"photo_{idx:03d}.jpg"
            raw_path = out_dir / f"photo_{idx:03d}.raw"
            raw_path.write_bytes(raw)
            print(f"wrote {_process(raw, dest)} in {time.time() - t0:.1f}s")
            idx += 1
            live_count += 1
            scanned += 1
    finally:
        bot.close()

    print(f"done — {scanned} frame(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
