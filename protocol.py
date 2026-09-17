#!/usr/bin/env python3
"""PS100 vendor command helpers derived from a WorkScan USB capture.

Transport is BOT with vendor SCSI opcodes:
  0xC5 — control / status / parameter write
  0xC3 — image buffer read

C5 CDB (16 bytes):
  c5 07 00 00 00 00 00 00 00 <len> <mode> <sub> 00 00 00 00
  mode 0x02 + sub 0x03 = host→device parameter/data OUT
  mode 0xff / 0x02 + sub = device→host status polls

C3 CDB (16 bytes):
  c3 07 <offset BE32> <blocks BE24> <rem> 00 00 00 00 00 00
  Transfer size = blocks*256 + rem. WorkScan reads in bursts of
  9×65536 + 1×32256, then polls until the 24-byte "channel 2" status
  reports ready (first LE dword == 1) before the next burst.
"""

from __future__ import annotations

import struct
import time

from bot import Bot, hx

# Default WorkScan letter/gray/300dpi-style parameter block from capture.
SCAN_PARAM_BLOCK = bytes.fromhex(
    "24000000000000004f0000000000000000470000012c012c"
    "000000000000000000002880000041c88080800508000003"
    "00000000000000000000ff1de0ff001e6010721000006400"
    "640064000000000b000000000000000000"
)

PARAM_WIDTH = 2592  # pixels per channel
PARAM_CHANNELS = 3  # R|G|B packed side-by-side in each scanline
PARAM_HEIGHT = 1971  # raw rows in the WorkScan capture
# Feed sampling is denser than X optics, but not a clean 2× — 0.5 was too flat,
# 1.0 too tall. ~0.75 matches a vertical sticker's proportions better.
PARAM_DISPLAY_Y_SCALE = 0.75
EXPECTED_IMAGE_BYTES = PARAM_WIDTH * PARAM_HEIGHT * PARAM_CHANNELS  # 15_326_496
DEFAULT_START_OFFSET = 0x00462240
KEEPALIVE_OUT = bytes.fromhex("280000000a0d1feae000")
# 0x1e60 in the param block is 7776 = width * channels (bytes per scanline).
PARAM_ROW_STRIDE = PARAM_WIDTH * PARAM_CHANNELS


def c5_cdb(length: int, mode: int, sub: int) -> bytes:
    if not 0 <= length <= 0xFF:
        raise ValueError("C5 length must fit in one byte")
    cdb = bytearray(16)
    cdb[0] = 0xC5
    cdb[1] = 0x07
    cdb[9] = length
    cdb[10] = mode
    cdb[11] = sub
    return bytes(cdb)


def c3_cdb(offset: int, length: int) -> bytes:
    """Encode a C3 read. Size is blocks-of-256 plus a remainder byte."""
    cdb = bytearray(16)
    cdb[0] = 0xC3
    cdb[1] = 0x07
    cdb[2:6] = offset.to_bytes(4, "big")
    blocks, rem = divmod(length, 256)
    cdb[6:9] = blocks.to_bytes(3, "big")
    cdb[9] = rem
    return bytes(cdb)


def c5_out(bot: Bot, name: str, payload: bytes, timeout: int = 3000, quiet: bool = False):
    cdb = c5_cdb(len(payload), 0x02, 0x03)
    return bot.command(
        name, cdb, data_len=len(payload), direction="out", out_data=payload, timeout=timeout, quiet=quiet
    )


def c5_in(
    bot: Bot,
    name: str,
    length: int,
    mode: int = 0xFF,
    sub: int = 0x02,
    timeout: int = 2000,
    quiet: bool = False,
):
    cdb = c5_cdb(length, mode, sub)
    return bot.command(name, cdb, data_len=length, direction="in", timeout=timeout, quiet=quiet)


def ping(bot: Bot) -> bytes | None:
    """Short status read; successful devices answer with ASCII 'NOVA'."""
    data, status = c5_in(bot, "status NOVA", 4, 0xFF, 0x02)
    if bot.verbose and data:
        print(f"  ping {data!r} status={status}")
    return data


def prepare_scan(bot: Bot) -> None:
    """Send the unique pre-scan OUT sequence observed in WorkScan."""
    steps = [
        ("setup 2a/95", bytes.fromhex("2a0095000000000002000000")),
        ("setup 2a/96", bytes.fromhex("2a0096000000000002000000")),
        ("setup 2a/83", bytes.fromhex("2a008300000000001200040000000000000004000000000000000400")),
        ("scan params", SCAN_PARAM_BLOCK),
        ("arm", bytes.fromhex("16000000000000000000")),
        ("start", bytes.fromhex("1b000000008000000000")),
        ("post-start", KEEPALIVE_OUT),
    ]
    for name, payload in steps:
        _data, status = c5_out(bot, name, payload)
        if status not in (0, None):
            raise RuntimeError(f"{name} failed status={status}")


def stop_scan(bot: Bot) -> None:
    c5_out(bot, "stop", bytes.fromhex("17000000000000000000"))


def read_progress(bot: Bot) -> tuple[bool, int | None, bytes | None]:
    """Read channel-2 24-byte progress. ready when LE dword0 == 1."""
    # Light preamble like WorkScan immediately before the progress pair.
    for sub in (0x04, 0x06, 0x04, 0x03, 0x03, 0x05):
        c5_in(bot, f"poll ff/{sub:02x}", 16, 0xFF, sub, timeout=500, quiet=True)
    c5_in(bot, "progress ch1", 24, 0x02, 0x01, timeout=500, quiet=True)
    c5_in(bot, "poll ff/04", 16, 0xFF, 0x04, timeout=500, quiet=True)
    data, _status = c5_in(bot, "progress ch2", 24, 0x02, 0x02, timeout=500, quiet=True)
    if not data or len(data) < 24:
        return False, None, data
    ready = struct.unpack_from("<I", data, 0)[0] == 1
    offset = struct.unpack_from("<I", data, 20)[0]
    return ready, offset if offset else None, data


def wait_for_buffer(bot: Bot, fallback: int, timeout: float = 3.0) -> int:
    """Poll briefly for a ready buffer; fall back to the computed offset.

    WorkScan only needed ~0.5s here. A hard wait on the ready bit can spin
    forever if that dword never flips on Linux, so we always bound the wait.
    """
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        ready, offset, raw = read_progress(bot)
        last = (ready, offset, raw)
        if ready and offset is not None:
            if bot.verbose:
                print(f"  buffer ready offset={offset:#x}")
            return offset
        c5_out(bot, "keepalive", KEEPALIVE_OUT, timeout=500, quiet=True)
        time.sleep(0.03)
    if bot.verbose and last is not None:
        ready, offset, raw = last
        print(
            f"  buffer wait timed out after {timeout:.1f}s "
            f"(ready={ready} offset={offset} raw={hx(raw) if raw else 'none'}); "
            f"using fallback {fallback:#x}"
        )
    return fallback


def read_image(
    bot: Bot,
    total_bytes: int = EXPECTED_IMAGE_BYTES,
    start_offset: int | None = None,
    chunk_sizes: list[int] | None = None,
) -> bytes:
    """Read the scan buffer with 0xC3 in WorkScan burst/poll cadence."""
    sizes = list(chunk_sizes) if chunk_sizes is not None else capture_chunk_sizes()
    if total_bytes != sum(sizes):
        sizes = []
        rem = total_bytes
        while rem > 0:
            n = min(65536, rem)
            sizes.append(n)
            rem -= n

    if start_offset is None:
        print("waiting for scan buffer (max 3s)...")
        offset = wait_for_buffer(bot, fallback=DEFAULT_START_OFFSET, timeout=3.0)
    else:
        offset = start_offset

    parts: list[bytes] = []
    done = 0
    bursts = 0
    for i, n in enumerate(sizes):
        cdb = c3_cdb(offset, n)
        data, status = bot.command(
            f"image @ {offset:#x} +{n}",
            cdb,
            data_len=n,
            direction="in",
            timeout=15000,
        )
        if not data or status not in (0, None):
            bot.clear(0x81)
            bot.clear(0x02)
            # One recovery attempt after a short buffer wait.
            offset = wait_for_buffer(bot, fallback=offset, timeout=2.0)
            cdb = c3_cdb(offset, n)
            data, status = bot.command(
                f"image-retry @ {offset:#x} +{n}",
                cdb,
                data_len=n,
                direction="in",
                timeout=15000,
            )
            if not data or status not in (0, None):
                bot.clear(0x81)
                bot.clear(0x02)
                raise RuntimeError(
                    f"image read failed at offset {offset:#x} n={n} status={status} "
                    f"(got {0 if not data else len(data)} bytes, {done}/{total_bytes} so far)"
                )
        parts.append(data)
        done += len(data)
        offset = (offset + n) & 0xFFFFFFFF
        if bot.verbose:
            print(f"  got {len(data)}/{n} next_off={offset:#x} total={done}")

        # After each short (end of a 10-chunk burst), briefly sync progress.
        if n == 32256 and i + 1 < len(sizes):
            bursts += 1
            print(f"burst {bursts}/24 — {done}/{EXPECTED_IMAGE_BYTES} bytes")
            offset = wait_for_buffer(bot, fallback=offset, timeout=2.0)

    return b"".join(parts)


def _rescale_y_gray(src: bytes, width: int, src_h: int, y_scale: float) -> tuple[int, bytes]:
    """Scale gray image height by y_scale (e.g. 0.75 keeps 75% of rows)."""
    if y_scale <= 0:
        raise ValueError("y_scale must be positive")
    if abs(y_scale - 1.0) < 1e-6:
        return src_h, src
    dst_h = max(1, int(round(src_h * y_scale)))
    out = bytearray(width * dst_h)
    for y in range(dst_h):
        # Nearest-neighbor is fine for docs; maps dst row -> src row.
        sy = min(src_h - 1, int(y / y_scale + 1e-6))
        out[y * width : (y + 1) * width] = src[sy * width : (sy + 1) * width]
    return dst_h, bytes(out)


def _rescale_y_rgb(src: bytes, width: int, src_h: int, y_scale: float) -> tuple[int, bytes]:
    if y_scale <= 0:
        raise ValueError("y_scale must be positive")
    if abs(y_scale - 1.0) < 1e-6:
        return src_h, src
    dst_h = max(1, int(round(src_h * y_scale)))
    out = bytearray(width * dst_h * 3)
    row = width * 3
    for y in range(dst_h):
        sy = min(src_h - 1, int(y / y_scale + 1e-6))
        out[y * row : (y + 1) * row] = src[sy * row : (sy + 1) * row]
    return dst_h, bytes(out)


def extract_channel(raw: bytes, channel: int = 1, width: int = PARAM_WIDTH) -> tuple[int, int, bytes]:
    """Extract one color channel from row-packed R|G|B scanlines."""
    stride = width * PARAM_CHANNELS
    if len(raw) < stride or len(raw) % stride:
        height = len(raw) // width
        return width, height, raw[: width * height]
    height = len(raw) // stride
    if not 0 <= channel < PARAM_CHANNELS:
        raise ValueError("channel out of range")
    out = bytearray(width * height)
    base = channel * width
    for y in range(height):
        row = raw[y * stride + base : y * stride + base + width]
        out[y * width : (y + 1) * width] = row
    return width, height, bytes(out)


def content_bbox(gray: bytes, width: int, height: int, thr: int = 25) -> tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) of the non-dark content region."""
    col_hit = [False] * width
    row_hit = [False] * height
    step = max(1, min(width, height) // 400)
    for y in range(0, height, step):
        row = gray[y * width : (y + 1) * width]
        for x in range(0, width, step):
            if row[x] > thr:
                col_hit[x] = True
                row_hit[y] = True
    cols = [x for x, h in enumerate(col_hit) if h]
    rows = [y for y, h in enumerate(row_hit) if h]
    if not cols or not rows:
        return 0, 0, width, height
    # Expand hits to neighbors lost by stepping.
    x0, x1 = max(0, min(cols) - step), min(width, max(cols) + step + 1)
    y0, y1 = max(0, min(rows) - step), min(height, max(rows) + step + 1)
    return x0, y0, x1, y1


def _crop(gray: bytes, width: int, height: int, box: tuple[int, int, int, int]) -> tuple[int, int, bytes]:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    out = bytearray(w * h)
    for y in range(h):
        src = (y0 + y) * width + x0
        out[y * w : (y + 1) * w] = gray[src : src + w]
    return w, h, bytes(out)


def _gradient_anisotropy(gray: bytes, width: int, height: int) -> float:
    """Return |log(mean|Gx|/mean|Gy|)| — 0 is isotropic (good aspect)."""
    if width < 4 or height < 4:
        return 999.0
    gx = 0.0
    gy = 0.0
    n = 0
    # Sparse sample for speed.
    ys = range(1, height - 1, max(1, height // 200))
    xs = range(1, width - 1, max(1, width // 200))
    for y in ys:
        base = y * width
        for x in xs:
            p = base + x
            gx += abs(gray[p + 1] - gray[p - 1])
            gy += abs(gray[p + width] - gray[p - width])
            n += 1
    if n == 0 or gx < 1e-6 or gy < 1e-6:
        return 999.0
    import math

    return abs(math.log((gx / n) / (gy / n)))


def _circle_score(gray: bytes, width: int, height: int) -> float:
    """Higher is better: reward dark circular blobs (rivets, holes, round logos).

    Uses a few radii and a cheap radial-variance ring score; returns 0 if nothing
    useful is found so gradient scoring remains primary.
    """
    import math

    if width < 32 or height < 32:
        return 0.0
    # Downsample to keep this cheap.
    scale = max(1, max(width, height) // 256)
    w, h = width // scale, height // scale
    small = bytearray(w * h)
    for y in range(h):
        sy = min(height - 1, y * scale)
        for x in range(w):
            small[y * w + x] = gray[sy * width + min(width - 1, x * scale)]

    best = 0.0
    for radius in (6, 8, 10, 14, 18):
        if 2 * radius + 4 >= min(w, h):
            continue
        for cy in range(radius + 2, h - radius - 2, max(2, h // 40)):
            for cx in range(radius + 2, w - radius - 2, max(2, w // 40)):
                cval = small[cy * w + cx]
                ring = []
                for a in range(0, 360, 30):
                    x = int(cx + radius * math.cos(a * math.pi / 180))
                    y = int(cy + radius * math.sin(a * math.pi / 180))
                    ring.append(small[y * w + x])
                if not ring:
                    continue
                mean = sum(ring) / len(ring)
                var = sum((v - mean) ** 2 for v in ring) / len(ring)
                contrast = abs(mean - cval)
                if var < 80 and contrast > 25:
                    score = contrast / (1.0 + var)
                    if score > best:
                        best = score
    return best


def estimate_yscale(
    raw: bytes,
    width: int = PARAM_WIDTH,
    lo: float = 0.45,
    hi: float = 1.05,
    steps: int = 25,
) -> tuple[float, dict]:
    """Estimate Y display scale so square pixels look correct.

    Searches y_scale in [lo, hi], scoring each candidate by gradient isotropy on
    the content crop (primary) plus a weak circular-feature bonus. Returns
    (best_scale, diagnostics).
    """
    _w, raw_h, green = extract_channel(raw, channel=1, width=width)
    box = content_bbox(green, width, raw_h)
    cw, ch, crop = _crop(green, width, raw_h, box)
    if cw < 16 or ch < 16:
        return PARAM_DISPLAY_Y_SCALE, {"reason": "no-content", "bbox": box}

    best_s = PARAM_DISPLAY_Y_SCALE
    best_score = -1e9
    ranked: list[tuple[float, float, float, float]] = []
    for i in range(steps):
        s = lo + (hi - lo) * i / max(1, steps - 1)
        _hh, scaled = _rescale_y_gray(crop, cw, ch, s)
        aniso = _gradient_anisotropy(scaled, cw, _hh)
        circ = _circle_score(scaled, cw, _hh)
        # Lower anisotropy is better; circular features break ties.
        score = -aniso + 0.05 * circ
        ranked.append((score, s, aniso, circ))
        if score > best_score:
            best_score = score
            best_s = s

    ranked.sort(reverse=True)
    # Snap to a few friendly values when very close.
    for snap in (0.5, 2 / 3, 0.75, 0.8, 0.85, 1.0):
        if abs(best_s - snap) <= (hi - lo) / steps:
            best_s = snap
            break

    return round(best_s, 4), {
        "bbox": box,
        "content": (cw, ch),
        "top": [(round(s, 4), round(an, 4), round(c, 3)) for _, s, an, c in ranked[:5]],
        "score": round(best_score, 4),
    }


def resolve_yscale(raw: bytes, y_scale: float | str | None = None, width: int = PARAM_WIDTH) -> float:
    """Return a numeric y_scale; accepts 'auto' / None for estimation."""
    if y_scale is None or (isinstance(y_scale, str) and y_scale.lower() in ("auto", "a")):
        s, info = estimate_yscale(raw, width=width)
        print(f"auto yscale={s} (content {info.get('content')}, top={info.get('top')})")
        return s
    return float(y_scale)


def decode_rowsbs_rgb(
    raw: bytes,
    width: int = PARAM_WIDTH,
    y_scale: float = PARAM_DISPLAY_Y_SCALE,
) -> tuple[int, int, bytes]:
    """Decode scanlines packed as R[w]|G[w]|B[w] into interleaved RGB."""
    stride = width * PARAM_CHANNELS
    if len(raw) < stride or len(raw) % stride:
        height = len(raw) // width
        return width, height, raw[: width * height]
    raw_h = len(raw) // stride
    out = bytearray(width * raw_h * 3)
    for y in range(raw_h):
        row = raw[y * stride : y * stride + stride]
        r = row[0:width]
        g = row[width : 2 * width]
        b = row[2 * width : 3 * width]
        o = y * width * 3
        for x in range(width):
            j = o + 3 * x
            out[j] = r[x]
            out[j + 1] = g[x]
            out[j + 2] = b[x]
    h, scaled = _rescale_y_rgb(bytes(out), width, raw_h, y_scale)
    return width, h, scaled


def stretch_contrast(data: bytes) -> bytes:
    if not data:
        return data
    step = max(1, len(data) // 50000)
    sample = sorted(data[::step])
    lo = sample[max(0, len(sample) // 100)]
    hi = sample[min(len(sample) - 1, len(sample) * 99 // 100)]
    if hi <= lo:
        hi = lo + 1
    scale = 255 / (hi - lo)
    return bytes(min(255, max(0, int((b - lo) * scale))) for b in data)


def write_ppm(
    path: str,
    raw: bytes,
    width: int = PARAM_WIDTH,
    y_scale: float | str | None = PARAM_DISPLAY_Y_SCALE,
) -> tuple[int, int]:
    y_scale = resolve_yscale(raw, y_scale, width=width)
    w, h, rgb = decode_rowsbs_rgb(raw, width=width, y_scale=y_scale)
    rgb = stretch_contrast(rgb)
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(rgb)
    return w, h


def write_pgm(
    path: str,
    raw: bytes,
    width: int = PARAM_WIDTH,
    y_scale: float | str | None = PARAM_DISPLAY_Y_SCALE,
) -> tuple[int, int]:
    """Grayscale from the green channel (no RGB fringing), aspect-corrected."""
    y_scale = resolve_yscale(raw, y_scale, width=width)
    w, raw_h, green = extract_channel(raw, channel=1, width=width)
    h, gray = _rescale_y_gray(green, w, raw_h, y_scale)
    gray = stretch_contrast(gray)
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(gray)
    return w, h


def summarize_param_block(block: bytes = SCAN_PARAM_BLOCK) -> str:
    dpi = int.from_bytes(block[20:22], "big")  # 01 2c = 300
    row_stride = int.from_bytes(block[0x3F:0x41], "big")  # 1e 60 = 7776
    disp_h = int(round(PARAM_HEIGHT * PARAM_DISPLAY_Y_SCALE))
    return (
        f"param_len={len(block)} dpi={dpi} row_stride={row_stride} "
        f"decode={PARAM_WIDTH}x{disp_h} rowsbs-RGB (yscale={PARAM_DISPLAY_Y_SCALE})"
    )


def capture_chunk_sizes() -> list[int]:
    """Exact 0xC3 transfer sizes from the WorkScan capture."""
    sizes: list[int] = []
    for _ in range(24):
        sizes.extend([65536] * 9)
        sizes.append(32256)
    sizes.extend([65536] * 6)
    sizes.append(3360)
    assert sum(sizes) == EXPECTED_IMAGE_BYTES
    return sizes
