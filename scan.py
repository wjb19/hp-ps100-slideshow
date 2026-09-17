#!/usr/bin/env python3
"""Scan one page from the HP PS100 using the WorkScan USB sequence.

Usage:
  sudo python3 scan.py                  # write scan.raw + scan.pgm
  sudo python3 scan.py --out /tmp/page  # /tmp/page.raw and .pgm
  sudo python3 scan.py --ping           # status only (expects 'NOVA')
  sudo python3 scan.py -v

Requires the scanner on USB (03f0:53f0). Detaches usbscan/usblp if needed
via libusb auto-detach. Put a sheet in the feeder before scanning.
"""

from __future__ import annotations

import argparse
import sys
import time

from bot import Bot, hx
from protocol import (
    EXPECTED_IMAGE_BYTES,
    PARAM_DISPLAY_Y_SCALE,
    PARAM_WIDTH,
    estimate_yscale,
    ping,
    prepare_scan,
    read_image,
    resolve_yscale,
    stop_scan,
    summarize_param_block,
    write_pgm,
    write_ppm,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="HP PS100 Linux scan (capture-derived)")
    ap.add_argument("--out", default="scan", help="output prefix (default: scan)")
    ap.add_argument("--ping", action="store_true", help="only query NOVA status")
    ap.add_argument("--width", type=int, default=PARAM_WIDTH, help="PGM/PPM width")
    ap.add_argument(
        "--yscale",
        default="auto",
        help="vertical scale vs raw rows, or 'auto' (default) to detect from content",
    )
    ap.add_argument("--bytes", type=int, default=EXPECTED_IMAGE_BYTES, help="expected image size")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    bot = Bot(verbose=args.verbose)
    try:
        data = ping(bot)
        if not data:
            print("no status response; is this a PS100?", file=sys.stderr)
            return 1
        print(f"status: {data!r} ({hx(data)})")
        if args.ping:
            return 0

        print(summarize_param_block())
        print("starting scan — paper should feed")
        t0 = time.time()
        prepare_scan(bot)
        raw = read_image(bot, total_bytes=args.bytes, start_offset=None)
        stop_scan(bot)
        dt = time.time() - t0
        raw_path = f"{args.out}.raw"
        pgm_path = f"{args.out}.pgm"
        ppm_path = f"{args.out}.ppm"
        with open(raw_path, "wb") as f:
            f.write(raw)
        yscale = resolve_yscale(raw, args.yscale, width=args.width)
        w, h = write_ppm(ppm_path, raw, width=args.width, y_scale=yscale)
        write_pgm(pgm_path, raw, width=args.width, y_scale=yscale)
        print(
            f"wrote {raw_path} ({len(raw)} bytes), {ppm_path} and {pgm_path} "
            f"({w}x{h}, yscale={yscale}) in {dt:.1f}s"
        )
        return 0
    finally:
        bot.close()


if __name__ == "__main__":
    raise SystemExit(main())
