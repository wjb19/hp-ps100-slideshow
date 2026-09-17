#!/usr/bin/env python3
"""Rebuild a PGM from ps100-scan.pcapng for offline verification."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

from protocol import PARAM_WIDTH, write_pgm


def iter_epb(path: Path):
    data = path.read_bytes()
    off = 0
    while off + 8 <= len(data):
        btype, blen = struct.unpack_from("<II", data, off)
        if blen < 12 or off + blen > len(data):
            break
        body = data[off + 8 : off + blen - 4]
        if btype == 6 and len(body) >= 20:
            _iface, ts_hi, ts_lo, caplen, _orig = struct.unpack_from("<IIIII", body, 0)
            yield (ts_hi << 32) | ts_lo, body[20 : 20 + caplen]
        off += blen


def extract(path: Path) -> bytes:
    packets = list(iter_epb(path))
    image = bytearray()
    state = None
    for _ts, pkt in packets:
        if len(pkt) < 27:
            continue
        hdr = struct.unpack_from("<H", pkt, 0)[0]
        info = pkt[16]
        ep = pkt[21]
        xfer = pkt[22]
        dlen = struct.unpack_from("<I", pkt, 23)[0]
        payload = pkt[hdr : hdr + dlen]
        din = (info & 1) == 1
        if xfer != 3:
            continue
        if (not din) and ep == 0x02 and len(payload) == 31 and payload[:4] == b"USBC":
            if payload[15] == 0xC3:
                state = "c3"
            else:
                state = None
        elif state == "c3" and din and ep == 0x81:
            if len(payload) == 13:
                state = None
            elif payload:
                image.extend(payload)
    return bytes(image)


def main() -> int:
    pcap = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/bill/Downloads/ps100-scan.pcapng")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/bill/Downloads/hp-ps100/capture-image")
    raw = extract(pcap)
    raw_path = out.with_suffix(".raw")
    raw_path.write_bytes(raw)
    # USBPcap 64KiB URBs are 28 bytes short in this capture; pad to CBW total
    # (7776×1971) so the parameter-block width divides cleanly.
    target = PARAM_WIDTH * 1971
    if len(raw) < target:
        raw = raw + bytes(target - len(raw))
    w, h = write_pgm(str(out.with_suffix(".pgm")), raw, width=PARAM_WIDTH)
    print(f"wrote {raw_path} ({raw_path.stat().st_size} bytes)")
    print(f"wrote {out.with_suffix('.pgm')} ({w}x{h})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
