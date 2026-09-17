#!/usr/bin/env python3
"""USB Bulk-Only Transport for the HP PS100 (03f0:53f0).

The device uses standard CBW wrappers (signature 'USBC') but returns CSW
signatures of 0x00000000 instead of 'USBS'. Vendor opcodes 0xC5 / 0xC3 use
16-byte CDBs; empty/10-byte probes of those opcodes fail.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import struct
import time
from typing import Optional

VID, PID = 0x03F0, 0x53F0
IFACE, EP_IN, EP_OUT = 0, 0x81, 0x02
CBW_SIG = 0x43425355
CSW_SIG_USBS = 0x53425355
CSW_SIG_NOVA = 0x00000000

lib = ctypes.CDLL(ctypes.util.find_library("usb-1.0") or "libusb-1.0.so.0")
for name, args, rest in (
    ("libusb_init", [ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int),
    ("libusb_exit", [ctypes.c_void_p], None),
    ("libusb_open_device_with_vid_pid", [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint16], ctypes.c_void_p),
    ("libusb_close", [ctypes.c_void_p], None),
    ("libusb_set_auto_detach_kernel_driver", [ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
    ("libusb_claim_interface", [ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
    ("libusb_release_interface", [ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
    ("libusb_clear_halt", [ctypes.c_void_p, ctypes.c_ubyte], ctypes.c_int),
    ("libusb_reset_device", [ctypes.c_void_p], ctypes.c_int),
    (
        "libusb_bulk_transfer",
        [
            ctypes.c_void_p,
            ctypes.c_ubyte,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_uint,
        ],
        ctypes.c_int,
    ),
    ("libusb_error_name", [ctypes.c_int], ctypes.c_char_p),
):
    fn = getattr(lib, name)
    fn.argtypes = args
    fn.restype = rest


def ename(code: int) -> str:
    if code >= 0:
        return "ok"
    name = lib.libusb_error_name(code)
    return name.decode() if name else str(code)


def hx(data: bytes, limit: int = 96) -> str:
    shown = data[:limit]
    suffix = " ..." if len(data) > limit else ""
    return " ".join(f"{b:02x}" for b in shown) + suffix


class Bot:
    def __init__(self, verbose: bool = False):
        self.ctx = ctypes.c_void_p()
        if lib.libusb_init(ctypes.byref(self.ctx)) < 0:
            raise SystemExit("libusb_init failed")
        self.h = None
        self.tag = 1
        self.verbose = verbose
        self.open()

    def open(self) -> None:
        if self.h:
            lib.libusb_release_interface(self.h, IFACE)
            lib.libusb_close(self.h)
        self.h = lib.libusb_open_device_with_vid_pid(self.ctx, VID, PID)
        if not self.h:
            raise SystemExit("cannot open HP PS100 (03f0:53f0) — is it plugged in?")
        lib.libusb_set_auto_detach_kernel_driver(self.h, 1)
        rc = lib.libusb_claim_interface(self.h, IFACE)
        if rc < 0:
            raise SystemExit(f"claim interface: {ename(rc)}")

    def reset(self) -> None:
        rc = lib.libusb_reset_device(self.h)
        if self.verbose:
            print(f"device reset: {ename(rc)}")
        time.sleep(0.5)
        self.open()

    def bulk_out(self, payload: bytes, timeout: int) -> tuple[int, int]:
        buf = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
        n = ctypes.c_int()
        rc = lib.libusb_bulk_transfer(self.h, EP_OUT, buf, len(payload), ctypes.byref(n), timeout)
        return rc, n.value

    def bulk_in(self, size: int, timeout: int) -> tuple[int, bytes]:
        buf = (ctypes.c_ubyte * size)()
        n = ctypes.c_int()
        rc = lib.libusb_bulk_transfer(self.h, EP_IN, buf, size, ctypes.byref(n), timeout)
        return rc, bytes(buf[: n.value])

    def clear(self, ep: int) -> None:
        lib.libusb_clear_halt(self.h, ep)

    def command(
        self,
        name: str,
        cdb: bytes | list[int],
        data_len: int = 0,
        direction: str = "in",
        out_data: bytes = b"",
        timeout: int = 5000,
        quiet: bool = False,
    ) -> tuple[Optional[bytes], Optional[int]]:
        cdb = bytes(cdb)
        if len(cdb) > 16:
            raise ValueError("cdb too long")
        if direction == "out" and out_data and data_len == 0:
            data_len = len(out_data)
        if direction == "out" and out_data and len(out_data) != data_len:
            raise ValueError("out_data length must match data_len")

        cbw = bytearray(31)
        flags = 0x80 if direction == "in" else 0x00
        struct.pack_into("<IIIBBB", cbw, 0, CBW_SIG, self.tag, data_len, flags, 0, 16 if len(cdb) == 16 else len(cdb))
        # PS100 vendor commands always use a 16-byte CDB slot.
        cdb_pad = cdb + bytes(16 - len(cdb))
        cbw[15:31] = cdb_pad[:16]
        tag = self.tag
        self.tag = (self.tag + 1) & 0xFFFFFFFF

        show = self.verbose and not quiet
        if show:
            print(f"\n== {name}")
            print(f"  cdb  {hx(cdb_pad)}")
            print(f"  dir  {direction} len={data_len} tag={tag:#x}")

        rc, n = self.bulk_out(bytes(cbw), min(timeout, 2000))
        if rc < 0 or n != 31:
            if show:
                print(f"  cbw  {ename(rc)} n={n}")
            self.clear(EP_OUT)
            return None, None

        data = b""
        if data_len:
            if direction == "in":
                remaining = data_len
                parts: list[bytes] = []
                while remaining > 0:
                    rc, chunk = self.bulk_in(remaining, timeout)
                    if rc < 0:
                        if show:
                            print(f"  data {ename(rc)} after {sum(map(len, parts))} bytes")
                        if not parts:
                            self.clear(EP_IN)
                        break
                    if not chunk:
                        break
                    parts.append(chunk)
                    remaining -= len(chunk)
                    # Device sometimes returns short URBs; keep reading until
                    # the requested CBW length is satisfied or IN stalls.
                    if remaining > 0 and len(chunk) < 512:
                        continue
                data = b"".join(parts)
                if show:
                    print(f"  data n={len(data)} (asked {data_len})")
            else:
                rc, n = self.bulk_out(out_data, timeout)
                if show:
                    print(f"  data {ename(rc)} n={n}")
                if rc < 0:
                    self.clear(EP_OUT)

        rc, csw = self.bulk_in(13, min(timeout, 3000))
        status = None
        if rc == 0 and len(csw) == 13:
            sig, got_tag, residue, status = struct.unpack("<IIIB", csw)
            ok = sig in (CSW_SIG_USBS, CSW_SIG_NOVA)
            if show:
                label = "USBS" if sig == CSW_SIG_USBS else (f"{sig:#010x}" if ok else hx(csw[:4]))
                print(f"  csw  {label} tag={got_tag:#x} residue={residue} status={status}")
            if status not in (0, None):
                # Failed data phase often leaves a halt; clear both bulk EPs.
                self.clear(EP_IN)
                self.clear(EP_OUT)
        else:
            if show:
                print(f"  csw  {ename(rc)} n={len(csw)} {hx(csw)}")
            self.clear(EP_IN)
            self.clear(EP_OUT)
        return data, status

    def close(self) -> None:
        if self.h:
            lib.libusb_release_interface(self.h, IFACE)
            lib.libusb_close(self.h)
            self.h = None
        lib.libusb_exit(self.ctx)
