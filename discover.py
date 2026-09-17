#!/usr/bin/env python3
"""Speak USB Bulk-Only Transport to the HP PS100 and dump what it implements.

The first probe showed the device answers a BOT-wrapped SCSI INQUIRY.
This completes the status phase and issues read-only SCSI commands.
"""

import ctypes
import ctypes.util
import struct
import sys
import time

VID, PID = 0x03F0, 0x53F0
IFACE, EP_IN, EP_OUT = 0, 0x81, 0x02
CBW_SIG = 0x43425355
CSW_SIG = 0x53425355

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
    ("libusb_bulk_transfer", [
        ctypes.c_void_p, ctypes.c_ubyte, ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_uint,
    ], ctypes.c_int),
    ("libusb_error_name", [ctypes.c_int], ctypes.c_char_p),
):
    fn = getattr(lib, name)
    fn.argtypes = args
    fn.restype = rest


def ename(code):
    if code >= 0:
        return "ok"
    name = lib.libusb_error_name(code)
    return name.decode() if name else str(code)


def hx(data, limit=96):
    shown = data[:limit]
    suffix = " ..." if len(data) > limit else ""
    return " ".join(f"{b:02x}" for b in shown) + suffix


def ascii(data):
    return "".join(chr(b) if 32 <= b < 127 else "." for b in data)


class Bot:
    def __init__(self):
        self.ctx = ctypes.c_void_p()
        if lib.libusb_init(ctypes.byref(self.ctx)) < 0:
            raise SystemExit("libusb_init failed")
        self.h = None
        self.tag = 1
        self.open()

    def open(self):
        if self.h:
            lib.libusb_release_interface(self.h, IFACE)
            lib.libusb_close(self.h)
        self.h = lib.libusb_open_device_with_vid_pid(self.ctx, VID, PID)
        if not self.h:
            raise SystemExit("cannot open HP PS100")
        lib.libusb_set_auto_detach_kernel_driver(self.h, 1)
        rc = lib.libusb_claim_interface(self.h, IFACE)
        if rc < 0:
            raise SystemExit(f"claim interface: {ename(rc)}")

    def reset(self):
        rc = lib.libusb_reset_device(self.h)
        print(f"device reset: {ename(rc)}")
        time.sleep(0.5)
        self.open()

    def bulk(self, ep, payload, timeout):
        if ep == EP_OUT:
            buf = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
            n = ctypes.c_int()
            rc = lib.libusb_bulk_transfer(self.h, ep, buf, len(payload), ctypes.byref(n), timeout)
            return rc, n.value, b""
        buf = (ctypes.c_ubyte * payload)()
        n = ctypes.c_int()
        rc = lib.libusb_bulk_transfer(self.h, ep, buf, payload, ctypes.byref(n), timeout)
        return rc, n.value, bytes(buf[: n.value])

    def command(self, name, cdb, data_len=0, direction="in", timeout=1500):
        cdb = bytes(cdb)
        if len(cdb) > 16:
            raise ValueError("cdb too long")
        cbw = bytearray(31)
        struct.pack_into("<IIIBBB", cbw, 0, CBW_SIG, self.tag, data_len, 0x80 if direction == "in" else 0x00, 0, len(cdb))
        cbw[15:15 + len(cdb)] = cdb
        self.tag = (self.tag + 1) & 0xFFFFFFFF

        print(f"\n== {name}")
        print(f"  cdb  {hx(cdb)}")
        rc, n, _ = self.bulk(EP_OUT, bytes(cbw), min(timeout, 800))
        if rc < 0 or n != 31:
            print(f"  cbw  {ename(rc)} n={n}")
            lib.libusb_clear_halt(self.h, EP_OUT)
            return None, None

        data = b""
        if data_len:
            rc, n, data = self.bulk(EP_IN if direction == "in" else EP_OUT, data_len, timeout)
            print(f"  data {ename(rc)} n={n}")
            if data:
                print(f"  hex  {hx(data)}")
                print(f"  text {ascii(data)}")
            if rc < 0:
                lib.libusb_clear_halt(self.h, EP_IN if direction == "in" else EP_OUT)

        rc, n, csw = self.bulk(EP_IN, 13, timeout)
        status = None
        if rc == 0 and len(csw) == 13:
            sig, tag, residue, status = struct.unpack("<IIIB", csw)
            ok = sig == CSW_SIG
            print(f"  csw  {'USBS' if ok else hx(csw[:4])} tag={tag} residue={residue} status={status}")
        else:
            print(f"  csw  {ename(rc)} n={n} {hx(csw)}")
            lib.libusb_clear_halt(self.h, EP_IN)
        return data, status

    def close(self):
        if self.h:
            lib.libusb_release_interface(self.h, IFACE)
            lib.libusb_close(self.h)
        lib.libusb_exit(self.ctx)


def parse_inquiry(data):
    if not data or len(data) < 36:
        return
    pdt = data[0] & 0x1F
    vendor = data[8:16].decode("ascii", "replace").strip()
    product = data[16:32].decode("ascii", "replace").strip()
    rev = data[32:36].decode("ascii", "replace").strip()
    print(f"  parsed type={pdt} rmb={bool(data[1] & 0x80)} vendor={vendor!r} product={product!r} rev={rev!r}")


def parse_capacity(data):
    if not data or len(data) < 8:
        return None
    last, block = struct.unpack(">II", data[:8])
    blocks = last + 1
    print(f"  parsed blocks={blocks} block={block} size={blocks * block} bytes")
    return blocks, block


def parse_sense(data):
    if not data or len(data) < 14:
        return
    key = data[2] & 0x0F
    asc, ascq = data[12], data[13]
    print(f"  sense key={key:#x} asc/ascq={asc:02x}/{ascq:02x}")


def main():
    bot = Bot()
    try:
        data, status = bot.command("INQUIRY", [0x12, 0x00, 0x00, 0x00, 0x60, 0x00], 0x60)
        parse_inquiry(data)
        if status not in (0, None) and status != 0:
            bot.reset()

        data, _ = bot.command("VPD supported pages", [0x12, 0x01, 0x00, 0x00, 0x40, 0x00], 0x40)
        data, _ = bot.command("VPD serial", [0x12, 0x01, 0x80, 0x00, 0x40, 0x00], 0x40)
        data, status = bot.command("TEST UNIT READY", [0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        if status not in (0, None):
            sense, _ = bot.command("REQUEST SENSE", [0x03, 0x00, 0x00, 0x00, 0x12, 0x00], 0x12)
            parse_sense(sense)

        data, status = bot.command("READ CAPACITY(10)", [0x25, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], 8)
        cap = parse_capacity(data) if status == 0 else None
        if status not in (0, None):
            sense, _ = bot.command("REQUEST SENSE", [0x03, 0x00, 0x00, 0x00, 0x12, 0x00], 0x12)
            parse_sense(sense)

        bot.command("MODE SENSE(6) all", [0x1A, 0x00, 0x3F, 0x00, 0xFC, 0x00], 0xFC)
        bot.command("READ FORMAT CAPACITIES", [0x23, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFC, 0x00], 0xFC)

        if cap and 1 <= cap[1] <= 4096 and cap[0] <= 16 * 1024 * 1024:
            bot.command("READ(10) LBA0", [0x28, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00], cap[1])
    finally:
        bot.close()


if __name__ == "__main__":
    main()
