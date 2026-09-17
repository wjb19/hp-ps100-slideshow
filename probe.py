#!/usr/bin/env python3
"""Read-only protocol probe for the HP PS100 (03f0:53f0).

Sends inquiry/status-style packets only. Resets the device between
attempts so a stall from one guess does not poison the next.
"""

import ctypes
import ctypes.util
import sys
import time

VID = 0x03F0
PID = 0x53F0
IFACE = 0
EP_IN = 0x81
EP_OUT = 0x02

lib = ctypes.CDLL(ctypes.util.find_library("usb-1.0") or "libusb-1.0.so.0")
lib.libusb_init.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
lib.libusb_init.restype = ctypes.c_int
lib.libusb_exit.argtypes = [ctypes.c_void_p]
lib.libusb_open_device_with_vid_pid.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint16]
lib.libusb_open_device_with_vid_pid.restype = ctypes.c_void_p
lib.libusb_close.argtypes = [ctypes.c_void_p]
lib.libusb_set_auto_detach_kernel_driver.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.libusb_set_auto_detach_kernel_driver.restype = ctypes.c_int
lib.libusb_claim_interface.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.libusb_claim_interface.restype = ctypes.c_int
lib.libusb_release_interface.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.libusb_release_interface.restype = ctypes.c_int
lib.libusb_clear_halt.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
lib.libusb_clear_halt.restype = ctypes.c_int
lib.libusb_reset_device.argtypes = [ctypes.c_void_p]
lib.libusb_reset_device.restype = ctypes.c_int
lib.libusb_bulk_transfer.argtypes = [
    ctypes.c_void_p, ctypes.c_ubyte, ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.c_uint,
]
lib.libusb_bulk_transfer.restype = ctypes.c_int
lib.libusb_control_transfer.argtypes = [
    ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint16,
    ctypes.c_uint16, ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint,
]
lib.libusb_control_transfer.restype = ctypes.c_int
lib.libusb_error_name.argtypes = [ctypes.c_int]
lib.libusb_error_name.restype = ctypes.c_char_p


def ename(code):
    if code >= 0:
        return "ok"
    name = lib.libusb_error_name(code)
    return name.decode() if name else str(code)


def hx(data):
    return " ".join(f"{b:02x}" for b in data)


def ascii(data):
    return "".join(chr(b) if 32 <= b < 127 else "." for b in data)


def bot_inquiry():
    # USB Mass Storage Command Block Wrapper around SCSI INQUIRY (36 bytes).
    cbw = bytearray(31)
    cbw[0:4] = b"USBC"
    cbw[4:8] = b"\x01\x00\x00\x00"
    cbw[8:12] = (36).to_bytes(4, "little")
    cbw[12] = 0x80
    cbw[13] = 0
    cbw[14] = 6
    cbw[15:21] = bytes([0x12, 0x00, 0x00, 0x00, 0x24, 0x00])
    return bytes(cbw)


def candidates():
    inquiry10 = bytes([0x12, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00])
    inquiry60 = bytes([0x12, 0x00, 0x00, 0x00, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00])
    tur = bytes(10)
    sense = bytes([0x03, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00])
    return [
        ("avision INQUIRY/36", inquiry10, 64),
        ("avision INQUIRY/96", inquiry60, 128),
        ("avision TUR", tur, 16),
        ("avision REQUEST SENSE", sense, 32),
        ("INQUIRY unpadded", bytes([0x12, 0x00, 0x00, 0x00, 0x24, 0x00]), 64),
        ("len-le10 + INQUIRY", bytes([0x0a, 0x00, 0x00, 0x00]) + inquiry10, 64),
        ("len-be10 + INQUIRY", bytes([0x00, 0x00, 0x00, 0x0a]) + inquiry10, 64),
        ("dir-in + INQUIRY", bytes([0x00, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00]) + inquiry10, 64),
        ("BOT INQUIRY", bot_inquiry(), 64),
        ("short status 00", bytes([0x00]), 16),
        ("short status 01 00", bytes([0x01, 0x00]), 16),
        ("esc I", b"\x1bI", 32),
        ("esc i", b"\x1bi", 32),
        ("text INQUIRY", b"INQUIRY\r\n", 64),
        ("text GETSTATUS", b"GETSTATUS\r", 64),
    ]


class Dev:
    def __init__(self):
        self.ctx = ctypes.c_void_p()
        rc = lib.libusb_init(ctypes.byref(self.ctx))
        if rc < 0:
            raise SystemExit(f"libusb_init {ename(rc)}")
        self.h = None

    def open(self):
        if self.h:
            lib.libusb_release_interface(self.h, IFACE)
            lib.libusb_close(self.h)
            self.h = None
        self.h = lib.libusb_open_device_with_vid_pid(self.ctx, VID, PID)
        if not self.h:
            raise SystemExit("scanner not openable")
        lib.libusb_set_auto_detach_kernel_driver(self.h, 1)
        rc = lib.libusb_claim_interface(self.h, IFACE)
        if rc < 0:
            raise SystemExit(f"claim {ename(rc)}")

    def reset(self):
        rc = lib.libusb_reset_device(self.h)
        print(f"  reset {ename(rc)}")
        time.sleep(0.4)
        self.open()
        for ep in (EP_IN, EP_OUT):
            crc = lib.libusb_clear_halt(self.h, ep)
            if crc < 0:
                print(f"  clear halt ep {ep:#x} {ename(crc)}")

    def write(self, payload, timeout=800):
        buf = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
        n = ctypes.c_int()
        rc = lib.libusb_bulk_transfer(self.h, EP_OUT, buf, len(payload), ctypes.byref(n), timeout)
        return rc, n.value

    def read(self, size, timeout=600):
        buf = (ctypes.c_ubyte * size)()
        n = ctypes.c_int()
        rc = lib.libusb_bulk_transfer(self.h, EP_IN, buf, size, ctypes.byref(n), timeout)
        return rc, bytes(buf[: n.value])

    def control_in(self, bm, req, value, index, length, timeout=300):
        buf = (ctypes.c_ubyte * length)()
        rc = lib.libusb_control_transfer(self.h, bm, req, value, index, buf, length, timeout)
        data = bytes(buf[: rc]) if rc > 0 else b""
        return rc, data

    def close(self):
        if self.h:
            lib.libusb_release_interface(self.h, IFACE)
            lib.libusb_close(self.h)
            self.h = None
        lib.libusb_exit(self.ctx)


def main():
    dev = Dev()
    dev.open()
    try:
        print("control reads (device-to-host only)")
        hits = 0
        for bm, label in ((0xC0, "vendor-device"), (0xC1, "vendor-iface"), (0xA1, "class-iface")):
            for req in range(0, 8):
                rc, data = dev.control_in(bm, req, 0, 0, 16)
                if rc > 0:
                    hits += 1
                    print(f"  {label} req={req:#04x} n={rc} {hx(data)} {ascii(data)}")
                elif rc != -9 and rc != -7 and rc != -1:
                    # -9 pipe, -7 timeout, -1 io. Print anything else.
                    print(f"  {label} req={req:#04x} {ename(rc)}")
        if not hits:
            print("  no control-in replies")

        for name, payload, rsize in candidates():
            print(f"\n== {name}")
            dev.reset()
            rc, n = dev.write(payload)
            print(f"  write {ename(rc)} n={n}  {hx(payload)}")
            if rc < 0:
                continue
            rc, data = dev.read(rsize, timeout=900)
            print(f"  read  {ename(rc)} n={len(data)}")
            if data:
                print(f"  data  {hx(data)}")
                print(f"  ascii {ascii(data)}")
            else:
                # Second chance: a 1-byte status may arrive late.
                rc2, data = dev.read(8, timeout=400)
                if data:
                    print(f"  late  {ename(rc2)} {hx(data)} {ascii(data)}")
    finally:
        dev.close()


if __name__ == "__main__":
    main()
