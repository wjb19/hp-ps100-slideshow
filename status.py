#!/usr/bin/env python3
"""Read-only status queries over the PS100 BOT transport."""

import struct
import sys
import time

sys.path.insert(0, "/home/bill/Downloads/hp-ps100")
from discover import Bot, hx, parse_sense

def av_read(code, length):
    # command_read: opc, bitset1, datatype, readtype, qual[2], len[3], control
    cdb = bytes([0x28, 0x00, code, 0x00, 0x00, 0x00]) + length.to_bytes(3, "big") + bytes([0x00])
    return cdb


def main():
    bot = Bot()
    try:
        data, status = bot.command("INQUIRY", [0x12, 0x00, 0x00, 0x00, 0x24, 0x00], 0x24)
        if data and len(data) >= 36:
            print(f"  id {data[8:16]!r} {data[16:32]!r} {data[32:36]!r}")

        _, status = bot.command("TEST UNIT READY", [0x00] * 6)
        if status != 0:
            sense, _ = bot.command("REQUEST SENSE", [0x03, 0x00, 0x00, 0x00, 0x12, 0x00], 0x12)
            parse_sense(sense)

        queries = [
            ("media check 1", [0x08, 0x00, 0x00, 0x00, 0x01, 0x00], 1),
            ("media check 4", [0x08, 0x00, 0x00, 0x00, 0x04, 0x00], 4),
            ("get data status", [0x34, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x00], 12),
            ("light status", av_read(0xA0, 1), 1),
            ("button status", av_read(0xA1, 8), 8),
            ("firmware status", av_read(0x90, 9), 9),
            ("accessories", av_read(0x64, 16), 16),
            ("general ability", av_read(0xD2, 32), 32),
            ("flash info", av_read(0x6A, 16), 16),
            ("get event", [0x4A, 0x01, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x08, 0x00], 8),
        ]
        for name, cdb, length in queries:
            data, status = bot.command(name, cdb, length)
            if status not in (0, None):
                sense, _ = bot.command("REQUEST SENSE", [0x03, 0x00, 0x00, 0x00, 0x12, 0x00], 0x12)
                parse_sense(sense)
    finally:
        bot.close()


if __name__ == "__main__":
    main()
