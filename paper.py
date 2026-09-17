#!/usr/bin/env python3
"""See whether paper in the HP PS100 feeder changes SCSI sense or exposes a volume."""

import sys

sys.path.insert(0, "/home/bill/Downloads/hp-ps100")
from discover import Bot, parse_capacity, parse_sense


def av_read(code, length):
    return bytes([0x28, 0x00, code, 0x00, 0x00, 0x00]) + length.to_bytes(3, "big") + bytes([0x00])


def sense(bot):
    data, _ = bot.command("REQUEST SENSE", [0x03, 0x00, 0x00, 0x00, 0x12, 0x00], 0x12)
    parse_sense(data)
    return data


def main():
    bot = Bot()
    try:
        data, status = bot.command("INQUIRY", [0x12, 0x00, 0x00, 0x00, 0x24, 0x00], 0x24)
        if data and len(data) >= 36:
            print(f"  id {data[8:16]!r} {data[16:32]!r} {data[32:36]!r}")
        if status not in (0, None):
            sense(bot)

        _, status = bot.command("TEST UNIT READY", [0x00] * 6)
        if status != 0:
            sense(bot)

        data, status = bot.command(
            "READ CAPACITY(10)",
            [0x25, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            8,
        )
        cap = parse_capacity(data) if status == 0 else None
        if status != 0:
            sense(bot)

        if cap and 1 <= cap[1] <= 4096 and cap[0] <= 16 * 1024 * 1024:
            bot.command("READ(10) LBA0", [0x28, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00], cap[1])

        for name, cdb, length in (
            ("media check", [0x08, 0x00, 0x00, 0x00, 0x04, 0x00], 4),
            ("get data status", [0x34, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x00], 12),
            ("light status", av_read(0xA0, 1), 1),
            ("button status", av_read(0xA1, 8), 8),
            ("get event", [0x4A, 0x01, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x08, 0x00], 8),
        ):
            data, status = bot.command(name, cdb, length)
            if status not in (0, None):
                sense(bot)
    finally:
        bot.close()


if __name__ == "__main__":
    main()
