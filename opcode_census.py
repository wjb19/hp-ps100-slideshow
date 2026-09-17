#!/usr/bin/env python3
"""Classify SCSI opcodes by the sense they return. No data phase.

A status of 0 means the scanner executed the command. Stop immediately.
Sense 24/00 means the opcode exists but the fields are wrong.
Sense 20/00 means the opcode is not implemented.
"""

import sys

sys.path.insert(0, "/home/bill/Downloads/hp-ps100")
from discover import Bot, parse_sense

# Known to move media, format, or write. Do not send even with zeros.
SKIP = {
    0x00,  # TUR, already known
    0x03,  # REQUEST SENSE
    0x04,  # FORMAT UNIT
    0x12,  # INQUIRY
    0x15,  # MODE SELECT
    0x1B,  # START STOP / Avision SCAN (already accepted, did not scan)
    0x2A,  # WRITE(10)
    0x2E,  # WRITE AND VERIFY
    0x31,  # OBJECT POSITION (feeds paper on Avision)
    0x55,  # MODE SELECT(10)
    0xAA,  # WRITE(12)
}

# Empty command success is normal here. Keep going.
HARMLESS_OK = {0x1A, 0x5A}


def sense_code(bot):
    data, _ = bot.command("REQUEST SENSE", [0x03, 0x00, 0x00, 0x00, 0x12, 0x00], 0x12, timeout=600)
    if not data or len(data) < 14:
        return None
    key = data[2] & 0x0F
    return key, data[12], data[13]


def main():
    bot = Bot()
    known = []
    missing = []
    other = []
    try:
        start = int(sys.argv[1], 0) if len(sys.argv) > 1 else 0x1C
        for opc in range(start, 0x100):
            if opc in SKIP or opc in (0x00, 0x03, 0x12):
                continue
            cdb = bytes([opc]) + bytes(9)
            print(f"\n-- opcode {opc:#04x}", flush=True)
            _, status = bot.command(f"opc {opc:#04x}", cdb, 0, timeout=500)
            if status == 0:
                # Vendor commands that run with no parameters may move paper.
                if opc >= 0x80 and opc not in HARMLESS_OK:
                    print(f"EXECUTED {opc:#04x} with an empty CDB. Stopping so it cannot continue.", flush=True)
                    known.append((opc, "executed"))
                    break
                known.append((opc, "ok-empty"))
                print("  accepted empty", flush=True)
                continue
            code = sense_code(bot)
            if code is None:
                other.append((opc, "no sense"))
                print(f"  no sense", flush=True)
                continue
            key, asc, ascq = code
            label = f"{key:#x}/{asc:02x}/{ascq:02x}"
            print(f"  sense {label}", flush=True)
            if key == 5 and asc == 0x24:
                known.append((opc, label))
            elif key == 5 and asc == 0x20:
                missing.append(opc)
            else:
                other.append((opc, label))
    finally:
        bot.close()

    print("\n==== census ====")
    print("implemented or unexpected:")
    for opc, label in known + other:
        print(f"  {opc:#04x} {label}")
    print(f"not implemented: {len(missing)}")
    print(" ".join(f"{o:#04x}" for o in missing))


if __name__ == "__main__":
    main()
