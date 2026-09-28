# HP PS100 slideshow scanner

Linux driver and pipeline for the **HP Photosmart PS100** (USB `03f0:53f0`). Scan prints over USB and write fixed-size slideshow frames so a player can advance without layout jumps.

Protocol notes are reverse-engineered from USB captures: Bulk-Only Transport (BOT) with vendor SCSI opcodes `0xC5` (control/status) and `0xC3` (image read).

## Features

- **Live USB scanning** via libusb (auto-detaches `usbscan` / kernel drivers)
- **Fixed portal output** — every JPEG is the same size (default 1080p at **2× density** → 3840×2160 for zoom headroom)
- **`--old-photos` path** for vintage prints:
  - letterbox / contain into the portal (**no cover-crop**)
  - keep white mount / border and native aspect ratio
  - auto deskew (interior Hough + projection; torn borders don’t pin to 0°)
  - auto vertical scale, face-refined when faces are present
  - gentle contrast / color lift
- **Face-aware framing** (YuNet ONNX) when portal crop is enabled; subject-aware fallback when no faces
- **Batch scanning** with Enter between sheets; rejects stale scanner buffers so consecutive pages don’t duplicate
- **Background JPEG processing** so the next scan can start while the previous frame is still encoding
- **Reprocess saved `.raw`** files without rescanning
- Optional **`--smooth-grid`** mesh soften (Radon + spatial morph)

## Requirements

- Linux with **libusb-1.0**
- Python 3.10+ recommended
- Packages: `numpy`, `pillow`, `opencv-python-headless` (for faces / deskew helpers)
- Root or USB permissions for live scan (`sudo`, or udev rules + group membership)

```bash
cd hp-ps100
python3 -m venv .venv
.venv/bin/pip install numpy pillow opencv-python-headless
```

The face model ships in-repo: `data/face_detection_yunet_2023mar.onnx`.

## Quick start (vintage prints)

Plug in the PS100, load a photo in the feeder, then:

```bash
sudo .venv/bin/python photo_slideshow.py --old-photos --out ~/Pictures/ps100-album
```

- **Enter** — scan next sheet  
- **q** — quit  
- Outputs `photo_NNNN.jpg` (+ matching `.raw` for reprocessing; raws are gitignored)

### Batch of N sheets

```bash
sudo .venv/bin/python photo_slideshow.py --old-photos --out ~/Pictures/ps100-album --count 12
```

`--count` only caps how many scans; you still press Enter after loading each new print.

## Reprocess a raw

```bash
.venv/bin/python photo_slideshow.py --old-photos \
  --from-raw ~/Pictures/ps100-album/photo_0045.raw \
  --out ~/Pictures/ps100-album --start 45
```

`--from-raw` is repeatable. Add `--scan` to continue into live capture afterward.

## Useful options

| Flag | Purpose |
|------|---------|
| `--portal 1920x1080` / `4:3` / `3:2` | Portal size or named ratio |
| `--density 2` | Render scale vs portal (default 2; use `1` for 1:1 pixels) |
| `--enhance 1.3` | Stronger lift for faded prints (`--old-photos` uses 1.15) |
| `--yscale auto` / `0.85` | Vertical scale vs raw rows |
| `--deskew 6` / `--no-deskew` | Max auto-deskew degrees |
| `--smooth-grid` | Soften scanner/paper mesh (off unless set) |
| `--crop-to-portal` | Fill portal AR by cropping (face-aware when OpenCV is available) |
| `--no-keep-border` | Allow cover-crop even on bordered vintage prints |
| `--matte 28,26,24` | Letterbox RGB |
| `--start N` | Starting `photo_NNNN` index (0 = auto next free) |
| `-v` | Verbose USB / decode logging |

## One-shot capture (raw + PGM)

Lower-level single page without slideshow framing:

```bash
sudo .venv/bin/python scan.py --out /tmp/ps100
sudo .venv/bin/python scan.py --ping   # expect status 'NOVA'
```

## Layout

| Path | Role |
|------|------|
| `photo_slideshow.py` | Main scan → portal album pipeline |
| `protocol.py` | WorkScan sequence, progress/buffer wait, decode, y-scale |
| `bot.py` | libusb BOT transport for `03f0:53f0` |
| `scan.py` | Minimal raw/PGM capture |
| `data/face_detection_yunet_2023mar.onnx` | YuNet face model |
| `discover.py` / `probe.py` / `status.py` | Protocol exploration helpers |

## Notes

- Live scan needs the device on USB; put a sheet in the feeder before each capture.
- Album directories created under `sudo` end up root-owned — use `sudo chown` if you want to edit them as a normal user.
- Scan transfer time is mostly device + BOT burst cadence (~15 MB/page), not host USB port speed.
