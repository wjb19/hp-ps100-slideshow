# HP PS100 — resume notes

Last working session (2026-09-16): Linux scan → portal slideshow pipeline.

## Where things live
- Code: this repo (`photo_slideshow.py`, `bot.py`, `protocol.py`, `scan.py`, …)
- Face model: `data/face_detection_yunet_2023mar.onnx`
- Album output (local, not in repo): `~/Pictures/ps100-album-out/`
- Venv (local): `.venv/` — recreate with OpenCV if needed

## Typical commands
```bash
# vintage prints (recommended)
sudo .venv/bin/python photo_slideshow.py --old-photos --out ~/Pictures/ps100-album-out

# optional mesh/grid soften (off unless flagged)
sudo .venv/bin/python photo_slideshow.py --old-photos --smooth-grid --out ~/Pictures/ps100-album-out

# reprocess a saved raw
.venv/bin/python photo_slideshow.py --old-photos --from-raw ~/Pictures/ps100-album-out/photo_010.raw \
  --out ~/Pictures/ps100-album-out --start 10
```

## Pipeline highlights already in code
- BOT C5/C3 scan transport; row-sbs RGB decode; auto yscale / deskew / bed trim
- Portal fit with white-border AR keep; face-aware YuNet crop; subject-aware crop when no faces
- Default 2× render density (e.g. 3840×2160 from 1080p)
- `--smooth-grid`: opt-in Radon-guided destripe + spatial morph (FFT dropped — caused edge spectral bands)

## Open / next
- Live scan more prints; tune `--smooth-grid` further if mesh still strong
- Album `.raw` files stay on disk only (gitignored)
