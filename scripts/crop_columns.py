"""Crop a page image into (optional header) + left column + right column.

Geometry is auto-detected per page:
- gutter_x: column boundary (global min of ink density in the central 30-70%
  band, computed over the lower 2/3 of the page so the header doesn't fool it).
- header_end: first row where the gutter region (gutter_x +/- 18 px) becomes
  consistently white, indicating the two-column layout has started. If no such
  row is found near the top, we assume there is no header.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
from PIL import Image


def detect_geometry(gray: np.ndarray) -> dict:
    H, W = gray.shape
    ink = (gray < 200).astype(np.uint8)

    # gutter_x
    band = ink[int(H * 0.33): int(H * 0.95), :]
    col = band.mean(axis=0)
    smooth = np.convolve(col, np.ones(25) / 25, mode="same")
    central = slice(int(W * 0.35), int(W * 0.65))
    gutter_x = central.start + int(np.argmin(smooth[central]))

    # header_end: the first row where the *right* column begins to have ink.
    # Anything above that row may be a centred title/author block spanning the
    # full page width, so we crop it as a single "header" image.
    row_density_full = ink.mean(axis=1)
    rows_with_text = np.where(row_density_full > 0.005)[0]
    top = int(rows_with_text[0]) if len(rows_with_text) else 0
    bot = int(rows_with_text[-1]) if len(rows_with_text) else H - 1

    # Use the interior of the right column (skip 60 px next to the gutter and
    # 60 px from the right edge) so page-edge speckle doesn't trigger us.
    right_interior = ink[:, gutter_x + 60: W - 60]
    right_row = right_interior.mean(axis=1)
    right_sm = np.convolve(right_row, np.ones(6) / 6, mode="same")
    # First row in the right column with substantial ink (a real text line).
    right_has_ink = np.where(right_sm > 0.04)[0]
    header_end = None
    if len(right_has_ink):
        first_right_text = int(right_has_ink[0])
        # If the right column starts noticeably below the top of the text
        # block, treat the rows above as a header. Otherwise, no header.
        if first_right_text > top + 60:
            # Back off to the last "white" row before the right column begins,
            # so we don't slice through a glyph.
            scan_from = max(first_right_text - 1, top)
            y = scan_from
            while y > top and right_sm[y] > 0.005:
                y -= 1
            header_end = y

    # column horizontal extents: leftmost/rightmost ink within each half
    left_band = ink[:, :gutter_x]
    right_band = ink[:, gutter_x:]
    left_cols = np.where(left_band.mean(axis=0) > 0.01)[0]
    right_cols = np.where(right_band.mean(axis=0) > 0.01)[0]
    left_x0 = int(left_cols[0]) if len(left_cols) else 0
    left_x1 = int(left_cols[-1]) if len(left_cols) else gutter_x - 1
    right_x0 = gutter_x + (int(right_cols[0]) if len(right_cols) else 0)
    right_x1 = gutter_x + (int(right_cols[-1]) if len(right_cols) else (W - gutter_x - 1))

    return {
        "size": (W, H),
        "gutter_x": gutter_x,
        "header_end": header_end,
        "text_top": top,
        "text_bot": bot,
        "left_x": (left_x0, left_x1),
        "right_x": (right_x0, right_x1),
    }


def crop_page(image_path: Path, out_dir: Path, pad: int = 12) -> list[Path]:
    im = Image.open(image_path)
    gray = np.asarray(im.convert("L"))
    g = detect_geometry(gray)
    W, H = g["size"]
    gx = g["gutter_x"]
    he = g["header_end"]
    top = max(g["text_top"] - pad, 0)
    bot = min(g["text_bot"] + pad, H - 1)
    lx0, lx1 = g["left_x"]
    rx0, rx1 = g["right_x"]
    lx0 = max(lx0 - pad, 0)
    lx1 = min(lx1 + pad, gx - 1)
    rx0 = max(rx0 - pad, gx + 1)
    rx1 = min(rx1 + pad, W - 1)

    stem = image_path.stem  # e.g. page-01
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    if he is not None and he > top + 20:
        header = im.crop((lx0, top, rx1, he))
        p = out_dir / f"{stem}-00-header.png"
        header.save(p)
        results.append(p)
        col_top = he
    else:
        col_top = top

    left = im.crop((lx0, col_top, lx1, bot))
    p = out_dir / f"{stem}-01-left.png"
    left.save(p)
    results.append(p)

    right = im.crop((rx0, col_top, rx1, bot))
    p = out_dir / f"{stem}-02-right.png"
    right.save(p)
    results.append(p)

    print(f"{image_path.name}: gutter_x={gx} header_end={he} -> {[r.name for r in results]}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pages_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--pad", type=int, default=12)
    args = ap.parse_args()
    inputs = sorted(args.pages_dir.glob("page-*.png"))
    if not inputs:
        print("no page-*.png inputs found", file=sys.stderr)
        sys.exit(1)
    for p in inputs:
        crop_page(p, args.out_dir, pad=args.pad)


if __name__ == "__main__":
    main()
