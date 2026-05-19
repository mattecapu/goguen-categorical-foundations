"""Run tesseract on every cropped column image, writing one .txt per crop."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path


def run_tesseract(image: Path, out_stem: Path, lang: str, psm: int) -> None:
    cmd = [
        "tesseract", str(image), str(out_stem),
        "-l", lang,
        "--psm", str(psm),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"tesseract failed on {image}: {res.stderr}", file=sys.stderr)
        sys.exit(res.returncode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("columns_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--psm", type=int, default=6)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(args.columns_dir.glob("page-*.png"))
    if not images:
        print("no column images found", file=sys.stderr)
        sys.exit(1)
    for img in images:
        out_stem = args.out_dir / img.stem  # tesseract appends .txt
        run_tesseract(img, out_stem, args.lang, args.psm)
        print(f"ocr: {img.name} -> {out_stem.name}.txt")


if __name__ == "__main__":
    main()
