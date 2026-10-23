#!/usr/bin/env python3
"""Uniformly sample frames from a video into a capture folder for Real2Sim/COLMAP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ~N evenly spaced frames from a video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("2.mp4"),
        help="Input video path (relative paths are repo-root relative).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/exp1/images"),
        help="Capture directory. Frames are written to <out_dir>/images/.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=250,
        help="Target number of frames to save.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="frame_",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".jpg",
        choices=[".jpg", ".jpeg", ".png"],
        help="Output image extension.",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=95,
        help="JPEG quality (0-100), ignored for PNG.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear existing images/ before writing.",
    )
    return parser.parse_args()


def resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]

    video_path = resolve_path(args.video, repo_root)
    out_dir = resolve_path(args.out_dir, repo_root)
    images_dir = out_dir / "images"

    if not video_path.is_file():
        print(f"Error: video not found: {video_path}", file=sys.stderr)
        return 1
    if args.num_frames <= 0:
        print("Error: --num_frames must be > 0", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: failed to open video: {video_path}", file=sys.stderr)
        return 1

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if total <= 0:
        cap.release()
        print("Error: video reports 0 frames.", file=sys.stderr)
        return 1

    num_save = min(args.num_frames, total)
    # Evenly spaced unique frame indices covering the whole clip.
    indices = np.linspace(0, total - 1, num=num_save, dtype=np.int64)
    indices = np.unique(indices)
    num_save = len(indices)

    if args.overwrite and images_dir.exists():
        for old in images_dir.glob(f"{args.prefix}*{args.ext}"):
            old.unlink()
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Video: {video_path}")
    print(f"  frames={total} fps={fps:.3f} size={width}x{height}")
    print(f"Saving {num_save} frames -> {images_dir}")

    write_params = []
    if args.ext in {".jpg", ".jpeg"}:
        write_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]

    saved = 0
    for out_i, frame_i in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_i))
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"Warning: failed to read frame index {frame_i}", file=sys.stderr)
            continue
        out_path = images_dir / f"{args.prefix}{out_i:04d}{args.ext}"
        if not cv2.imwrite(str(out_path), frame, write_params):
            print(f"Warning: failed to write {out_path}", file=sys.stderr)
            continue
        saved += 1

    cap.release()
    print(f"Done. Wrote {saved} images.")
    return 0 if saved > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
