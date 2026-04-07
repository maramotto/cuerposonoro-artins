#!/usr/bin/env python3
"""Headless camera diagnostic script.

Captures frames from the default camera and prints live stats to the
console.  Designed to run on the Jetson Orin Nano without any display.
No cv2.imshow(), no Qt, no GUI of any kind.

Usage:
    python tests/manual/manual_camera.py
    python tests/manual/manual_camera.py --duration 10
    python tests/manual/manual_camera.py --save-frames 30 --duration 60
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless camera diagnostic — no display required",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0,
        help="Stop after N seconds (default: 0 = run until Ctrl+C)",
    )
    parser.add_argument(
        "--save-frames",
        type=int,
        default=0,
        metavar="N",
        help="Save every Nth frame as JPEG (default: 0 = disabled)",
    )
    parser.add_argument(
        "--save-dir",
        default="tests/manual/logs/frames",
        help="Directory for saved frames (default: tests/manual/logs/frames)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera device {args.device}", file=sys.stderr)
        sys.exit(1)

    # Read one frame to get actual resolution
    ret, frame = cap.read()
    if not ret:
        print("ERROR: cannot read from camera", file=sys.stderr)
        cap.release()
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"Camera opened: {w}x{h} on /dev/video{args.device}")

    if args.save_frames > 0:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Saving every {args.save_frames} frames to {args.save_dir}/")

    if args.duration > 0:
        print(f"Running for {args.duration}s")
    else:
        print("Running until Ctrl+C")

    # Counters
    total_frames = 0
    failed_frames = 0
    interval_frames = 0
    start_time = time.monotonic()
    interval_start = start_time

    running = True

    def on_signal(signum: int, frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    # The first frame was already read above — count it
    total_frames = 1
    interval_frames = 1

    while running:
        ret, frame = cap.read()
        if not ret:
            failed_frames += 1
            continue

        total_frames += 1
        interval_frames += 1

        # Save frame if requested
        if args.save_frames > 0 and total_frames % args.save_frames == 0:
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(
                args.save_dir, f"frame_{ts}_{total_frames:06d}.jpg",
            )
            cv2.imwrite(filename, frame)

        # Print stats every second
        now = time.monotonic()
        elapsed_interval = now - interval_start
        if elapsed_interval >= 1.0:
            fps = interval_frames / elapsed_interval
            total_elapsed = now - start_time
            avg_fps = total_frames / total_elapsed if total_elapsed > 0 else 0
            print(
                f"[{total_elapsed:6.1f}s] "
                f"fps={fps:5.1f}  avg={avg_fps:5.1f}  "
                f"frames={total_frames}  failed={failed_frames}"
            )
            interval_frames = 0
            interval_start = now

        # Duration limit
        if args.duration > 0 and (now - start_time) >= args.duration:
            break

    cap.release()

    # Final summary
    total_elapsed = time.monotonic() - start_time
    avg_fps = total_frames / total_elapsed if total_elapsed > 0 else 0
    print()
    print("--- Summary ---")
    print(f"Duration:      {total_elapsed:.1f}s")
    print(f"Total frames:  {total_frames}")
    print(f"Failed frames: {failed_frames}")
    print(f"Average FPS:   {avg_fps:.1f}")


if __name__ == "__main__":
    main()
