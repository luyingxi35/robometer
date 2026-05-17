#!/usr/bin/env python3

from pathlib import Path
import subprocess
import argparse


DEFAULT_VIDEO_DIR = (
    "/home/yingxi/RoboFAC/mani_envs/data_collection/"
    "progress_collection/PegInsertionVertical-v1/failure_labeled/videos"
)

DEFAULT_TASK = "Insert the blue peg into the orange hole."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_dir",
        type=str,
        default=DEFAULT_VIDEO_DIR,
        help="Directory containing mp4 videos",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help="Task description",
    )
    parser.add_argument(
        "--num_frames",
        type=str,
        default=30,
        help="Number of frames to process",
    )

    args = parser.parse_args()

    video_dir = Path(args.video_dir)

    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory does not exist: {video_dir}")

    mp4_files = sorted(video_dir.glob("*.mp4"))

    if len(mp4_files) == 0:
        print(f"No mp4 files found in: {video_dir}")
        return

    print(f"Found {len(mp4_files)} mp4 files.")

    for video_path in mp4_files:
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/example_inference_local.py",
            "--model-path",
            "/home/yingxi/robometer/robometer/Robometer-4B",
            "--video",
            str(video_path),
            "--task",
            args.task,
            "--num-frames",
            str(args.num_frames),
        ]

        print("\n" + "=" * 80)
        print("Running:")
        print(" ".join(cmd))
        print("=" * 80)

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"Failed on: {video_path}")
        else:
            print(f"Finished: {video_path}")


if __name__ == "__main__":
    main()
