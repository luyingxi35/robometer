#!/usr/bin/env python3
"""
Convert RoboFAC formal1 replay videos into a local HuggingFace dataset for Robometer evaluation.

This script scans traj_* directories under a formal1 root, keeps only replay/third_rs.avi,
and writes a load_from_disk-compatible dataset directory. The dataset is intended to be
consumed by robometer.data.scripts.preprocess_local_hf_datasets before running evaluation.
"""

import argparse
import os
import re
import shutil
from collections import Counter

import numpy as np
from datasets import Dataset


DEFAULT_INPUT_ROOT = "/data/yingxi/robometer/formal1"
DEFAULT_TASK = "Insert the peg vertically into the target hole."
DEFAULT_DATA_SOURCE = "local_peg_insertion_formal1"
DEFAULT_SUCCESS_IDS = ""
TRAJ_PATTERN = re.compile(r"^traj_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert formal1 replay/third_rs.avi trajectories into a local HF dataset for Robometer."
    )
    parser.add_argument(
        "--input-root",
        default=DEFAULT_INPUT_ROOT,
        help=f"Path to the formal1 root directory. Default: {DEFAULT_INPUT_ROOT}",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save the output load_from_disk dataset")
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help=f"Task text stored in each example. Default: {DEFAULT_TASK}",
    )
    parser.add_argument(
        "--success-ids",
        default=DEFAULT_SUCCESS_IDS,
        help=f"Comma-separated traj indices treated as success. Default: {DEFAULT_SUCCESS_IDS}",
    )
    parser.add_argument(
        "--data-source",
        default=DEFAULT_DATA_SOURCE,
        help=f"data_source field to store in each example. Default: {DEFAULT_DATA_SOURCE}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow removing an existing output directory before saving the dataset.",
    )
    return parser.parse_args()


def parse_success_ids(raw_ids: str) -> set[int]:
    success_ids: set[int] = set()
    for token in raw_ids.split(","):
        token = token.strip()
        if not token:
            continue
        success_ids.add(int(token))
    return success_ids


def validate_args(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.input_root):
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")
    if os.path.exists(args.output_dir):
        if not args.overwrite:
            raise FileExistsError(f"Output path already exists: {args.output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(args.output_dir)


def count_video_frames(video_path: str) -> int:
    try:
        import decord  # type: ignore

        reader = decord.VideoReader(video_path, num_threads=1)
        frame_count = len(reader)
        del reader
    except Exception:
        import cv2

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()

    if frame_count <= 0:
        raise ValueError(f"Video has zero frames: {video_path}")
    return frame_count


def build_target_progress(frame_count: int, is_success: bool) -> list[float]:
    if is_success:
        return np.linspace(0.0, 1.0, num=frame_count, dtype=np.float32).tolist()
    return np.zeros(frame_count, dtype=np.float32).tolist()


def iter_formal1_rows(
    input_root: str,
    task: str,
    data_source: str,
    success_ids: set[int],
) -> tuple[list[dict], list[str], Counter, list[int]]:
    rows: list[dict] = []
    skipped: list[str] = []
    counts: Counter = Counter()
    missing_success_ids: list[int] = []

    for entry_name in sorted(os.listdir(input_root)):
        match = TRAJ_PATTERN.match(entry_name)
        if not match:
            continue

        traj_id = int(match.group(1))
        video_path = os.path.abspath(os.path.join(input_root, entry_name, "replay", "third_rs.avi"))
        if not os.path.exists(video_path):
            skipped.append(f"{entry_name}: missing replay/third_rs.avi")
            if traj_id in success_ids:
                missing_success_ids.append(traj_id)
            continue

        try:
            frame_count = count_video_frames(video_path)
        except Exception as exc:
            skipped.append(f"{entry_name}: unreadable replay/third_rs.avi ({exc})")
            if traj_id in success_ids:
                missing_success_ids.append(traj_id)
            continue

        is_success = traj_id in success_ids
        quality_label = "successful_labeled" if is_success else "failure_labeled"
        partial_success = 1.0 if is_success else 0.0

        rows.append(
            {
                "id": f"formal1_traj_{traj_id}",
                "task": task,
                "frames_video": video_path,
                "quality_label": quality_label,
                "partial_success": partial_success,
                "target_progress": build_target_progress(frame_count, is_success),
                "lang_vector": None,
                "is_robot": True,
                "data_source": data_source,
                "preference_group_id": None,
                "preference_rank": None,
            }
        )
        counts["valid_videos"] += 1
        counts[quality_label] += 1

    return rows, skipped, counts, sorted(missing_success_ids)


def print_summary(
    input_root: str,
    output_dir: str,
    total_traj_dirs: int,
    rows: list[dict],
    skipped: list[str],
    counts: Counter,
    success_ids: set[int],
    missing_success_ids: list[int],
) -> None:
    print(f"Input root: {input_root}")
    print(f"Output dir: {output_dir}")
    print(f"Total traj_* directories: {total_traj_dirs}")
    print(f"Valid replay videos: {counts['valid_videos']}")
    print(f"Skipped trajectories: {len(skipped)}")
    print(f"Configured success ids: {sorted(success_ids)}")
    print("Quality label counts:")
    print(f"  successful_labeled: {counts['successful_labeled']}")
    print(f"  failure_labeled: {counts['failure_labeled']}")

    if rows:
        print("Sample entries:")
        for sample in (rows[0], rows[-1] if len(rows) > 1 else None):
            if sample is None:
                continue
            print(
                "  "
                f"{sample['id']} -> {sample['quality_label']}, "
                f"frames_video={sample['frames_video']}, "
                f"target_progress_len={len(sample['target_progress'])}, "
                f"target_progress_last={sample['target_progress'][-1]:.1f}"
            )

    if skipped:
        print("Skipped details:")
        for item in skipped:
            print(f"  - {item}")

    if missing_success_ids:
        print(
            "Note: the following configured success ids were skipped because replay/third_rs.avi "
            f"was missing: {missing_success_ids}"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    success_ids = parse_success_ids(args.success_ids)

    total_traj_dirs = sum(
        1 for entry_name in os.listdir(args.input_root) if TRAJ_PATTERN.match(entry_name) is not None
    )
    rows, skipped, counts, missing_success_ids = iter_formal1_rows(
        input_root=args.input_root,
        task=args.task,
        data_source=args.data_source,
        success_ids=success_ids,
    )

    if not rows:
        raise RuntimeError("No valid replay/third_rs.avi trajectories were found.")

    dataset = Dataset.from_list(rows)
    dataset.save_to_disk(args.output_dir)

    print_summary(
        input_root=args.input_root,
        output_dir=args.output_dir,
        total_traj_dirs=total_traj_dirs,
        rows=rows,
        skipped=skipped,
        counts=counts,
        success_ids=success_ids,
        missing_success_ids=missing_success_ids,
    )


if __name__ == "__main__":
    main()
