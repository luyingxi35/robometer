#!/usr/bin/env python3
"""
Split a local Hugging Face `load_from_disk` dataset into train/eval datasets.

Example:
    uv run python robometer/scripts/split_local_hf_dataset.py                 --input /data/yingxi/robometer/progress_collection_randomized/PegInsertionVertical-v1/hf_dataset                 --train-output /data/yingxi/robometer/progress_collection_randomized/PegInsertionVertical-v1/hf_dataset_train                 --eval-output /data/yingxi/robometer/progress_collection_randomized/PegInsertionVertical-v1/hf_dataset_eval                 --seed 42
"""

import argparse
import os
import random
from collections import Counter

from datasets import Dataset, load_from_disk

REQUIRED_EVAL_LABEL_COUNTS = {
    "successful_labeled": 30,
    "failure_labeled": 30,
    "suboptimal_labeled": 30,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a local HF dataset into train/eval datasets using fixed quality_label sampling."
    )
    parser.add_argument("--input", required=True, help="Path to the input load_from_disk dataset directory")
    parser.add_argument("--train-output", required=True, help="Path to save the train split")
    parser.add_argument("--eval-output", help="Path to save the eval split")
    parser.add_argument("--test-output", help="Deprecated alias for --eval-output")
    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help="Deprecated and ignored. Eval split size is fixed at 30 per required quality_label.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible shuffling")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output directories",
    )
    args = parser.parse_args()
    if args.eval_output and args.test_output and args.eval_output != args.test_output:
        raise ValueError("--eval-output and --test-output must match when both are provided.")
    args.eval_output = args.eval_output or args.test_output
    if not args.eval_output:
        raise ValueError("One of --eval-output or --test-output is required.")
    return args


def validate_output_path(path: str, overwrite: bool) -> None:
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"Output path already exists: {path}. Use --overwrite to replace it.")


def get_quality_label_counts(dataset: Dataset) -> Counter:
    return Counter(dataset["quality_label"])


def select_eval_indices(dataset: Dataset, seed: int) -> tuple[list[int], Counter]:
    if "quality_label" not in dataset.column_names:
        raise ValueError("Input dataset must contain a 'quality_label' column.")

    label_to_indices: dict[str, list[int]] = {label: [] for label in REQUIRED_EVAL_LABEL_COUNTS}
    for idx, quality_label in enumerate(dataset["quality_label"]):
        if quality_label in label_to_indices:
            label_to_indices[quality_label].append(idx)

    available_counts = Counter({label: len(indices) for label, indices in label_to_indices.items()})
    insufficient = {
        label: required
        for label, required in REQUIRED_EVAL_LABEL_COUNTS.items()
        if available_counts[label] < required
    }
    if insufficient:
        counts_str = ", ".join(f"{label}={available_counts[label]}" for label in REQUIRED_EVAL_LABEL_COUNTS)
        required_str = ", ".join(f"{label}={count}" for label, count in REQUIRED_EVAL_LABEL_COUNTS.items())
        raise ValueError(
            "Input dataset does not have enough rows for fixed eval sampling. "
            f"Required counts: {required_str}. Available counts: {counts_str}."
        )

    rng = random.Random(seed)
    eval_indices: list[int] = []
    eval_counts = Counter()
    for label, required_count in REQUIRED_EVAL_LABEL_COUNTS.items():
        indices = list(label_to_indices[label])
        rng.shuffle(indices)
        selected = indices[:required_count]
        eval_indices.extend(selected)
        eval_counts[label] = len(selected)

    return eval_indices, eval_counts


def main() -> None:
    args = parse_args()

    dataset = load_from_disk(args.input)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected Dataset from load_from_disk, got: {type(dataset)}")

    validate_output_path(args.train_output, args.overwrite)
    validate_output_path(args.eval_output, args.overwrite)

    total_rows = len(dataset)
    source_counts = get_quality_label_counts(dataset)
    eval_indices, eval_counts = select_eval_indices(dataset, seed=args.seed)

    eval_index_set = set(eval_indices)
    train_indices = [idx for idx in range(total_rows) if idx not in eval_index_set]

    eval_dataset = dataset.select(eval_indices)
    train_dataset = dataset.select(train_indices)

    train_dataset.save_to_disk(args.train_output)
    eval_dataset.save_to_disk(args.eval_output)

    print(f"Input dataset: {args.input}")
    print(f"Total rows: {total_rows}")
    print("Source quality_label counts:")
    for label, count in sorted(source_counts.items()):
        print(f"  {label}: {count}")
    print("Eval quality_label counts:")
    for label in REQUIRED_EVAL_LABEL_COUNTS:
        print(f"  {label}: {eval_counts[label]}")
    print(f"Train rows: {len(train_dataset)} -> {args.train_output}")
    print(f"Eval rows: {len(eval_dataset)} -> {args.eval_output}")
    print(f"Seed: {args.seed}")

    if args.test_size is not None:
        print("Note: --test-size is deprecated and ignored.")


if __name__ == "__main__":
    main()
