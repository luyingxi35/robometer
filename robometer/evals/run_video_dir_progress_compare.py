#!/usr/bin/env python3
"""Run RL-aligned Robometer progress prediction over a directory of videos.

This script is intentionally video-directory based: it does not require a
processed HuggingFace dataset. For each input video it mirrors RLinf PushCube
absolute-reward preprocessing by uniformly downsampling the full trajectory to
``max_frames`` before POSTing one progress sample to the Robometer eval server.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
import requests

DEFAULT_TASK = "Push the cube across the tabletop until its center lies inside the target region."
DEFAULT_MODELS = [
    "robometer-4b=/data/yingxi/robometer/robometer-4b",
    "checkpoint-400-5tasks=/data/yingxi/robometer/checkpoint-400-5tasks",
]
MODEL_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str


def natural_key(path: Path) -> List[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.stem)]


def parse_model_specs(values: Optional[Sequence[str]]) -> List[ModelSpec]:
    raw = list(values) if values else list(DEFAULT_MODELS)
    specs: List[ModelSpec] = []
    for item in raw:
        if "=" not in item:
            raise ValueError(f"--model must be NAME=PATH, got: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"--model must be NAME=PATH, got: {item}")
        if not Path(path).exists():
            raise FileNotFoundError(f"Model path does not exist for {name}: {path}")
        specs.append(ModelSpec(name=name, path=path))
    if len(specs) < 2:
        raise ValueError("At least two --model entries are expected for comparison plots.")
    return specs


def robometer_downsample_indices(total: int, cap: int) -> List[int]:
    """Match RLinf _robometer_downsample_indices: linspace + int truncation + unique."""
    if total <= 0:
        return []
    if total <= cap:
        return list(range(total))
    idx = np.linspace(0, total - 1, cap).astype(int)
    seen = set()
    out: List[int] = []
    for value in idx:
        ivalue = int(value)
        if ivalue not in seen:
            seen.add(ivalue)
            out.append(ivalue)
    return out


def load_video_frames(video_path: Path) -> Tuple[np.ndarray, float]:
    import decord  # type: ignore

    vr = decord.VideoReader(str(video_path), num_threads=1)
    try:
        fps = float(vr.get_avg_fps())
    except Exception:
        fps = 0.0
    total = len(vr)
    if total <= 0:
        raise RuntimeError(f"Video has no frames: {video_path}")
    frames = vr.get_batch(list(range(total))).asnumpy()
    del vr
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    if frames.ndim != 4 or frames.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected frames shaped (T,H,W,C), got {frames.shape} for {video_path}")
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    return frames, fps


def np_to_npy_file_tuple(arr: np.ndarray, filename: str) -> Tuple[str, io.BytesIO, str]:
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    return filename, buf, "application/octet-stream"


def build_multipart_payload(samples: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    files: Dict[str, Any] = {}
    data: Dict[str, str] = {}
    numpy_fields = ["frames", "lang_vector", "video_embeddings"]
    for i, sample in enumerate(samples):
        sample_copy = json.loads(json.dumps(sample, default=str))
        traj = sample.get("trajectory", {})
        traj_copy = sample_copy.get("trajectory", {})
        for field in numpy_fields:
            val = traj.get(field)
            if val is None:
                continue
            if hasattr(val, "detach") and hasattr(val, "cpu"):
                val = val.detach().cpu().numpy()
            if isinstance(val, np.ndarray):
                key = f"sample_{i}_trajectory_{field}"
                files[key] = np_to_npy_file_tuple(val, f"{key}.npy")
                traj_copy[field] = {"__numpy_file__": key}
            else:
                traj_copy[field] = val
        if "frames_shape" in traj_copy and isinstance(traj_copy["frames_shape"], (tuple, list)):
            traj_copy["frames_shape"] = [int(x) for x in traj_copy["frames_shape"]]
        sample_copy["trajectory"] = traj_copy
        data[f"sample_{i}"] = json.dumps(sample_copy)
    data["use_frame_steps"] = "false"
    return files, data


def post_evaluate_batch_npy(server_url: str, samples: List[Dict[str, Any]], timeout_s: float) -> Dict[str, Any]:
    files, data = build_multipart_payload(samples)
    response = requests.post(
        server_url.rstrip("/") + "/evaluate_batch_npy",
        files=files,
        data=data,
        timeout=timeout_s,
    )
    response.raise_for_status()
    return response.json()


def make_progress_sample(
    frames: np.ndarray,
    task: str,
    sample_id: str,
    video_path: Path,
    sampled_indices: Sequence[int],
    total_frames: int,
    fps: float,
) -> Dict[str, Any]:
    return {
        "sample_type": "progress",
        "trajectory": {
            "frames": frames,
            "frames_shape": list(frames.shape),
            "task": task,
            "id": sample_id,
            "metadata": {
                "video_path": str(video_path),
                "frame_step": int(sampled_indices[-1]) if sampled_indices else -1,
                "sampled_frame_indices": [int(x) for x in sampled_indices],
                "num_frames": int(total_frames),
                "fps": float(fps),
            },
            "video_embeddings": None,
        },
    }


def interpolate_progress(total_frames: int, sampled_indices: Sequence[int], progress: Sequence[float]) -> List[float]:
    if total_frames <= 0:
        return []
    if not sampled_indices or not progress:
        return [0.0] * total_frames
    x = np.asarray(sampled_indices[: len(progress)], dtype=np.float32)
    y = np.asarray(progress[: len(sampled_indices)], dtype=np.float32)
    if x.size == 1:
        return [float(y[0])] * total_frames
    return np.interp(np.arange(total_frames, dtype=np.float32), x, y).astype(float).tolist()


def finite_clamped(values: Sequence[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            f = float(value)
        except Exception:
            f = 0.0
        if not math.isfinite(f):
            f = 0.0
        out.append(max(0.0, min(1.0, f)))
    return out


def extract_progress_lists(outputs: Dict[str, Any]) -> Tuple[List[List[float]], List[List[float]]]:
    progress_pred = (outputs.get("outputs_progress") or {}).get("progress_pred") or []
    success_probs = (outputs.get("outputs_success") or {}).get("success_probs") or []
    progress = [finite_clamped(row if isinstance(row, list) else [row]) for row in progress_pred]
    success = [finite_clamped(row if isinstance(row, list) else [row]) for row in success_probs]
    return progress, success


def wait_for_server(server_url: str, timeout_s: float, proc: Optional[subprocess.Popen[Any]] = None) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"Server process exited before becoming healthy at {server_url}; exit_code={proc.returncode}")
        try:
            response = requests.get(server_url.rstrip("/") + "/health", timeout=5.0)
            if response.ok and response.json().get("status") == "healthy":
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(3.0)
    raise TimeoutError(f"Server did not become healthy at {server_url}: {last_error}")


def read_config_base_model_id(config_path: Path) -> Optional[str]:
    if not config_path.is_file():
        return None
    for line in config_path.read_text().splitlines():
        match = re.match(r"^\s*base_model_id:\s*(.+?)\s*$", line)
        if match:
            return match.group(1).strip().strip(chr(34)).strip(chr(39))
    return None


def make_config_override_if_needed(model: ModelSpec, log_dir: Path, fallback_base_model_id: Optional[str]) -> Optional[Path]:
    if not fallback_base_model_id:
        return None
    fallback = Path(fallback_base_model_id)
    if not fallback.exists():
        return None
    config_path = Path(model.path) / "config.yaml"
    base_model_id = read_config_base_model_id(config_path)
    if not base_model_id:
        return None
    if Path(base_model_id).exists():
        return None
    text = config_path.read_text()
    replaced = re.sub(
        r"(^\s*base_model_id:\s*).*$",
        lambda m: f"{m.group(1)}{fallback_base_model_id}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    override_path = log_dir / f"{model.name}_config.yaml"
    override_path.write_text(replaced)
    print(
        f"[config] {model.name}: base_model_id {base_model_id} not found; using {fallback_base_model_id}",
        flush=True,
    )
    return override_path


def start_server(
    model: ModelSpec,
    repo_dir: Path,
    gpu_id: str,
    port: int,
    server_batch_size: int,
    num_gpus: int,
    startup_timeout_s: float,
    log_dir: Path,
    fallback_base_model_id: Optional[str],
) -> subprocess.Popen[str]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{model.name}_server.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("TMPDIR", "/data/yingxi/tmp")
    env.setdefault("HF_HOME", "/data/yingxi/.cache/huggingface")
    env.setdefault("HF_DATASETS_CACHE", "/data/yingxi/.cache/huggingface/datasets")
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    config_override = make_config_override_if_needed(model, log_dir, fallback_base_model_id)
    if config_override is not None:
        env["ROBOMETER_CONFIG_YAML_PATH"] = str(config_override)
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "robometer/evals/eval_server.py",
        f"model_path={model.path}",
        "server_url=127.0.0.1",
        f"server_port={port}",
        f"num_gpus={num_gpus}",
        f"batch_size={server_batch_size}",
    ]
    log_fh = open(log_path, "w", buffering=1)
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_dir),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        wait_for_server(f"http://127.0.0.1:{port}", startup_timeout_s, proc=proc)
    except Exception:
        terminate_process(proc)
        raise
    return proc


def terminate_process(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


def run_model_inference(
    model: ModelSpec,
    video_paths: Sequence[Path],
    task: str,
    output_dir: Path,
    server_url: str,
    max_frames: int,
    request_batch_size: int,
    timeout_s: float,
) -> Dict[str, Any]:
    model_dir = output_dir / model.name
    model_dir.mkdir(parents=True, exist_ok=True)
    videos_out: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    prepared: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for video_path in video_paths:
        try:
            frames, fps = load_video_frames(video_path)
            total = int(frames.shape[0])
            sampled_indices = robometer_downsample_indices(total, max_frames)
            sampled_frames = frames[sampled_indices]
            sample_id = video_path.stem
            sample = make_progress_sample(
                sampled_frames,
                task,
                sample_id,
                video_path.resolve(),
                sampled_indices,
                total,
                fps,
            )
            meta = {
                "id": sample_id,
                "video_path": str(video_path.resolve()),
                "num_frames": total,
                "fps": fps,
                "sampled_frame_indices": sampled_indices,
            }
            prepared.append((sample, meta))
        except Exception as exc:  # noqa: BLE001
            failures.append({"video_path": str(video_path), "error": repr(exc)})

    for start in range(0, len(prepared), request_batch_size):
        chunk = prepared[start : start + request_batch_size]
        samples = [item[0] for item in chunk]
        metas = [item[1] for item in chunk]
        try:
            outputs = post_evaluate_batch_npy(server_url, samples, timeout_s=timeout_s)
            progress_lists, success_lists = extract_progress_lists(outputs)
            if len(progress_lists) != len(samples):
                raise RuntimeError(f"Expected {len(samples)} progress rows, got {len(progress_lists)}")
            for i, (meta, progress) in enumerate(zip(metas, progress_lists)):
                sampled_indices = meta["sampled_frame_indices"]
                progress = progress[: len(sampled_indices)]
                row = {
                    "id": meta["id"],
                    "video_path": meta["video_path"],
                    "num_frames": meta["num_frames"],
                    "fps": meta["fps"],
                    "sampled_frame_indices": [int(x) for x in sampled_indices[: len(progress)]],
                    "sampled_progress": [float(x) for x in progress],
                    "interpolated_progress": interpolate_progress(meta["num_frames"], sampled_indices, progress),
                }
                if i < len(success_lists):
                    row["sampled_success_probs"] = success_lists[i][: len(progress)]
                videos_out.append(row)
        except Exception as exc:  # noqa: BLE001
            for meta in metas:
                failures.append({"video_path": meta["video_path"], "error": repr(exc)})

    result = {
        "model_name": model.name,
        "model_path": model.path,
        "task": task,
        "max_frames": max_frames,
        "use_frame_steps": False,
        "num_videos": len(video_paths),
        "num_successful": len(videos_out),
        "num_failed": len(failures),
        "videos": sorted(videos_out, key=lambda r: natural_key(Path(r["video_path"]))),
        "failures": failures,
    }
    with open(model_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def crop_center_width(frame: np.ndarray, width_fraction: float) -> np.ndarray:
    if width_fraction <= 0 or width_fraction >= 1.0:
        return frame
    width = int(frame.shape[1])
    crop_width = max(1, int(round(width * width_fraction)))
    left = max(0, (width - crop_width) // 2)
    return frame[:, left : left + crop_width]


def choose_thumbnail_indices(sampled_indices: Sequence[int], thumbnail_frames: int) -> List[int]:
    if not sampled_indices:
        return []
    if len(sampled_indices) <= thumbnail_frames:
        return [int(x) for x in sampled_indices]
    positions = np.linspace(0, len(sampled_indices) - 1, thumbnail_frames).astype(int)
    return [int(sampled_indices[int(pos)]) for pos in positions]


def plot_trajectory(
    video_path: Path,
    rows_by_model: Dict[str, Dict[str, Any]],
    model_order: Sequence[str],
    output_path: Path,
    thumbnail_frames: int,
    plot_interpolated: bool,
    thumbnail_crop_fraction: float,
) -> None:
    frames, fps = load_video_frames(video_path)
    raw_total = int(frames.shape[0])
    first_row = next(iter(rows_by_model.values()))
    total = min(raw_total, max(1, int(first_row.get("num_frames", raw_total))))
    thumb_indices = choose_thumbnail_indices(first_row["sampled_frame_indices"], thumbnail_frames)
    fig = plt.figure(figsize=(16, 7.5))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.35], hspace=0.16)
    ax_img = fig.add_subplot(grid[0, 0])
    ax_curve = fig.add_subplot(grid[1, 0])

    if thumb_indices:
        xspan = max(float(total), 1.0)
        if len(thumb_indices) > 1:
            min_gap = min(max(1, b - a) for a, b in zip(thumb_indices[:-1], thumb_indices[1:]))
        else:
            min_gap = max(total, 1)
        zoom_width_px = max(24.0, (min_gap / xspan) * fig.get_figwidth() * fig.dpi * 0.82)
        for frame_idx in thumb_indices:
            frame = frames[min(max(frame_idx, 0), raw_total - 1)]
            frame = crop_center_width(frame, thumbnail_crop_fraction)
            zoom = zoom_width_px / max(float(frame.shape[1]), 1.0)
            image = OffsetImage(frame, zoom=zoom, interpolation="nearest")
            image.image.axes = ax_img
            ab = AnnotationBbox(
                image,
                (frame_idx, 0.5),
                xycoords="data",
                frameon=False,
                box_alignment=(0.5, 0.5),
                pad=0.0,
            )
            ax_img.add_artist(ab)
            ax_img.axvline(frame_idx, color="black", alpha=0.18, linewidth=0.8)
            ax_curve.axvline(frame_idx, color="black", alpha=0.12, linewidth=0.8)
    ax_img.set_ylim(0.0, 1.0)
    ax_img.set_yticks([])
    ax_img.set_xlim(-1, max(total - 1, 1))
    ax_img.set_xticks(thumb_indices)
    if fps and fps > 0:
        ax_img.set_xticklabels([f"{x}\n{x / fps:.1f}s" for x in thumb_indices], fontsize=7)
    else:
        ax_img.set_xticklabels([str(x) for x in thumb_indices], fontsize=7)
    ax_img.tick_params(axis="x", labelbottom=True, pad=1)
    ax_img.set_title(f"{video_path.name} sampled thumbnails aligned to original frame index", fontsize=10)

    for idx, model_name in enumerate(model_order):
        row = rows_by_model[model_name]
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        x = np.asarray(row["sampled_frame_indices"], dtype=np.float32)
        y = np.asarray(row["sampled_progress"], dtype=np.float32)
        if plot_interpolated and row.get("interpolated_progress"):
            ax_curve.plot(
                np.arange(len(row["interpolated_progress"])),
                row["interpolated_progress"],
                color=color,
                alpha=0.25,
                linewidth=1.5,
            )
        ax_curve.plot(x, y, marker="o", markersize=3.5, linewidth=2.0, color=color, label=model_name)

    ax_curve.set_ylim(-0.03, 1.03)
    ax_curve.set_xlim(-1, max(total - 1, 1))
    ax_curve.set_xticks(thumb_indices)
    ax_curve.set_xticklabels([str(x) for x in thumb_indices], fontsize=8)
    ax_curve.set_ylabel("Robometer progress")
    ax_curve.set_xlabel("original video frame index")
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(loc="best", fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_overview(
    output_path: Path,
    common_ids: Sequence[str],
    rows: Dict[str, Dict[str, Dict[str, Any]]],
    model_order: Sequence[str],
    max_cols: int,
) -> None:
    if not common_ids:
        return
    cols = min(max_cols, len(common_ids))
    rows_n = int(math.ceil(len(common_ids) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3.5, rows_n * 2.4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for plot_idx, traj_id in enumerate(common_ids):
        ax = axes[plot_idx // cols][plot_idx % cols]
        ax.axis("on")
        for model_idx, model_name in enumerate(model_order):
            row = rows[model_name][traj_id]
            ax.plot(
                row["sampled_frame_indices"],
                row["sampled_progress"],
                marker="o",
                markersize=2.0,
                linewidth=1.2,
                color=MODEL_COLORS[model_idx % len(MODEL_COLORS)],
                label=model_name if plot_idx == 0 else None,
            )
        ax.set_title(traj_id, fontsize=8)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), fontsize=9)
    fig.suptitle("Robometer progress comparison over sampled RL-aligned frames", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_visualizations(
    results_by_model: Dict[str, Dict[str, Any]],
    video_paths: Sequence[Path],
    output_dir: Path,
    thumbnail_frames: int,
    plot_interpolated: bool,
    overview_cols: int,
    thumbnail_crop_fraction: float,
) -> Dict[str, Any]:
    comparison_dir = output_dir / "comparison"
    plot_dir = comparison_dir / "trajectory_plots"
    model_order = list(results_by_model.keys())
    rows: Dict[str, Dict[str, Dict[str, Any]]] = {
        name: {row["id"]: row for row in result["videos"]}
        for name, result in results_by_model.items()
    }
    common_ids = sorted(set.intersection(*(set(model_rows.keys()) for model_rows in rows.values())), key=lambda x: natural_key(Path(x)))
    video_by_id = {path.stem: path for path in video_paths}
    for traj_id in common_ids:
        if traj_id not in video_by_id:
            continue
        plot_trajectory(
            video_path=video_by_id[traj_id],
            rows_by_model={model: rows[model][traj_id] for model in model_order},
            model_order=model_order,
            output_path=plot_dir / f"{traj_id}.png",
            thumbnail_frames=thumbnail_frames,
            plot_interpolated=plot_interpolated,
            thumbnail_crop_fraction=thumbnail_crop_fraction,
        )
    plot_overview(
        output_path=comparison_dir / "all_trajectories_grid.png",
        common_ids=common_ids,
        rows=rows,
        model_order=model_order,
        max_cols=overview_cols,
    )
    return {
        "common_trajectory_ids": common_ids,
        "num_common_trajectories": len(common_ids),
        "trajectory_plot_dir": str(plot_dir),
        "overview_plot": str(comparison_dir / "all_trajectories_grid.png"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--model", action="append", help="Model spec NAME=PATH. Repeat for comparisons.")
    parser.add_argument("--max-frames", type=int, default=30, help="RLinf-aligned Robometer frame cap.")
    parser.add_argument("--thumbnail-frames", type=int, default=15)
    parser.add_argument("--thumbnail-crop-fraction", type=float, default=0.5, help="Center crop width fraction for thumbnails; 1.0 disables crop.")
    parser.add_argument("--request-batch-size", type=int, default=4)
    parser.add_argument("--server-batch-size", type=int, default=16)
    parser.add_argument("--server-port", type=int, default=8011)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--server-startup-timeout-s", type=float, default=600.0)
    parser.add_argument("--overview-cols", type=int, default=5)
    parser.add_argument("--limit-videos", type=int, default=None, help="Optional smoke-test limit; default uses all videos.")
    parser.add_argument("--no-interpolated", action="store_true", help="Only plot sampled points, not per-frame interpolation.")
    parser.add_argument("--keep-server", action="store_true", help="Do not terminate launched server processes.")
    parser.add_argument("--fallback-base-model-id", default="/data/yingxi/robometer/Qwen3-VL-4B-Instruct", help="Runtime config fallback used when a checkpoint config references a missing local base_model_id.")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[2]
    if not args.video_dir.exists():
        raise FileNotFoundError(args.video_dir)
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.thumbnail_frames <= 0:
        raise ValueError("--thumbnail-frames must be positive")
    if args.request_batch_size <= 0:
        raise ValueError("--request-batch-size must be positive")
    if not (0.0 < float(args.thumbnail_crop_fraction) <= 1.0):
        raise ValueError("--thumbnail-crop-fraction must be in (0, 1]")
    if args.limit_videos is not None and args.limit_videos <= 0:
        raise ValueError("--limit-videos must be positive when set")

    models = parse_model_specs(args.model)
    video_paths = sorted(args.video_dir.glob("*.mp4"), key=natural_key)
    if args.limit_videos is not None:
        video_paths = video_paths[: int(args.limit_videos)]
    if not video_paths:
        raise FileNotFoundError(f"No .mp4 files found in {args.video_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    server_url = f"http://127.0.0.1:{args.server_port}"
    results_by_model: Dict[str, Dict[str, Any]] = {}
    launched: List[subprocess.Popen[Any]] = []
    start_time = time.time()

    try:
        for model in models:
            print(f"[server] starting {model.name}: {model.path}", flush=True)
            proc = start_server(
                model=model,
                repo_dir=repo_dir,
                gpu_id=str(args.gpu_id),
                port=int(args.server_port),
                server_batch_size=int(args.server_batch_size),
                num_gpus=int(args.num_gpus),
                startup_timeout_s=float(args.server_startup_timeout_s),
                log_dir=args.output_dir / "server_logs",
                fallback_base_model_id=args.fallback_base_model_id,
            )
            launched.append(proc)
            print(f"[infer] {model.name}: {len(video_paths)} videos", flush=True)
            results_by_model[model.name] = run_model_inference(
                model=model,
                video_paths=video_paths,
                task=args.task,
                output_dir=args.output_dir,
                server_url=server_url,
                max_frames=int(args.max_frames),
                request_batch_size=int(args.request_batch_size),
                timeout_s=float(args.timeout_s),
            )
            if not args.keep_server:
                print(f"[server] stopping {model.name}", flush=True)
                terminate_process(proc)
                launched.remove(proc)
                time.sleep(5.0)

        viz_report = create_visualizations(
            results_by_model=results_by_model,
            video_paths=video_paths,
            output_dir=args.output_dir,
            thumbnail_frames=int(args.thumbnail_frames),
            plot_interpolated=not args.no_interpolated,
            overview_cols=int(args.overview_cols),
            thumbnail_crop_fraction=float(args.thumbnail_crop_fraction),
        )
        run_report = {
            "task": args.task,
            "video_dir": str(args.video_dir.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "models": [{"name": m.name, "path": m.path} for m in models],
            "max_frames": int(args.max_frames),
            "thumbnail_frames": int(args.thumbnail_frames),
            "thumbnail_crop_fraction": float(args.thumbnail_crop_fraction),
            "use_frame_steps": False,
            "request_batch_size": int(args.request_batch_size),
            "limit_videos": args.limit_videos,
            "server_batch_size": int(args.server_batch_size),
            "num_input_videos": len(video_paths),
            "elapsed_seconds": time.time() - start_time,
            "visualization": viz_report,
            "failures": {name: result.get("failures", []) for name, result in results_by_model.items()},
        }
        comparison_dir = args.output_dir / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        with open(comparison_dir / "run_report.json", "w") as f:
            json.dump(run_report, f, indent=2)
        print(json.dumps(run_report, indent=2), flush=True)
    finally:
        if not args.keep_server:
            for proc in list(launched):
                terminate_process(proc)


if __name__ == "__main__":
    main()
