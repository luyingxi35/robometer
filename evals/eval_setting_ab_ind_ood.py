#!/usr/bin/env python3
"""Four-GPU resumable IND/OOD sweep for the setting A/B checkpoints."""
from __future__ import annotations
import argparse, importlib.util, json, os, subprocess, sys
from pathlib import Path

from datasets import concatenate_datasets, load_from_disk

BASE_PATH = Path(__file__).with_name("eval_natural_five_task.py")
SPEC = importlib.util.spec_from_file_location("robometer_eval_base", BASE_PATH)
base = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(base)
SETTINGS = {
    "A": Path("/data/yingxi/robometer/setting_a_nine_task_20260924/training_500step_single_image/setting_a_nine_task_fsdp_single_image"),
    "B": Path("/data/yingxi/robometer/setting_b_nine_task_20260924/training_500step_single_image/setting_b_nine_task_fsdp_single_image"),
}

def read_processed(cache_root: Path, key: str):
    ds = load_from_disk(str(cache_root / key / "processed_dataset"))
    return [dict(r) for r in ds]

def evaluate(setting: str, gpu: int, shard: int, world: int, args):
    cache = args.cache_root
    ind = read_processed(cache, "setting_ab_ind_test")
    ood = read_processed(cache, "setting_ab_ood_test")
    rows = ind + ood
    checkpoints = base.discover_checkpoints(args.checkpoint_dir or SETTINGS[setting])
    checkpoints = [c for i, c in enumerate(checkpoints) if i % world == shard]
    if not checkpoints:
        print(f"GPU {gpu}: no checkpoints assigned")
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    out = args.output_root / setting
    scratch = out / f"_worker_gpu{gpu}"
    scratch.mkdir(parents=True, exist_ok=True)
    for checkpoint in checkpoints:
        step = base.step_from_checkpoint(checkpoint)
        step_dir = scratch / f"step_{step:06d}"
        run_args = argparse.Namespace(output_dir=step_dir, top_k=args.top_k,
            pearson_window=5, pearson_threshold=-0.5, batch_size=args.batch_size,
            max_frames=args.max_frames, force=args.force)
        base.run_checkpoint(run_args, checkpoint, rows)
        record_path = step_dir / f"step_{step:06d}_records.json"
        records = json.loads(record_path.read_text())
        by_split = {
            "IND": [r for r in records if r.get("id") in {x["id"] for x in ind}],
            "OOD": [r for r in records if r.get("id") in {x["id"] for x in ood}],
        }
        metrics = []
        for split, split_records in by_split.items():
            for metric in base.build_metrics(split_records, step, str(checkpoint), args.top_k, 5, -0.5):
                metric["split"] = split
                metrics.append(metric)
        for metric in base.build_metrics(records, step, str(checkpoint), args.top_k, 5, -0.5):
            metric["split"] = "IND_OOD_POOLED"
            metrics.append(metric)
        base.atomic_json(out / f"step_{step:06d}_metrics_ind_ood.json", metrics)
        base.atomic_json(out / f"step_{step:06d}_records_ind_ood.json", by_split)
        base.atomic_json(out / f"step_{step:06d}_worker.json", {
            "setting": setting, "gpu": gpu, "checkpoint": str(checkpoint),
            "ind_n": len(by_split["IND"]), "ood_n": len(by_split["OOD"]),
        })

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setting", choices=("A", "B", "both"), default="both")
    p.add_argument("--ind-root", type=Path)
    p.add_argument("--ood-root", type=Path)
    p.add_argument("--cache-root", type=Path, default=Path("/data/yingxi/robometer/setting_ab_eval_data/processed"))
    p.add_argument("--output-root", type=Path, default=Path("/data/yingxi/robometer/setting_ab_eval"))
    p.add_argument("--checkpoint-dir", type=Path)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-frames", type=int, default=8)
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--force", action="store_true")
    p.add_argument("--worker", nargs=3, metavar=("SETTING", "GPU", "SHARD"))
    args = p.parse_args()
    if args.worker:
        setting, gpu, shard = args.worker
        evaluate(setting, int(gpu), int(shard), args.gpus, args)
        return
    settings = list(SETTINGS) if args.setting == "both" else [args.setting]
    script = str(Path(__file__).resolve())
    for setting in settings:
        procs = []
        for gpu in range(args.gpus):
            cmd = [sys.executable, script, "--setting", setting, "--cache-root", str(args.cache_root),
                   "--output-root", str(args.output_root), "--gpus", str(args.gpus),
                   "--top-k", str(args.top_k), "--batch-size", str(args.batch_size),
                   "--max-frames", str(args.max_frames), "--worker", setting, str(gpu), str(gpu)]
            if args.checkpoint_dir:
                cmd += ["--checkpoint-dir", str(args.checkpoint_dir)]
            if args.force:
                cmd += ["--force"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            log = args.output_root / setting / f"worker_gpu{gpu}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w") as f:
                procs.append((subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env), f))
        bad = []
        for proc, handle in procs:
            code = proc.wait(); handle.close()
            if code: bad.append(code)
        if bad:
            raise SystemExit(f"setting {setting}: worker exit codes {bad}")
        step_records = sorted((args.output_root / setting).glob("step_*_worker.json"))
        summary = [json.loads(x.read_text()) for x in step_records]
        base.atomic_json(args.output_root / setting / "sweep_summary.json", summary)

if __name__ == "__main__":
    main()
