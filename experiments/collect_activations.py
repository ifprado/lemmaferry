#!/usr/bin/env python3
"""Collect real model activations for the optional LLM alignment milestone.

No models are downloaded unless this script is explicitly run. See
docs/llm-experiment.md. This is an activation collector, not an intervention or
model-training implementation. Run --help without installing torch.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import sys
import time


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def peak_rss_bytes() -> int | None:
    """Process peak resident memory; macOS reports bytes and Linux reports KiB."""
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        return None
    return int(value if sys.platform == "darwin" else value * 1024)


def load_splits(paths: dict[str, Path]) -> dict[str, list[dict[str, str]]]:
    """Require disjoint IDs, families, and whitespace-normalized exact texts."""
    seen_ids: set[str] = set()
    family_split: dict[str, str] = {}
    text_split: dict[str, str] = {}
    result: dict[str, list[dict[str, str]]] = {}
    for split, path in paths.items():
        rows: list[dict[str, str]] = []
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("id", "family", "text"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"{path}:{line_no}: nonempty {key!r} required")
            if row["id"] in seen_ids:
                raise ValueError(f"Repeated ID: {row['id']}")
            seen_ids.add(row["id"])
            family = row["family"]
            if family in family_split and family_split[family] != split:
                raise ValueError(f"Family {family!r} appears across splits")
            family_split[family] = split
            text_key = " ".join(row["text"].split())
            if text_key in text_split and text_split[text_key] != split:
                raise ValueError(f"Repeated text across splits: {row['id']}")
            text_split[text_key] = split
            rows.append(row)
        if not rows:
            raise ValueError(f"Empty split: {path}")
        result[split] = rows
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Hugging Face model repository ID")
    p.add_argument("--revision", required=True, help="Exact 40-character model commit SHA")
    for split in ("train", "val", "test"):
        p.add_argument(f"--{split}", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New output directory")
    p.add_argument("--layer", type=int, default=14, help="Literal hidden_states tuple index")
    p.add_argument("--max-length", type=int, default=256, help="Reject longer inputs; never truncate")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--expected-tokenizer-sha", help="Fail if the tokenizer differs from the other model")
    return p


def main() -> None:
    started = time.perf_counter()
    args = parser().parse_args()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        raise SystemExit("--revision must be an exact 40-character commit SHA, not 'main'.")
    if args.layer < 0 or args.max_length < 1 or args.batch_size < 1:
        raise SystemExit("Layer must be nonnegative; length and batch size must be positive.")
    paths = {split: getattr(args, split) for split in ("train", "val", "test")}
    splits = load_splits(paths)
    if args.output.exists():
        raise SystemExit("Output directory already exists. Choose a new directory to preserve runs.")

    # Set this before importing torch or creating CUDA tensors.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Optional dependencies missing. See docs/llm-experiment.md.") from exc

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but unavailable.")
    if device == "cpu" and args.dtype == "float16":
        raise SystemExit("Use float32 for the CPU baseline.")

    common = dict(revision=args.revision, trust_remote_code=False,
                  local_files_only=args.local_files_only)
    tokenizer_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, **common)
    if not tokenizer.is_fast:
        raise SystemExit("This collector requires a fast tokenizer to record its full fingerprint.")
    tokenizer_sha = sha256(tokenizer.backend_tokenizer.to_str().encode())
    if args.expected_tokenizer_sha and args.expected_tokenizer_sha != tokenizer_sha:
        raise SystemExit("Tokenizer fingerprint mismatch. Do not assume token positions correspond.")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise SystemExit("Tokenizer has neither pad nor eos token; choose an explicit padding policy.")
        tokenizer.pad_token = tokenizer.eos_token

    # Validate lengths before downloading model weights. Base-model completion
    # prompts are literal text: no implicit chat template or appended answer.
    tokenized: dict[str, list[list[int]]] = {}
    for split, rows in splits.items():
        values = tokenizer([r["text"] for r in rows], add_special_tokens=False,
                           truncation=False, padding=False)["input_ids"]
        for row, ids in zip(rows, values):
            if not ids or len(ids) > args.max_length:
                raise ValueError(f"{row['id']}: {len(ids)} tokens; allowed 1..{args.max_length}")
        tokenized[split] = values

    args.output.mkdir(parents=True)
    manifest = {
        "status": "started", "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "revision": args.revision.lower(),
        "tokenizer_revision": args.revision.lower(), "tokenizer_sha256": tokenizer_sha,
        "layer_index": args.layer, "position": "last_nonpadding_token",
        "input_format": "literal_text_no_chat_template_no_special_tokens",
        "pooling": "none", "padding_side": "right", "seed": args.seed,
        "eval_mode": True, "inference_mode": True, "trust_remote_code": False,
        "use_safetensors": True, "device": device, "dtype": args.dtype,
        "max_length": args.max_length, "batch_size": args.batch_size,
        "deterministic_algorithms": "enabled_warn_only; not a cross-platform guarantee",
        "collector_sha256": sha256(Path(__file__).read_bytes()),
        "measurement": {
            "tokenizer_and_tokenization_wall_seconds": time.perf_counter() - tokenizer_started,
            "peak_process_rss_bytes": peak_rss_bytes(),
            "peak_process_rss_method": "getrusage(RUSAGE_SELF).ru_maxrss; process lifetime",
            "mps_peak_memory_api": "not_available; only sampled public allocator counters",
        },
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("numpy", "torch", "transformers", "huggingface-hub")},
        "inputs": {split: {"path": str(path), "sha256": sha256(path.read_bytes()),
                           "count": len(splits[split])} for split, path in paths.items()},
    }
    manifest_path = args.output / "manifest.json"

    def save_manifest() -> None:
        manifest["measurement"].update(
            total_wall_seconds=time.perf_counter() - started,
            peak_process_rss_bytes=peak_rss_bytes(),
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def sample_device_memory() -> None:
        if device != "mps":
            return
        torch.mps.synchronize()
        measurement = manifest["measurement"]
        for label, value in (("allocated", torch.mps.current_allocated_memory()),
                             ("driver_allocated", torch.mps.driver_allocated_memory())):
            key = f"mps_sampled_max_{label}_bytes"
            measurement[key] = max(measurement.get(key, 0), value)
        measurement["mps_memory_sampling"] = "after model load and each completed batch; not a continuous peak"

    save_manifest()
    try:
        model_started = time.perf_counter()
        model = AutoModel.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype),
                                          use_safetensors=True, **common)
        model.to(device)
        model.eval()
        model.requires_grad_(False)
        sample_device_memory()
        manifest["measurement"]["model_load_wall_seconds"] = time.perf_counter() - model_started
        model_source = Path(sys.modules[type(model).__module__].__file__)
        manifest["model_implementation"] = {"module": type(model).__module__,
                                            "sha256": sha256(model_source.read_bytes())}
        if args.layer > model.config.num_hidden_layers:
            raise ValueError(f"Layer {args.layer} exceeds {model.config.num_hidden_layers}")
        if args.max_length > model.config.max_position_embeddings:
            raise ValueError("max-length exceeds this model's configured positional limit")
        manifest["config"] = model.config.to_dict()
        outputs_meta = {}
        collection_started = time.perf_counter()
        with torch.inference_mode():
            for split, rows in splits.items():
                split_started = time.perf_counter()
                vectors = []
                token_hashes = []
                for start in range(0, len(rows), args.batch_size):
                    sequences = tokenized[split][start:start + args.batch_size]
                    batch = tokenizer.pad({"input_ids": sequences}, padding=True,
                                          return_attention_mask=True, return_tensors="pt")
                    batch = {key: value.to(device) for key, value in batch.items()}
                    out = model(**batch, output_hidden_states=True, use_cache=False,
                                return_dict=True)
                    states = out.hidden_states[args.layer]
                    mask = batch["attention_mask"]
                    positions = torch.arange(mask.shape[1], device=device).expand_as(mask)
                    last = positions.masked_fill(mask == 0, -1).max(dim=1).values
                    selected = states[torch.arange(states.shape[0], device=device), last]
                    vectors.append(selected.float().cpu().numpy())
                    sample_device_memory()
                    token_hashes.extend(sha256(json.dumps(ids).encode()) for ids in sequences)
                    del out, states, selected
                values = np.concatenate(vectors, axis=0)
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite activations; try float32 and inspect the model output.")
                output = args.output / f"{split}.npz"
                np.savez_compressed(output, vectors=values,
                                    ids=np.asarray([row["id"] for row in rows]),
                                    families=np.asarray([row["family"] for row in rows]),
                                    token_sha256=np.asarray(token_hashes),
                                    lengths=np.asarray([len(x) for x in tokenized[split]]))
                outputs_meta[split] = {"path": output.name, "shape": list(values.shape),
                                       "sha256": sha256(output.read_bytes()),
                                       "collection_wall_seconds": time.perf_counter() - split_started,
                                       "input_tokens": sum(len(x) for x in tokenized[split])}
                manifest["outputs"] = outputs_meta
                save_manifest()
        collection_seconds = time.perf_counter() - collection_started
        example_count = sum(len(rows) for rows in splits.values())
        manifest["measurement"].update(
            collection_wall_seconds=collection_seconds, collected_examples=example_count,
            examples_per_second=example_count / collection_seconds,
            input_tokens=sum(len(ids) for values in tokenized.values() for ids in values),
        )
        manifest["status"] = "completed"
        save_manifest()
        print(f"Collected activations in {args.output}; tokenizer SHA: {tokenizer_sha}")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        save_manifest()
        raise


if __name__ == "__main__":
    main()
