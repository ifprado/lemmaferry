#!/usr/bin/env python3
"""Train a small causal next-token student and an exactly matched label control.

The student starts from random weights, sees question/rules only, and learns one
answer token per example. This is a restricted synthetic supervised task, not
general-language-model distillation or a pretrained-model capability claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import resource
import sys
import time

import numpy as np


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+|[0-9]|[^\w\s]", text)


def run(args):
    import torch
    from torch import nn

    if args.output.exists():
        raise ValueError("Choose a new output directory")
    checkpoints = args.checkpoints or Path("runs") / args.output.name / "checkpoints"
    if checkpoints.exists():
        raise ValueError("Choose a new checkpoint directory")
    data = {split: [json.loads(line) for line in (args.inputs / f"{split}.jsonl").read_text().splitlines()]
            for split in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if {r["lineage"] for r in data[a]} & {r["lineage"] for r in data[b]}:
            raise ValueError("Underlying graph lineage crosses splits")
    words = sorted({t for row in data["train"] for t in tokens(row["ordinary_text"])})
    vocab = {token: i for i, token in enumerate(["<PAD>", "<UNK>"] + words)}
    encoded = {split: [[vocab.get(t, 1) for t in tokens(r["ordinary_text"])] for r in rows]
               for split, rows in data.items()}
    if max(len(seq) for rows in encoded.values() for seq in rows) > 1024:
        raise ValueError("Student prompt exceeds frozen context1024")
    gold = {split: [vocab[r["answer"]] for r in rows] for split, rows in data.items()}
    labels = [vocab["A"], vocab["B"]]

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.words = nn.Embedding(len(vocab), 128, padding_idx=0)
            self.positions = nn.Embedding(1024, 128)
            block = nn.TransformerEncoderLayer(128, 4, 512, dropout=0.0,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.layers = nn.TransformerEncoder(block, 2, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(128)
            self.head = nn.Linear(128, len(vocab))

        def forward(self, ids, lengths):
            positions = torch.arange(ids.shape[1], device=ids.device)
            x = self.words(ids) + self.positions(positions)[None]
            causal = torch.triu(torch.ones((ids.shape[1], ids.shape[1]), dtype=torch.bool, device=ids.device), diagonal=1)
            x = self.layers(x, mask=causal, src_key_padding_mask=(ids == 0))
            return self.head(self.norm(x[torch.arange(len(ids), device=ids.device), lengths - 1]))

    def batch(sequences):
        lengths = torch.tensor([len(s) for s in sequences], device=args.device)
        ids = torch.zeros((len(sequences), int(lengths.max())), dtype=torch.long, device=args.device)
        for i, sequence in enumerate(sequences):
            ids[i, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=args.device)
        return ids, lengths

    def evaluate(model, split):
        rows = []
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(data[split]), 8):
                logits = model(*batch(encoded[split][start:start + 8])).float().cpu()
                logp = torch.log_softmax(logits, dim=-1)
                for i, record in enumerate(data[split][start:start + 8]):
                    prediction = ("A", "B")[int(logits[i, labels].argmax())]
                    rows.append({"id": record["id"], "lineage": record["lineage"], "family": record["family"],
                                 "answer": record["answer"], "prediction": prediction,
                                 "correct": prediction == record["answer"],
                                 "answer_token_nll": float(-logp[i, vocab[record["answer"]]])})
        return {"count": len(rows), "two_choice_accuracy": float(np.mean([r["correct"] for r in rows])),
                "answer_token_nll": float(np.mean([r["answer_token_nll"] for r in rows])),
                "by_family": {f: float(np.mean([r["correct"] for r in rows if r["family"] == f]))
                              for f in sorted({r["family"] for r in rows})}, "records": rows}

    args.output.mkdir(parents=True)
    checkpoints.mkdir(parents=True)
    report = {"status": "started", "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Randomly initialized small causal next-token student on restricted synthetic graphs; no pretrained-general-model training or iterative generation.",
              "device": args.device, "dtype": "float32", "python": sys.version,
              "platform": platform.platform(),
              "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy")},
              "architecture": {"layers": 2, "hidden": 128, "heads": 4, "feedforward": 512, "context": 1024, "dropout": 0.0, "vocab_size": len(vocab)},
              "optimizer": {"name": "AdamW", "learning_rate": 0.001, "weight_decay": 0.01, "betas": [0.9, 0.95], "gradient_clip": 1.0},
              "selection": "No tuning or early stopping. Final epoch3 for all predeclared seeds1729,1730,1731.",
              "control": "Identical prompts, initialization, batch order, steps, and tokens; training answer labels permuted within train. This isolates correct answer supervision, not superiority over useful natural-language training.",
              "input_sha256": {split: hashlib.sha256((args.inputs / f"{split}.jsonl").read_bytes()).hexdigest() for split in data},
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "runs": []}
    output = args.output / "results.json"

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    started = time.perf_counter()
    try:
        for seed in (1729, 1730, 1731):
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
            initial = Student().state_dict()
            permutation = np.random.default_rng(seed + 999).permutation(len(data["train"]))
            shuffled = [gold["train"][int(i)] for i in permutation]
            for arm in ("verified_supervision", "permuted_labels"):
                model = Student().to(args.device)
                model.load_state_dict(initial)
                if arm == "verified_supervision":
                    untrained = evaluate(model, "test")
                training_labels = gold["train"] if arm == "verified_supervision" else shuffled
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.95), weight_decay=0.01)
                order_rng = np.random.default_rng(seed)
                updates = examples = prompt_tokens = padded_tokens = 0
                arm_start = time.perf_counter()
                model.train()
                for epoch in range(3):
                    order = order_rng.permutation(len(data["train"]))
                    epoch_losses = []
                    for start in range(0, len(order), 8):
                        indices = order[start:start + 8].tolist()
                        ids, lengths = batch([encoded["train"][i] for i in indices])
                        expected = torch.tensor([training_labels[i] for i in indices], device=args.device)
                        optimizer.zero_grad(set_to_none=True)
                        logits = model(ids, lengths)
                        loss = nn.functional.cross_entropy(logits, expected)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        updates += 1
                        examples += len(indices)
                        prompt_tokens += int(lengths.sum())
                        padded_tokens += ids.numel()
                        epoch_losses.append(float(loss.detach()))
                    print(f"seed{seed} {arm} epoch{epoch + 1}: loss{np.mean(epoch_losses):.4f}", flush=True)
                result = {"seed": seed, "arm": arm, "parameters": sum(p.numel() for p in model.parameters()),
                          "updates": updates, "supervised_answer_tokens": examples, "input_prompt_tokens": prompt_tokens,
                          "padded_input_tokens": padded_tokens, "training_seconds": time.perf_counter() - arm_start,
                          "validation": evaluate(model, "val"), "test": evaluate(model, "test")}
                checkpoint = checkpoints / f"seed{seed}-{arm}.pt"
                torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "vocabulary": vocab, "architecture": report["architecture"],
                            "input_sha256": report["input_sha256"], "seed": seed, "arm": arm}, checkpoint)
                result["checkpoint"] = {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
                if arm == "verified_supervision":
                    result["untrained_test"] = untrained
                report["runs"].append(result)
                save()
                del model, optimizer
                if args.device == "mps":
                    torch.mps.empty_cache()
        for seed in (1729, 1730, 1731):
            matched = [r for r in report["runs"] if r["seed"] == seed]
            for key in ("updates", "supervised_answer_tokens", "input_prompt_tokens", "padded_input_tokens", "parameters"):
                assert matched[0][key] == matched[1][key], (seed, key)
        report["summary"] = {arm: {"test_accuracy_mean": float(np.mean([r["test"]["two_choice_accuracy"] for r in report["runs"] if r["arm"] == arm])),
                                    "test_accuracy_std_across_seeds": float(np.std([r["test"]["two_choice_accuracy"] for r in report["runs"] if r["arm"] == arm]))}
                             for arm in ("verified_supervision", "permuted_labels")}
        delta = report["summary"]["verified_supervision"]["test_accuracy_mean"] - report["summary"]["permuted_labels"]["test_accuracy_mean"]
        report["summary"]["verified_minus_control_accuracy"] = delta
        report["summary"]["iteration_gate"] = "Only consider a later iterative experiment if fixed-data gains replicate across held-out grammars and stronger useful-data controls. This label-control run alone does not justify iteration."
        report.update(status="completed", total_seconds=time.perf_counter() - started)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["peak_process_rss_bytes"] = peak if sys.platform == "darwin" else peak * 1024
        save()
        print(json.dumps(report["summary"], indent=2), flush=True)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoints", type=Path, help="New directory for six independently reusable state dictionaries")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    run(parser.parse_args())
