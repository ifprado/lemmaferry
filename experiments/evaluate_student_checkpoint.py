#!/usr/bin/env python3
"""Evaluate a saved student in a separate process without loading either Qwen."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from train_fixed_student import build_student, tokens


def evaluate(checkpoint: Path, inputs: Path, device: str) -> dict:
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if hashlib.sha256(inputs.read_bytes()).hexdigest() != payload["input_sha256"]["test"]:
        raise ValueError("Test inputs differ from the frozen training experiment")
    vocab = payload["vocabulary"]
    model = build_student(len(vocab)).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    labels = [vocab["A"], vocab["B"]]
    data = [json.loads(line) for line in inputs.read_text().splitlines()]
    rows = []
    with torch.inference_mode():
        for start in range(0, len(data), 8):
            subset = data[start:start + 8]
            encoded = [[vocab.get(t, 1) for t in tokens(row["ordinary_text"])] for row in subset]
            lengths = torch.tensor([len(seq) for seq in encoded], device=device)
            ids = torch.zeros((len(subset), int(lengths.max())), dtype=torch.long, device=device)
            for i, seq in enumerate(encoded):
                ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            logits = model(ids, lengths).float().cpu()
            for i, record in enumerate(subset):
                prediction = ("A", "B")[int(logits[i, labels].argmax())]
                rows.append({"id": record["id"], "prediction": prediction, "answer": record["answer"],
                             "correct": prediction == record["answer"]})
    return {"status": "completed", "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "seed": payload["seed"], "arm": payload["arm"], "device": device,
            "count": len(rows), "two_choice_accuracy": float(np.mean([r["correct"] for r in rows])),
            "scope": "Standalone small synthetic student; no source model, recipient Qwen, translator, or proof certificate is supplied to the model.", "records": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="mps", choices=("mps", "cpu", "cuda"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Choose a new output path")
    report = evaluate(args.checkpoint, args.inputs, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Standalone accuracy: {report['two_choice_accuracy']:.6f}")
