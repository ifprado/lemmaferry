#!/usr/bin/env python3
"""Independent AI review: replay one fixed Qwen own-state control without runner helpers."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


started = time.perf_counter()
torch.manual_seed(1729)
torch.use_deterministic_algorithms(True, warn_only=True)
manifest = json.loads(Path("runs/transfer-v1/target/manifest.json").read_text())
row = json.loads(Path("data/transfer-v1/anchors/test.jsonl").read_text().splitlines()[0])
common = dict(revision=manifest["revision"], trust_remote_code=False, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(manifest["model"], use_fast=True, **common)
model = AutoModelForCausalLM.from_pretrained(manifest["model"], torch_dtype=torch.float32,
                                            use_safetensors=True, **common).to("mps").eval()
model.requires_grad_(False)
tokens = {key: value.to("mps") for key, value in
          tokenizer(row["ordinary_text"], add_special_tokens=False, return_tensors="pt").items()}
assert torch.all(tokens["attention_mask"] == 1)
captured = []


def capture(_module, _inputs, output):
    assert isinstance(output, torch.Tensor)
    captured.append(output[:, -1, :].detach().clone())


with torch.inference_mode():
    handle = model.model.layers[13].register_forward_hook(capture)
    try:
        base = model(**tokens, use_cache=True, output_hidden_states=True,
                     logits_to_keep=1, return_dict=True)
    finally:
        handle.remove()
    own = base.hidden_states[14][:, -1, :].detach().clone()
    assert torch.equal(captured[0], own)
    baseline = base.logits[:, -1, :].detach().clone()
    legacy = base.past_key_values.to_legacy_cache()
    before = tuple((key.clone(), value.clone()) for key, value in legacy)
    length = tokens["input_ids"].shape[1]
    prefix = tuple((key[..., :length - 1, :].clone(), value[..., :length - 1, :].clone())
                   for key, value in legacy)
    fresh = DynamicCache.from_legacy_cache(prefix)

    def patch(_module, _inputs, output):
        changed = output.clone()
        changed[:, -1, :] = own
        return changed

    handle = model.model.layers[13].register_forward_hook(patch)
    try:
        cached = model(input_ids=tokens["input_ids"][:, -1:],
                       attention_mask=tokens["attention_mask"], past_key_values=fresh,
                       cache_position=torch.tensor([length - 1], device="mps"),
                       use_cache=False, output_hidden_states=False, logits_to_keep=1,
                       return_dict=True).logits[:, -1, :].detach().clone()
        full = model(**tokens, use_cache=False, output_hidden_states=False,
                     logits_to_keep=1, return_dict=True).logits[:, -1, :].detach().clone()
    finally:
        handle.remove()
    unchanged = all(torch.equal(key, old_key) and torch.equal(value, old_value)
                    for (key, value), (old_key, old_value) in zip(legacy, before))
    assert unchanged
    checks = {}
    for name, logits in (("cached_prefix", cached), ("full_prompt", full)):
        checks[name] = {"maximum_absolute_logit_error": float((logits - baseline).abs().max()),
                        "allclose_at_fixed_tolerance": bool(torch.allclose(logits, baseline,
                                                                           atol=1e-4, rtol=1e-4))}
    assert checks["cached_prefix"]["allclose_at_fixed_tolerance"]
    assert checks["full_prompt"]["allclose_at_fixed_tolerance"]
    labels = [tokenizer.encode(" " + label, add_special_tokens=False)[0] for label in ("A", "B")]
    for label, identifier in zip(("A", "B"), labels):
        assert tokenizer.encode(row["ordinary_text"] + " " + label, add_special_tokens=False) == tokens["input_ids"][0].cpu().tolist() + [identifier]
    values = baseline[0].float().cpu()
    ordinary = {"prediction": ("A", "B")[int(values[labels].argmax())],
                "answer_token_nll": float(-torch.log_softmax(values, dim=-1)[labels[("A", "B").index(row["answer"])]]),
                "choice_token_ids": labels}

result = {"status": "passed", "reviewer": "delegated cousin AI agent; not human signoff",
          "completed_utc": datetime.now(timezone.utc).isoformat(), "example_id": row["id"],
          "model": manifest["model"], "revision": manifest["revision"], "device": "mps",
          "dtype": "float32", "layer_index": 14, "position": "last_nonpadding_token",
          "captured_block13_output_equals_hidden_states14": True,
          "tolerance": {"absolute": 1e-4, "relative": 1e-4}, "self_patch_checks": checks,
          "ordinary_scores": ordinary, "original_kv_cache_unchanged": unchanged,
          "source_hashes": {str(Path(__file__)): digest(__file__),
                            "experiments/evaluate_intervention.py": digest("experiments/evaluate_intervention.py")},
          "input_sha256": digest("data/transfer-v1/anchors/test.jsonl"),
          "recipient_report_sha256": digest("artifacts/transfer-v1/recipient-evaluation/results.json"),
          "elapsed_seconds": time.perf_counter() - started}
Path("runs/transfer-v1/independent-control.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
