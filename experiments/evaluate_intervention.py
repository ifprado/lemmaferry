#!/usr/bin/env python3
"""Score proof-assisted last-token activation interventions on frozen Qwen2.

Only the next answer token is scored. The proof-text control sees the exact
donor prompt. Prefix caching is checked against a full ordinary forward pass.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import random
import resource
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_activations import paired_collections
from lemmaferry.alignment import AffineRidge


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def layer_module(model, index: int):
    if model.config.model_type != "qwen2" or not 0 < index < model.config.num_hidden_layers:
        raise ValueError("This runner supports interior hidden_states boundaries of Qwen2 only")
    return model.model.layers[index - 1]


def forward_state(model, tokens: dict, index: int, use_cache: bool = False):
    captured = []

    def capture(_module, _args, output):
        if not hasattr(output, "shape"):
            raise ValueError("Unexpected Qwen2 decoder output; inspect the installed implementation")
        captured.append(output[:, -1, :].detach().clone())

    hook = layer_module(model, index).register_forward_hook(capture)
    try:
        out = model.model(**tokens, use_cache=use_cache, return_dict=True)
    finally:
        hook.remove()
    return model.lm_head(out.last_hidden_state[:, -1, :]), captured[0], out.past_key_values


def replay_patch(model, tokens: dict, index: int, vector, cache=None):
    import torch
    from transformers.cache_utils import DynamicCache

    def replace(_module, _args, output):
        patched = output.clone()
        patched[:, -1, :] = vector.to(device=output.device, dtype=output.dtype)
        return patched

    hook = layer_module(model, index).register_forward_hook(replace)
    try:
        if cache is None:
            out = model.model(**tokens, use_cache=False, return_dict=True)
        else:
            # Each mode gets an independent mutable cache. Earlier causal
            # positions cannot depend on a patch at the final prompt position.
            legacy = cache.to_legacy_cache()
            length = tokens["input_ids"].shape[1]
            prefix = tuple((k[..., :length - 1, :].clone(), v[..., :length - 1, :].clone())
                           for k, v in legacy)
            fresh = DynamicCache.from_legacy_cache(prefix)
            out = model.model(input_ids=tokens["input_ids"][:, -1:],
                              attention_mask=tokens["attention_mask"],
                              past_key_values=fresh, use_cache=False,
                              cache_position=torch.tensor([length - 1], device=tokens["input_ids"].device),
                              return_dict=True)
    finally:
        hook.remove()
    return model.lm_head(out.last_hidden_state[:, -1, :])


def label_ids(tokenizer, text: str) -> list[int]:
    prefix = tokenizer.encode(text, add_special_tokens=False)
    result = []
    for label in ("A", "B"):
        combined = tokenizer.encode(text + " " + label, add_special_tokens=False)
        if combined[:-1] != prefix or len(combined) != len(prefix) + 1:
            raise ValueError("Each answer must be one continuation token without retokenizing the prompt")
        result.append(combined[-1])
    return result


def score_logits(logits, labels: list[int], answer: str) -> dict:
    import torch
    values = logits.detach().float().cpu()[0]
    selected = values[labels]
    gold = ("A", "B").index(answer)
    return {"prediction": ("A", "B")[int(selected.argmax())],
            "correct": bool(int(selected.argmax()) == gold),
            "answer_token_nll": float(-torch.log_softmax(values, dim=-1)[labels[gold]]),
            "two_choice_nll": float(-torch.log_softmax(selected, dim=-1)[gold]),
            "greedy_token_correct": bool(int(values.argmax()) == labels[gold]),
            "choice_token_ids": labels}


def summarize(records: list[dict]) -> dict:
    return {"count": len(records),
            "two_choice_accuracy": float(np.mean([r["correct"] for r in records])),
            "answer_token_nll": float(np.mean([r["answer_token_nll"] for r in records])),
            "two_choice_nll": float(np.mean([r["two_choice_nll"] for r in records])),
            "greedy_token_accuracy": float(np.mean([r["greedy_token_correct"] for r in records]))}


def paired_interval(rows: list[dict], left: str, right: str, seed: int = 1729) -> dict:
    groups = {}
    for row in rows:
        groups.setdefault(row["lineage"], []).append(row)
    names = sorted(groups)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(2000):
        chosen = rng.integers(len(names), size=len(names))
        values = [float(row["scores"][left]["correct"]) - float(row["scores"][right]["correct"])
                  for i in chosen for row in groups[names[int(i)]]]
        samples.append(float(np.mean(values)))
    differences = [float(row["scores"][left]["correct"]) - float(row["scores"][right]["correct"]) for row in rows]
    return {"accuracy_difference": float(np.mean(differences)),
            "lineage_cluster_bootstrap_95_percent_interval": np.quantile(samples, [0.025, 0.975]).tolist(),
            "underlying_lineages": len(names), "test_grammars": len({r["family"] for r in rows}),
            "scope": "Conditional on the two chosen graph grammars; too few independent grammars for broad family-level claims."}


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen2 import modeling_qwen2

    if args.output.exists():
        raise ValueError("Choose a new output directory to retain previous attempts")
    sm, sx, tm, ty = paired_collections(args.source, args.target)
    fit = json.loads((args.translator / "results.json").read_text())
    if fit["source"]["manifest_sha256"] != digest(args.source / "manifest.json") or fit["target"]["manifest_sha256"] != digest(args.target / "manifest.json"):
        raise ValueError("Translator collections do not match the current manifests")
    if fit["map"]["sha256"] != digest(args.translator / "translator.npz"):
        raise ValueError("Translator hash mismatch")
    if fit["seed"] != args.seed:
        raise ValueError("Control seed must match the registered fitter seed")
    if digest(args.inputs / "test.jsonl") != sm["inputs"]["test"]["sha256"]:
        raise ValueError("Evaluation inputs differ from collection inputs")
    records = [json.loads(line) for line in (args.inputs / "test.jsonl").read_text().splitlines()]
    assert [r["id"] for r in records] == sx["test"]["ids"].tolist()
    if any(r["text"] != r["text_baseline"] for r in records):
        raise ValueError("Text baseline must receive identical donor information")
    index = sm["layer_index"]
    if tm["layer_index"] != index:
        raise ValueError("Source and target use different layer boundaries")
    manifest = sm if args.source_only else tm
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    common = dict(revision=manifest["revision"], trust_remote_code=False, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(manifest["model"], use_fast=True, **common)
    if hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest() != manifest["tokenizer_sha256"]:
        raise ValueError("Runtime tokenizer differs from collection")
    for r in records:
        label_ids(tokenizer, r["ordinary_text"])
        label_ids(tokenizer, r["text"])
    args.output.mkdir(parents=True)
    report = {"status": "started", "created_utc": datetime.now(timezone.utc).isoformat(),
              "experiment": "proof_assisted_next_token_intervention", "seed": args.seed,
              "source_only": args.source_only, "model": manifest["model"], "revision": manifest["revision"],
              "layer_index": index, "position": "last_nonpadding_token", "device": args.device,
              "dtype": "float32", "python": sys.version, "platform": platform.platform(),
              "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")},
              "input_sha256": digest(args.inputs / "test.jsonl"),
              "source_manifest_sha256": digest(args.source / "manifest.json"),
              "target_manifest_sha256": digest(args.target / "manifest.json"),
              "translator_report_sha256": digest(args.translator / "results.json"),
              "code_sha256": {"runner": digest(Path(__file__)), "qwen2_implementation": digest(Path(inspect.getfile(modeling_qwen2)))},
              "self_patch_tolerance": {"absolute": 1e-4, "relative": 1e-4},
              "information_policy": "Donor and proof-text baseline see identical certificate and question. Ordinary recipient sees question and rules only.",
              "control_policy": "Shuffled donor rows use a seeded random derangement. Translated zero donor means T(0)=intercept, not a zero recipient state.",
              "scope": "One constrained next-token choice; no claim of multi-token reasoning or standalone learning."}
    report_path = args.output / "results.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    started = time.perf_counter()
    try:
        model = AutoModelForCausalLM.from_pretrained(manifest["model"], torch_dtype=torch.float32,
                                                    use_safetensors=True, **common).to(args.device).eval()
        model.requires_grad_(False)
        report["load_seconds"] = time.perf_counter() - started
        with np.load(args.translator / "translator.npz", allow_pickle=False) as z:
            coef, intercept, mean = z["coef"], z["intercept"], z["target_mean"]
        donor = sx["test"]["vectors"]
        predicted = donor @ coef.T + intercept
        shuffled = AffineRidge(alpha=fit["selected_alpha"]).fit(
            sx["train"]["vectors"], ty["train"]["vectors"][np.random.default_rng(args.seed).permutation(len(ty["train"]["vectors"]))])
        wrong_predictions = shuffled.predict(donor)
        rng = np.random.default_rng(args.seed)
        donor_permutation = rng.permutation(len(records))
        while np.any(donor_permutation == np.arange(len(records))):
            donor_permutation = rng.permutation(len(records))
        report["donor_derangement"] = donor_permutation.tolist()
        results = []
        max_self_error = 0.0
        with torch.inference_mode():
            for i, r in enumerate(records):
                def tokenize(text):
                    return {k: v.to(args.device) for k, v in tokenizer(text, add_special_tokens=False, return_tensors="pt").items()}
                assisted = tokenize(r["text"])
                assisted_logits, assisted_state, _ = forward_state(model, assisted, index)
                expected_state = sx["test"]["vectors"][i] if args.source_only else ty["test"]["vectors"][i]
                if not np.allclose(assisted_state.float().cpu().numpy()[0], expected_state, atol=1e-4, rtol=1e-4):
                    raise ValueError("Runtime model state differs from the collected interface")
                row = {"id": r["id"], "family": r["family"], "lineage": r["lineage"],
                       "answer": r["answer"], "scores": {}}
                if args.source_only:
                    row["scores"]["source_proof_text"] = score_logits(assisted_logits, label_ids(tokenizer, r["text"]), r["answer"])
                else:
                    ordinary = tokenize(r["ordinary_text"])
                    base, own_state, cache = forward_state(model, ordinary, index, use_cache=True)
                    labels = label_ids(tokenizer, r["ordinary_text"])
                    row["scores"]["ordinary_recipient"] = score_logits(base, labels, r["answer"])
                    row["scores"]["proof_text_recipient"] = score_logits(assisted_logits, label_ids(tokenizer, r["text"]), r["answer"])
                    patched_self = replay_patch(model, ordinary, index, own_state, cache)
                    self_error = float((patched_self - base).abs().max())
                    if not torch.allclose(patched_self, base, atol=1e-4, rtol=1e-4):
                        # Some kernels differ between full prefill and cached
                        # one-token execution. Fall back to the exact full
                        # prompt policy instead of weakening the tolerance.
                        cache = None
                        patched_self = replay_patch(model, ordinary, index, own_state)
                        self_error = float((patched_self - base).abs().max())
                        if not torch.allclose(patched_self, base, atol=1e-4, rtol=1e-4):
                            raise ValueError("Self-patch fails the declared logit tolerance")
                    max_self_error = max(max_self_error, self_error)
                    row["self_patch_max_absolute_logit_error"] = self_error
                    row["cached_prefix_used"] = cache is not None
                    row["scores"]["self_patch"] = score_logits(patched_self, labels, r["answer"])
                    controls = {"translated_donor": predicted[i], "shuffled_training_pairs": wrong_predictions[i],
                                "shuffled_donor_rows": predicted[donor_permutation[i]],
                                "translated_zero_donor": intercept, "recipient_training_mean": mean,
                                "oracle_text_state_patch": assisted_state,
                                "raw_identity_donor": donor[i]}
                    for mode, values in controls.items():
                        vector = torch.as_tensor(values, device=args.device, dtype=torch.float32).reshape(1, -1)
                        logits = replay_patch(model, ordinary, index, vector, cache)
                        row["scores"][mode] = score_logits(logits, labels, r["answer"])
                    del cache
                results.append(row)
                if (i + 1) % 16 == 0:
                    print(f"Scored {i + 1}/{len(records)}", flush=True)
            modes = list(results[0]["scores"])
            report["metrics"] = {mode: summarize([row["scores"][mode] for row in results]) for mode in modes}
            report["by_family"] = {family: {mode: summarize([row["scores"][mode] for row in results if row["family"] == family]) for mode in modes}
                                   for family in sorted({row["family"] for row in results})}
            if not args.source_only:
                report["paired_comparisons"] = {baseline: paired_interval(results, "translated_donor", baseline, args.seed)
                                                for baseline in ("ordinary_recipient", "proof_text_recipient")}
                report["max_self_patch_absolute_logit_error"] = max_self_error
            report["records"] = results
        report["total_seconds"] = time.perf_counter() - started
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["peak_process_rss_bytes"] = peak if sys.platform == "darwin" else peak * 1024
        report["status"] = "completed"
        save()
        print(json.dumps(report["metrics"], indent=2), flush=True)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "target", "translator", "inputs", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--seed", type=int, default=1729)
    run(parser.parse_args())
