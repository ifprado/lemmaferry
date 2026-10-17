"""Persistent recipient learning from accepted, verified mathematical experience.

Torch is optional and imported only by recipient backends. The tiny byte model is
an executable pipeline check; it makes no claim about general-purpose capability.
The optional local Hugging Face backend updates a residual adapter in an actual
pretrained language model. Evaluation data is frozen before the first update and
never passed to curriculum scoring. Neither backend downloads models by default.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping, Sequence


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _example(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    for key in ("id", "prompt", "answer"):
        if not isinstance(result.get(key), str) or not result[key]:
            raise ValueError(f"example {key} must be a nonempty string")
    choices = list(result.get("choices", ["A", "B"]))
    if len(choices) < 2 or len(set(choices)) != len(choices) or result["answer"] not in choices:
        raise ValueError("each example requires distinct answer choices including its answer")
    if any(not isinstance(choice, str) or not choice for choice in choices):
        raise ValueError("answer choices must be nonempty strings")
    result.update(choices=choices, family=str(result.get("family", "unspecified")),
                  split=str(result.get("split", "train")), provenance=dict(result.get("provenance", {})))
    return result


@dataclass(frozen=True)
class FrozenEvaluation:
    """A file-bound, independently specified task distribution, never curriculum."""
    path: str
    sha256: str
    examples_sha256: str
    examples: tuple[dict[str, Any], ...]

    def verify(self) -> None:
        path = Path(self.path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.sha256:
            raise ValueError("fixed evaluation changed after registration")
        if _digest(list(self.examples)) != self.examples_sha256:
            raise ValueError("in-memory fixed evaluation changed after registration")


def freeze_evaluation(path: Path | str, examples: Iterable[Mapping[str, Any]]) -> FrozenEvaluation:
    """Create an immutable evaluation registration; never overwrite a different P."""
    path = Path(path)
    rows = [_example(row) for row in examples]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("evaluation must be nonempty and contain unique IDs")
    for row in rows:
        row["split"] = "independent_evaluation"
        row["provenance"]["use"] = "independent_evaluation_only"
    value = {"schema_version": 1, "purpose": "fixed independent evaluation; not curriculum",
             "registered_at_utc": datetime.now(timezone.utc).isoformat(),
             "examples_sha256": _digest(rows), "examples": rows}
    if path.exists():
        existing = load_evaluation(path)
        if existing.examples_sha256 != value["examples_sha256"]:
            raise ValueError("refusing to replace an existing evaluation distribution")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return load_evaluation(path)


def load_evaluation(path: Path | str) -> FrozenEvaluation:
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not value.get("examples"):
        raise ValueError("not a frozen evaluation file")
    rows = tuple(_example(row) for row in value["examples"])
    if any(r["split"] != "independent_evaluation" or
           r["provenance"].get("use") != "independent_evaluation_only" for r in rows):
        raise ValueError("evaluation purpose markers are missing")
    if _digest(list(rows)) != value["examples_sha256"]:
        raise ValueError("evaluation examples checksum mismatch")
    return FrozenEvaluation(str(path), hashlib.sha256(path.read_bytes()).hexdigest(),
                            value["examples_sha256"], rows)


def _disjoint(rows: Sequence[Mapping[str, Any]], evaluation: FrozenEvaluation) -> None:
    evaluation.verify()
    ids = {r["id"] for r in evaluation.examples}
    prompts = {r["prompt"] for r in evaluation.examples}
    for row in rows:
        if (row["id"] in ids or row["prompt"] in prompts or
                row.get("split") in {"test", "independent_evaluation"} or
                row.get("provenance", {}).get("use") == "independent_evaluation_only"):
            raise ValueError("independent evaluation/test data cannot enter training or feedback")


def build_experience(records: Iterable[Mapping[str, Any]], *, library_hash: str) -> list[dict[str, Any]]:
    """Render accepted library rows into supervised language choices.

    Caller must obtain records from the checked library's authenticated reader;
    this renderer is not a Lean verifier. ``library_hash`` binds the accepted
    snapshot. Explicit ``prompt``/``answer`` rows are also supported for alternate
    verified domains. Graph rows compare a valid shortest proof to a longer proof
    and explicitly ask for the shorter chain, not whether either is a proof.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", library_hash):
        raise ValueError("accepted library snapshot must have a SHA256 hash")
    result = []
    for wrapper in records:
        record = dict(wrapper if "learning_example" in wrapper else
                      wrapper.get("example", wrapper.get("record", wrapper.get("payload", wrapper))))
        identifier = str(wrapper.get("id", record.get("id", "")))
        if isinstance(record.get("learning_example"), Mapping):
            row = _example({**dict(record["learning_example"]), "id": identifier})
        elif "prompt" in record and isinstance(record.get("answer"), str):
            row = _example(record)
        elif "natural_language" in record and "short_path" in record and "long_path" in record:
            short, long = list(record["short_path"]), list(record["long_path"])
            if not short or len(long) <= len(short):
                raise ValueError("experience requires a distinct longer comparison proof")
            # Labels vary deterministically, rather than correlating A with shortest.
            short_first = int(hashlib.sha256(identifier.encode()).hexdigest(), 16) % 2 == 0
            a, b = (short, long) if short_first else (long, short)
            prompt = (record["natural_language"] + "\nChoose the shorter valid rule chain.\n"
                      + f"A: {' '.join(a)}\nB: {' '.join(b)}\nAnswer:")
            row = _example({"id": identifier, "prompt": prompt,
                            "answer": "A" if short_first else "B",
                            "family": record.get("family", "implication_graph"),
                            "split": record.get("split", "train")})
        elif record.get("lean_statement") and record.get("lean_proof"):
            # A theorem library may supply a semantically checked rendering at admission.
            rendered = record.get("learning_example")
            if not isinstance(rendered, Mapping):
                raise ValueError("non-graph lemma requires an admitted learning_example prompt/answer")
            row = _example({**dict(rendered), "id": identifier})
        else:
            raise ValueError("accepted library row has no supported language experience")
        row["provenance"].update(accepted_library_hash=library_hash,
                                 accepted_library_id=identifier,
                                 use="verified_language_training")
        result.append(row)
    if not result or len({r["id"] for r in result}) != len(result):
        raise ValueError("experience must be nonempty with unique IDs")
    return result


def useful_control(examples: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """True ordinary-language supervision, with matched byte length and answers.

    This factual retrieval control is deliberately simple. It controls supervised
    updates and compute, not every possible useful natural-language curriculum.
    The fixed-padding trainer additionally matches model token shapes, including
    pretrained tokenizers whose byte/token ratios differ.
    """
    result = []
    for original in examples:
        row = _example(original)
        if row["choices"] != ["A", "B"]:
            raise ValueError("matched ordinary-language control currently requires A/B choices")
        fact = "red" if row["answer"] == "A" else "tan"
        text = f"The recorded color is {fact}. What color was recorded?\nA: red\nB: tan\nAnswer:"
        length = len(row["prompt"].encode("utf-8"))
        if len(text.encode()) > length:
            raise ValueError("verified prompt is too short for the matched factual control")
        # Spaces before the answer marker retain its stable continuation boundary.
        text = text[:-7] + " " * (length - len(text.encode())) + "Answer:"
        row.update(id="ordinary-control:" + row["id"], prompt=text, family="ordinary_fact_retrieval")
        row.pop("translated_target", None)
        row["provenance"] = {"use": "ordinary_language_control", "matched_to": original["id"],
                             "correct_supervision": True, "fact": f"recorded color is {fact}"}
        result.append(row)
    return result


@dataclass(frozen=True)
class TrainingBudget:
    steps: int = 4
    batch_size: int = 2
    max_length: int = 1024
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    auxiliary_weight: float = 0.0

    def __post_init__(self):
        if (any(type(v) is not int for v in (self.steps, self.batch_size, self.max_length))
                or min(self.steps, self.batch_size, self.max_length) <= 0):
            raise ValueError("steps, batch size, and context must be positive")
        values = (self.learning_rate, self.weight_decay, self.gradient_clip, self.auxiliary_weight)
        if (not all(math.isfinite(v) for v in values) or self.learning_rate <= 0
                or self.weight_decay < 0 or self.gradient_clip <= 0 or self.auxiliary_weight < 0):
            raise ValueError("invalid optimizer budget")


class TinyRecipient:
    """Small persistent causal byte language model for a CPU pipeline smoke test."""
    backend = "tiny_byte_causal"
    scope = "randomly initialized small causal language model; not general-purpose capability evidence"

    def __init__(self, *, seed: int = 1729, hidden: int = 32, layers: int = 1,
                 heads: int = 2, max_length: int = 1024, device: str = "cpu"):
        import torch
        from torch import nn
        if hidden % heads or min(hidden, layers, heads, max_length) <= 0:
            raise ValueError("invalid tiny recipient architecture")
        self.config = dict(seed=seed, hidden=hidden, layers=layers, heads=heads,
                           max_length=max_length, device=device)
        torch.manual_seed(seed)
        class ByteCausalLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.tokens = nn.Embedding(259, hidden, padding_idx=0)
                self.positions = nn.Embedding(max_length, hidden)
                block = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout=0.0,
                          activation="gelu", batch_first=True, norm_first=True)
                self.blocks = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
                self.norm = nn.LayerNorm(hidden)
                self.head = nn.Linear(hidden, 259)
            def forward(self, ids, attention_mask):
                positions = torch.arange(ids.shape[1], device=ids.device)
                x = self.tokens(ids) + self.positions(positions)[None]
                causal = torch.triu(torch.ones((ids.shape[1], ids.shape[1]), dtype=torch.bool,
                                               device=ids.device), diagonal=1)
                return self.head(self.norm(self.blocks(x, mask=causal,
                                      src_key_padding_mask=~attention_mask.bool())))
        self.module = ByteCausalLM().to(device)
        self.device, self.pad_id, self.eos_id = device, 0, 2
        self.hidden_size = hidden
        self.representation_boundary = "final layer-normalized hidden state at last prompt token before output head"
        self.optimizer = None
        self.optimizer_config = None
        self.updates = 0
        self.frozen_evaluation_sha256 = None
        self.history: list[dict[str, Any]] = []

    def encode(self, prompt: str, answer: str) -> tuple[list[int], list[int]]:
        prefix = [1] + [b + 3 for b in prompt.encode("utf-8")]
        suffix = [b + 3 for b in (" " + answer).encode("utf-8")] + [self.eos_id]
        return prefix + suffix, [-100] * len(prefix) + suffix

    def trainable_parameters(self):
        return [p for p in self.module.parameters() if p.requires_grad]


class HFRecipient:
    """Local pretrained causal LM with a trainable final-hidden residual adapter.

    The adapter transforms the exact input to ``get_output_embeddings()``:
    ``h -> h + up(down(h))``. All pretrained parameters remain frozen. Its weights
    and AdamW state persist across iterations and are serialized separately from
    the cached base model. Model/tokenizer revisions must both be immutable SHAs.
    ``shared_base_from`` lets sequential treatment/control arms reuse one frozen
    checkpoint while keeping independent adapters and optimizers. Calls that share
    a base must run sequentially. No adapter hook remains between forward calls.
    This updates a general language model, but useful transfer still needs testing.
    """
    backend = "huggingface_residual_adapter"
    scope = "pretrained language model updated by a final-hidden residual adapter"

    def __init__(self, *, model_id: str, revision: str, tokenizer_revision: str | None = None,
                 rank: int = 8, seed: int = 1729, max_length: int = 1024,
                 device: str = "cpu", local_files_only: bool = True,
                 shared_base_from: HFRecipient | None = None):
        import torch
        from torch import nn
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer_revision = tokenizer_revision or revision
        if not all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in (revision, tokenizer_revision)):
            raise ValueError("pretrained model and tokenizer require immutable 40-character revisions")
        if rank <= 0 or max_length <= 0:
            raise ValueError("adapter rank and context must be positive")
        self.config = dict(model_id=model_id, revision=revision,
                           tokenizer_revision=tokenizer_revision, rank=rank, seed=seed,
                           max_length=max_length, device=device, local_files_only=local_files_only)
        if shared_base_from is not None:
            if not isinstance(shared_base_from, HFRecipient) or any(
                    shared_base_from.config[key] != self.config[key]
                    for key in ("model_id", "revision", "tokenizer_revision", "device")):
                raise ValueError("shared frozen base must have identical model/tokenizer revisions and device")
            self.tokenizer = shared_base_from.tokenizer
            base = shared_base_from.module.base
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=tokenizer_revision,
                                    local_files_only=local_files_only, trust_remote_code=False)
            base = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
                              local_files_only=local_files_only, trust_remote_code=False,
                              torch_dtype=torch.float32).to(device).eval()
        base.requires_grad_(False)
        hidden = base.get_output_embeddings().weight.shape[1]
        if max_length > getattr(base.config, "max_position_embeddings", max_length):
            raise ValueError("requested training context exceeds the pretrained model context")
        torch.manual_seed(seed)
        class AdaptedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = base
                self.down = nn.Linear(hidden, rank, bias=False).to(device)
                self.up = nn.Linear(rank, hidden, bias=False).to(device)
                nn.init.normal_(self.down.weight, std=0.02)
                nn.init.zeros_(self.up.weight)
            def patch(self, module, args):
                h = args[0]
                return (h + self.up(self.down(h)),) + args[1:]
            def forward(self, ids, attention_mask):
                # Each arm owns only its adapter. Shared frozen bases must never
                # retain hooks that apply another arm's adapter to this forward.
                handle = self.base.get_output_embeddings().register_forward_pre_hook(self.patch, prepend=True)
                try:
                    return self.base(input_ids=ids, attention_mask=attention_mask,
                                     use_cache=False).logits
                finally:
                    handle.remove()
            def train(self, mode=True):
                super().train(mode)
                self.base.eval()  # retain the pretrained model's deterministic dropout policy
                return self
        self.module = AdaptedModel().to(device)
        self.hidden_size = hidden
        self.representation_boundary = "post-adapter final hidden state at last prompt token before output head"
        self.device = device
        self.eos_id = self.tokenizer.eos_token_id
        if self.eos_id is None:
            raise ValueError("pretrained tokenizer needs an explicit EOS token")
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.eos_id
        self.optimizer = None
        self.optimizer_config = None
        self.updates = 0
        self.frozen_evaluation_sha256 = None
        self.history: list[dict[str, Any]] = []

    def encode(self, prompt: str, answer: str) -> tuple[list[int], list[int]]:
        prefix = self.tokenizer.encode(prompt, add_special_tokens=False)
        complete = self.tokenizer.encode(prompt + " " + answer, add_special_tokens=False)
        if not prefix or complete[:len(prefix)] != prefix or len(complete) == len(prefix):
            raise ValueError("answer retokenizes the prompt boundary; choose an explicit stable prompt separator")
        suffix = complete[len(prefix):] + [self.eos_id]
        return prefix + suffix, [-100] * len(prefix) + suffix

    def trainable_parameters(self):
        return [p for p in self.module.parameters() if p.requires_grad]


def _encoded(model, examples: Sequence[Mapping[str, Any]], max_length: int):
    if max_length > model.config["max_length"]:
        raise ValueError("training budget exceeds recipient context")
    result = [model.encode(r["prompt"], r["answer"]) for r in examples]
    if any(len(ids) > max_length for ids, _ in result):
        longest = max(len(ids) for ids, _ in result)
        raise ValueError(f"experience exceeds fixed context ({longest} > {max_length}); no truncation permitted")
    return result


def _batch(model, encoded, max_length):
    import torch
    ids = torch.full((len(encoded), max_length), model.pad_id, dtype=torch.long, device=model.device)
    targets = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for i, (tokens, labels) in enumerate(encoded):
        ids[i, :len(tokens)] = torch.tensor(tokens, device=model.device)
        targets[i, :len(tokens)] = torch.tensor(labels, device=model.device)
        mask[i, :len(tokens)] = 1
    return ids, targets, mask


def _parameter_digest(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.module.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parameter_sha256(model) -> str:
    """Hash trainable recipient tensors; pretrained base identity is pinned in config.

    Tiny models train all weights. HF recipients hash the adapter weights, while
    their frozen base and tokenizer are identified by model ID and immutable SHAs.
    """
    return _parameter_digest(model)


def train_update(model, examples: Iterable[Mapping[str, Any]], *, budget: TrainingBudget,
                 fixed_evaluation: FrozenEvaluation, seed: int = 1729) -> dict[str, Any]:
    """Perform real persistent optimizer updates without using P for selection.

    Each batch is padded to the registered context, so two arms with identical
    budgets execute identical batch/token shapes even for different curricula.
    Optimizer moments and weights continue across calls; changing optimizer
    hyperparameters mid-run is rejected. No automatic early stopping occurs.
    """
    import torch
    rows = [_example(r) for r in examples]
    if not rows:
        raise ValueError("empty training experience")
    _disjoint(rows, fixed_evaluation)
    if model.frozen_evaluation_sha256 not in (None, fixed_evaluation.sha256):
        raise ValueError("recipient cannot change its independently registered evaluation")
    model.frozen_evaluation_sha256 = fixed_evaluation.sha256
    encoded = _encoded(model, rows, budget.max_length)
    if budget.auxiliary_weight:
        import numpy as np
        try:
            targets = np.asarray([r.get("translated_target", []) for r in rows], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("translated targets must be numeric fixed-dimensional vectors") from exc
        if targets.shape != (len(rows), model.hidden_size) or not np.isfinite(targets).all():
            raise ValueError("positive auxiliary weight requires a finite translated target at the recipient hidden dimension for every row")
    config = dict(learning_rate=budget.learning_rate, weight_decay=budget.weight_decay,
                  gradient_clip=budget.gradient_clip)
    if model.optimizer is None:
        model.optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=budget.learning_rate,
                          weight_decay=budget.weight_decay, betas=(0.9, 0.95))
        model.optimizer_config = config
    elif model.optimizer_config != config:
        raise ValueError("persistent optimizer settings cannot change across iterations")
    initial = _parameter_digest(model)
    losses, language_losses, auxiliary_losses = [], [], []
    answer_tokens = actual_tokens = 0
    order = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(order)
    model.module.train()
    for step in range(budget.steps):
        indices = [order[(step * budget.batch_size + k) % len(order)] for k in range(budget.batch_size)]
        selected = [encoded[i] for i in indices]
        ids, labels, mask = _batch(model, selected, budget.max_length)
        model.optimizer.zero_grad(set_to_none=True)
        captured = []
        head = model.module.head if model.backend == TinyRecipient.backend else model.module.base.get_output_embeddings()
        handle = head.register_forward_pre_hook(lambda module, args: captured.append(args[0])) if budget.auxiliary_weight else None
        try:
            logits = model.module(ids, mask)
        finally:
            if handle is not None:
                handle.remove()
        shifted = labels[:, 1:]
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                                 shifted.reshape(-1), ignore_index=-100)
        language_loss = loss
        auxiliary_loss = torch.zeros((), device=model.device)
        if budget.auxiliary_weight:
            hidden = captured[0]
            prompt_positions = torch.tensor([sum(t == -100 for t in selected[k][1]) - 1
                                             for k in range(len(selected))], device=model.device)
            states = hidden[torch.arange(len(selected), device=model.device), prompt_positions]
            target = torch.tensor([rows[i]["translated_target"] for i in indices],
                                  dtype=states.dtype, device=model.device)
            if not bool(torch.isfinite(target).all()):
                raise ValueError("translated target contains non-finite values")
            auxiliary_loss = torch.nn.functional.mse_loss(states, target)
            loss = loss + budget.auxiliary_weight * auxiliary_loss
        if not bool(torch.isfinite(loss)):
            raise ValueError("non-finite recipient training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), budget.gradient_clip)
        model.optimizer.step()
        model.updates += 1
        losses.append(float(loss.detach().cpu()))
        language_losses.append(float(language_loss.detach().cpu()))
        auxiliary_losses.append(float(auxiliary_loss.detach().cpu()))
        actual_tokens += int(mask.sum().cpu())
        answer_tokens += int((shifted != -100).sum().cpu())
    final = _parameter_digest(model)
    if initial == final:
        raise ValueError("optimizer completed without changing recipient parameters")
    report = {"backend": model.backend, "scope": model.scope, "budget": asdict(budget),
              "updates_this_iteration": budget.steps, "cumulative_updates": model.updates,
              "processed_padded_tokens": budget.steps * budget.batch_size * budget.max_length,
              "nonpadding_tokens": actual_tokens, "supervised_answer_tokens": answer_tokens,
              "examples_seen": budget.steps * budget.batch_size,
              "examples_sha256": _digest(rows), "fixed_evaluation_sha256": fixed_evaluation.sha256,
              "parameter_sha256_before": initial, "parameter_sha256_after": final,
              "mean_training_token_nll": sum(language_losses) / len(language_losses),
              "mean_auxiliary_hidden_mse": sum(auxiliary_losses) / len(auxiliary_losses),
              "mean_total_training_loss": sum(losses) / len(losses), "step_losses": losses,
              "seed": seed}
    model.history.append(report)
    return report


def _measure(model, rows: Sequence[Mapping[str, Any]], max_length: int | None = None) -> dict[str, Any]:
    import torch
    max_length = max_length or model.config["max_length"]
    records = []
    model.module.eval()
    with torch.inference_mode():
        for row in rows:
            choices = [_example({**row, "answer": answer}) for answer in row["choices"]]
            encoded = _encoded(model, choices, max_length)
            # Evaluation uses the necessary sequence length, never a training budget.
            ids, labels, mask = _batch(model, encoded, max(len(x[0]) for x in encoded))
            logits = model.module(ids, mask)
            losses = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                        labels[:, 1:].reshape(-1), ignore_index=-100, reduction="none")
            losses = losses.reshape(len(choices), -1)
            sums = losses.sum(dim=1).float().cpu().tolist()
            counts = (labels[:, 1:] != -100).sum(dim=1).cpu().tolist()
            prediction = row["choices"][min(range(len(sums)), key=sums.__getitem__)]
            gold = row["choices"].index(row["answer"])
            records.append({"id": row["id"], "family": row["family"], "answer": row["answer"],
                            "prediction": prediction, "correct": prediction == row["answer"],
                            "answer_nll": sums[gold], "answer_token_nll": sums[gold] / counts[gold],
                            "answer_tokens": counts[gold],
                            "choice_nll": dict(zip(row["choices"], sums))})
    return {"count": len(records), "risk_mean_answer_nll": sum(r["answer_nll"] for r in records) / len(records),
            "accuracy": sum(r["correct"] for r in records) / len(records),
            "prediction_rule": "minimum full-answer sequence NLL over declared choices, including EOS",
            "records": records}


def evaluate(model, fixed_evaluation: FrozenEvaluation) -> dict[str, Any]:
    """Estimate R_P on a fixed file, without training or curriculum side effects."""
    fixed_evaluation.verify()
    if model.frozen_evaluation_sha256 not in (None, fixed_evaluation.sha256):
        raise ValueError("evaluation distribution differs from recipient registration")
    report = _measure(model, fixed_evaluation.examples)
    report.update(fixed_evaluation_sha256=fixed_evaluation.sha256,
                  recipient_parameter_sha256=_parameter_digest(model), cumulative_updates=model.updates)
    return report


def represent(model, prompts: Sequence[str]):
    """Extract recipient-space vectors at a declared, reproducible boundary.

    Returns a finite float32 NumPy array [prompts, hidden_size]. Each vector is
    taken after the final norm and adapter (if any), at the last prompt token,
    with no answer tokens or future information in its causal context.
    """
    import numpy as np
    import torch
    values = []
    model.module.eval()
    head = model.module.head if model.backend == TinyRecipient.backend else model.module.base.get_output_embeddings()
    with torch.inference_mode():
        for prompt in prompts:
            encoded, labels = model.encode(prompt, "A")
            length = sum(label == -100 for label in labels)
            if length > model.config["max_length"]:
                raise ValueError("representation prompt exceeds recipient context")
            captured = []
            handle = head.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
            try:
                ids = torch.tensor([encoded[:length]], device=model.device)
                model.module(ids, torch.ones_like(ids))
                value = captured[0][0, -1].float().cpu().numpy().copy()
            finally:
                handle.remove()
            values.append(value)
    result = np.stack(values) if values else np.empty((0, model.hidden_size), dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("non-finite recipient representation")
    return result


def score_candidates(model, candidates: Iterable[Mapping[str, Any]], *,
                     fixed_evaluation: FrozenEvaluation) -> list[dict[str, Any]]:
    """Rank verified curriculum candidates by current recipient surprise, never P.

    This implements a feedback heuristic, not a proof that high-loss problems are
    informative. The generator must independently verify and diversify candidates.
    """
    rows = [_example(row) for row in candidates]
    if not rows:
        return []
    _disjoint(rows, fixed_evaluation)
    measured = _measure(model, rows)["records"]
    return sorted(({"id": r["id"], "score": r["answer_nll"], "criterion": "training_candidate_answer_nll"}
                   for r in measured), key=lambda row: (-row["score"], row["id"]))


def save_recipient(model, path: Path | str) -> dict[str, Any]:
    """Serialize trainable weights, AdamW moments, and cumulative iteration state."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu() for name, value in model.module.state_dict().items()
             if model.backend != HFRecipient.backend or name.startswith(("down.", "up."))}
    value = {"schema_version": 1, "backend": model.backend, "config": model.config,
             "trainable_state_dict": state, "optimizer_state_dict": model.optimizer.state_dict() if model.optimizer else None,
             "optimizer_config": model.optimizer_config, "updates": model.updates,
             "fixed_evaluation_sha256": model.frozen_evaluation_sha256, "history": model.history,
             "parameter_sha256": _parameter_digest(model),
             "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    torch.save(value, path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "parameter_sha256": value["parameter_sha256"], "cumulative_updates": model.updates}


def load_recipient(path: Path | str, *, device: str | None = None):
    """Reload a checkpoint without resetting weights or optimizer moments."""
    import torch
    value = torch.load(Path(path), map_location="cpu", weights_only=True)
    if value.get("schema_version") != 1:
        raise ValueError("unsupported recipient checkpoint")
    config = dict(value["config"])
    if device is not None:
        config["device"] = device
    if value["backend"] == TinyRecipient.backend:
        model = TinyRecipient(**config)
        model.module.load_state_dict(value["trainable_state_dict"])
    elif value["backend"] == HFRecipient.backend:
        model = HFRecipient(**config)
        missing, unexpected = model.module.load_state_dict(value["trainable_state_dict"], strict=False)
        if unexpected or any(not name.startswith("base.") for name in missing):
            raise ValueError("pretrained adapter checkpoint has mismatched trainable keys")
    else:
        raise ValueError("unknown recipient backend")
    model.optimizer_config = value["optimizer_config"]
    if value["optimizer_state_dict"] is not None:
        opt = model.optimizer_config
        model.optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=opt["learning_rate"],
                                            weight_decay=opt["weight_decay"], betas=(0.9, 0.95))
        model.optimizer.load_state_dict(value["optimizer_state_dict"])
        for state in model.optimizer.state.values():
            for key, tensor in state.items():
                if isinstance(tensor, torch.Tensor) and key != "step":
                    state[key] = tensor.to(model.device)
    model.updates = value["updates"]
    model.frozen_evaluation_sha256 = value["fixed_evaluation_sha256"]
    model.history = value["history"]
    if _parameter_digest(model) != value["parameter_sha256"]:
        raise ValueError("recipient checkpoint parameter checksum mismatch")
    return model
