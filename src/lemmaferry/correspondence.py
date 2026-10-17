"""Bind formal and language views without assuming their token streams agree.

The hashes preserve the supplied semantic pairing and library provenance. They
cannot certify an English rendering: that requires an independent audit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def build_correspondence(
    identifier: str, family: str, formal_text: str, language_text: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind two views to an immutable verified-library snapshot supplied by caller.

    The library accepts/validates the theorem before calling this function. This
    function deliberately does not turn a metadata flag into a Lean check.
    """
    for name, value in (("id", identifier), ("family", family),
                        ("formal_text", formal_text), ("language_text", language_text)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"nonempty {name} required")
    if " ".join(formal_text.split()) == " ".join(language_text.split()):
        raise ValueError("correspondence requires distinct formal and language views")
    snapshot = provenance.get("library_snapshot_sha256")
    if not isinstance(snapshot, str) or not re.fullmatch(r"[0-9a-f]{64}", snapshot):
        raise ValueError("provenance requires library_snapshot_sha256")
    row = {
        "id": identifier, "family": family,
        "formal_text": formal_text, "language_text": language_text,
        "provenance": dict(provenance),
        "formal_text_sha256": _sha(formal_text.encode()),
        "language_text_sha256": _sha(language_text.encode()),
    }
    row["correspondence_sha256"] = _sha(_canonical(row))
    return row


def _validated(row: Mapping[str, Any]) -> dict[str, Any]:
    rebuilt = build_correspondence(row["id"], row["family"], row["formal_text"],
                                   row["language_text"], row["provenance"])
    if dict(row) != rebuilt:
        raise ValueError("correspondence hash or view content differs")
    return rebuilt


def write_correspondence(
    splits: Mapping[str, Iterable[Mapping[str, Any]]], output: str | Path,
) -> dict[str, Any]:
    """Write separate collector-ready views and their cross-domain pairing manifest."""
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("train, val and test correspondence splits required")
    rows_by_split = {split: [_validated(row) for row in splits[split]]
                     for split in ("train", "val", "test")}
    ids: set[str] = set()
    family_splits: dict[str, str] = {}
    text_splits: dict[tuple[str, str], str] = {}
    snapshots: set[str] = set()
    for split, rows in rows_by_split.items():
        if not rows or (split == "train" and len(rows) < 2):
            raise ValueError("nonempty splits and at least two train pairs required")
        for row in rows:
            if row["id"] in ids:
                raise ValueError("correspondence IDs repeat across rows or splits")
            ids.add(row["id"])
            family = row["family"]
            if family in family_splits and family_splits[family] != split:
                raise ValueError("correspondence structural families cross splits")
            family_splits[family] = split
            snapshots.add(row["provenance"]["library_snapshot_sha256"])
            for view in ("formal", "language"):
                key = (view, " ".join(row[f"{view}_text"].split()))
                if key in text_splits and text_splits[key] != split:
                    raise ValueError("correspondence text repeats across splits")
                text_splits[key] = split
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for domain in ("math", "language"):
        (output / domain).mkdir()
    manifest: dict[str, Any] = {
        "schema": "lemmaferry.correspondence.v1", "status": "prepared",
        "library_snapshots_sha256": sorted(snapshots), "splits": {},
        "scope": "Caller-supplied corresponding views of accepted theorems; language meaning requires a separate audit.",
        "input_policy": "Distinct formal and natural-language views; tokenizer hashes and token lengths may differ.",
    }
    for split, rows in rows_by_split.items():
        info: dict[str, Any] = {"count": len(rows), "pairs": rows}
        for domain, field in (("math", "formal_text"), ("language", "language_text")):
            path = output / domain / f"{split}.jsonl"
            values = [{"id": row["id"], "family": row["family"], "text": row[field],
                       "correspondence_sha256": row["correspondence_sha256"]} for row in rows]
            path.write_text("".join(json.dumps(v, sort_keys=True) + "\n" for v in values), encoding="utf-8")
            info[domain] = {"path": str(path.relative_to(output)), "sha256": _sha(path.read_bytes())}
        manifest["splits"][split] = info
    manifest["manifest_content_sha256"] = _sha(_canonical(manifest))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_correspondence(path: str | Path) -> dict[str, Any]:
    """Check manifest content, collector-input files, and every supplied pairing."""
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        content = {key: value for key, value in manifest.items() if key != "manifest_content_sha256"}
        if manifest.get("manifest_content_sha256") != _sha(_canonical(content)):
            raise ValueError("correspondence manifest checksum mismatch")
        if manifest.get("schema") != "lemmaferry.correspondence.v1" or manifest.get("status") != "prepared":
            raise ValueError("unsupported correspondence manifest")
        if set(manifest["splits"]) != {"train", "val", "test"}:
            raise ValueError("correspondence splits missing")
        ids: set[str] = set()
        families: dict[str, str] = {}
        text_splits: dict[tuple[str, str], str] = {}
        snapshots = set()
        for split in ("train", "val", "test"):
            info = manifest["splits"][split]
            rows = [_validated(row) for row in info["pairs"]]
            if len(rows) != info["count"] or not rows or (split == "train" and len(rows) < 2):
                raise ValueError("correspondence row counts differ")
            for row in rows:
                if row["id"] in ids:
                    raise ValueError("correspondence IDs repeat")
                ids.add(row["id"])
                if row["family"] in families and families[row["family"]] != split:
                    raise ValueError("correspondence families cross splits")
                families[row["family"]] = split
                snapshots.add(row["provenance"]["library_snapshot_sha256"])
                for domain, field in (("math", "formal_text"), ("language", "language_text")):
                    text_key = (domain, " ".join(row[field].split()))
                    if text_key in text_splits and text_splits[text_key] != split:
                        raise ValueError("correspondence text repeats across splits")
                    text_splits[text_key] = split
            for domain, field in (("math", "formal_text"), ("language", "language_text")):
                expected_path = f"{domain}/{split}.jsonl"
                if info[domain]["path"] != expected_path:
                    raise ValueError("unexpected correspondence input path")
                input_path = path.parent / expected_path
                if _sha(input_path.read_bytes()) != info[domain]["sha256"]:
                    raise ValueError("correspondence input checksum mismatch")
                actual = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                expected = [{"id": row["id"], "family": row["family"], "text": row[field],
                             "correspondence_sha256": row["correspondence_sha256"]} for row in rows]
                if actual != expected:
                    raise ValueError("correspondence input IDs, views, or pair hashes differ")
        if sorted(snapshots) != manifest["library_snapshots_sha256"]:
            raise ValueError("correspondence library snapshots differ")
        return manifest
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid correspondence manifest: {path}") from exc


def prepare_correspondence_records(
    records: Iterable[Mapping[str, Any]], *, library_snapshot_sha256: str,
) -> list[dict[str, Any]]:
    """Adapter for the original checked implication records; no new Lean claims."""
    from .logic import Edge, check_example, path_atoms

    result = []
    for record in records:
        check_example(record)
        status = record["verification"]
        if status.get("lean") != "verified" or status.get("lean_axioms") != "none":
            raise ValueError("correspondence needs previously Lean-verified records")
        edges = [Edge(**edge) for edge in record["edges"]]
        atoms = path_atoms(edges, record["source_atom"], record["short_path"])
        lines = [record["natural_language"], "A derivation:",
                 f"Start with {record['atoms'][atoms[0]]['label']}."]
        for edge_id, atom in zip(record["short_path"], atoms[1:]):
            lines.append(f"Apply rule {edge_id} to obtain {record['atoms'][atom]['label']}.")
        result.append(build_correspondence(
            record["id"], record["family"], record["lean_declaration"], "\n".join(lines),
            {"library_snapshot_sha256": library_snapshot_sha256,
             "record_sha256": _sha(_canonical(record)),
             "verification": dict(status), "adapter": "checked_implication_v1"},
        ))
    return result
