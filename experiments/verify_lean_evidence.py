#!/usr/bin/env python3
"""Require complete checked records and retained audits for a Lean reproduction.

This validates evidence produced by check_lean.py; it does not run Lean itself.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_lean import audit_no_axioms, check_example  # noqa: E402


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def verify(input_dir: Path, results_dir: Path, check_workdir: Path,
           toolchain: Path, count: int) -> dict:
    version_path = results_dir / "lean-version.txt"
    version = version_path.read_text(encoding="utf-8").strip()
    pin = toolchain.read_text(encoding="utf-8").strip()
    match = re.fullmatch(r"leanprover/lean4:v([0-9]+\.[0-9]+\.[0-9]+)", pin)
    if not match or not re.search(rf"\bversion {re.escape(match[1])}(?:,|\))", version):
        raise ValueError("actual Lean version does not match the pinned toolchain")
    paths = {toolchain, version_path}
    splits = {}
    for split in ("train", "validation", "test"):
        source_path = input_dir / f"{split}.jsonl"
        checked_path = results_dir / f"{split}.checked.jsonl"
        source, checked = read_records(source_path), read_records(checked_path)
        if len(source) != count or len(checked) != count:
            raise ValueError(f"{split}: expected {count} source and checked records")
        ids = [record["id"] for record in source]
        if len(set(ids)) != count or ids != [record["id"] for record in checked]:
            raise ValueError(f"{split}: duplicate or mismatched checked IDs")
        audits = 0
        paths.update((source_path, checked_path))
        for original, record in zip(source, checked):
            check_example(record)
            if original.get("split") != split or record.get("split") != split:
                raise ValueError(f"{split}: wrong split metadata")
            without_status = lambda row: {k: v for k, v in row.items() if k != "verification"}
            if without_status(original) != without_status(record):
                raise ValueError(f"{record['id']}: checked content differs from generated input")
            status = record["verification"]
            if any(status.get(field) != "verified" for field in
                   ("python_certificate", "render_roundtrip", "lean", "short_proof", "long_proof")):
                raise ValueError(f"{record['id']}: a required verification is missing")
            if (status.get("lean_axioms") != "none" or status.get("reason") != "verified"
                    or status.get("lean_version") != version):
                raise ValueError(f"{record['id']}: missing axiom or compiler-version evidence")
            log = (check_workdir / status["lean_log"]).resolve()
            if not log.is_relative_to(results_dir.resolve()):
                raise ValueError(f"{record['id']}: compiler log is outside this run")
            output = log.read_text(encoding="utf-8")
            for name in (record["lean_name"], record["lean_name"] + "_long"):
                if not audit_no_axioms(output, name):
                    raise ValueError(f"{name}: retained log lacks the zero-axiom audit")
                audits += 1
            paths.add(log)
        splits[split] = {"examples": count, "proofs": count * 2,
                         "recognized_zero_axiom_audits": audits}
    source_paths = [Path(__file__), Path(__file__).with_name("check_lean.py"),
                    Path(__file__).with_name("generate_logic.py"),
                    Path(__file__).resolve().parents[1] / "src/lemmaferry/logic.py",
                    Path(__file__).resolve().parents[1] / "lean/lake-manifest.json"]
    paths.update(source_paths)
    return {"status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
            "github_sha": os.environ.get("GITHUB_SHA"), "toolchain": pin,
            "lean_version": version, "splits": splits,
            "total_examples": count * 3, "total_checked_proofs": count * 6,
            "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted(paths, key=str)}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/ci-lean"))
    parser.add_argument("--check-workdir", type=Path, default=Path("lean"))
    parser.add_argument("--toolchain", type=Path, default=Path("lean/lean-toolchain"))
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("count must be positive")
    output = args.results_dir / "evidence.json"
    output.unlink(missing_ok=True)  # Never leave stale passed evidence after a rejected rerun.
    try:
        summary = verify(args.input_dir, args.results_dir, args.check_workdir,
                         args.toolchain, args.count)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        print(f"Lean evidence rejected: {exc}", file=sys.stderr)
        return 1
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Verified evidence for {summary['total_examples']} examples / "
          f"{summary['total_checked_proofs']} proofs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
