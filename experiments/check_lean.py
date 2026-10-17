#!/usr/bin/env python3
"""Run Lean locally, audit axiom output, and record per-example verification.

The default is fail closed: unavailable Lean returns exit code 2 and records
not_run. A Python graph certificate is never labeled Lean verified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lemmaferry.logic import check_example  # noqa: E402

FORBIDDEN = re.compile(r"\b(?:sorry|admit|axiom|unsafe)\b")


def source_for(examples: list[dict[str, Any]]) -> str:
    lines = ["set_option autoImplicit false", ""]
    for example in examples:
        for suffix, field in [("", "lean_declaration"), ("_long", "long_lean_declaration")]:
            declaration = example[field]
            if FORBIDDEN.search(declaration):
                raise ValueError("forbidden proof placeholder, declaration, or unsafe code")
            lines += [declaration, f"#print axioms {example['lean_name']}{suffix}", ""]
    return "\n".join(lines)


def audit_no_axioms(output: str, name: str) -> bool:
    # Lean 4 #print axioms uses this exact success sentence. Unrecognized output
    # is treated as an error rather than interpreted as a successful audit.
    pattern = rf"(?m)^['\"]?{re.escape(name)}['\"]? does not depend on any axioms\s*$"
    return re.search(pattern, output) is not None


def run_batch(examples: list[dict[str, Any]], lean: str, timeout: float,
              workdir: Path, stem: str) -> tuple[bool, str, str]:
    source_path = workdir / f"{stem}.lean"
    source_path.write_text(source_for(examples), encoding="utf-8")
    try:
        process = subprocess.run([lean, str(source_path)], capture_output=True, text=True,
                                 timeout=timeout, check=False)
        output = process.stdout + process.stderr
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return False, stdout + stderr, "timeout"
    except OSError as exc:
        return False, str(exc), "execution_error"
    if process.returncode != 0:
        return False, output, f"lean_exit_{process.returncode}"
    for example in examples:
        for name in (example["lean_name"], example["lean_name"] + "_long"):
            if not audit_no_axioms(output, name):
                return False, output, "axiom_audit_failed_or_unrecognized"
    return True, output, "verified"


def check_records(records: list[dict[str, Any]], lean: str | None, timeout: float = 30,
                  batch_size: int = 64, log_dir: Path | None = None) -> dict[str, int]:
    counts = {"verified": 0, "not_run": 0, "error": 0}
    good = []
    for record in records:
        verification = record.setdefault("verification", {})
        # Never carry an old compiler result or log reference into a fresh run.
        # Python certificate results are independently recomputed below.
        for field in ("lean_version", "lean_executable", "lean_log", "reason"):
            verification.pop(field, None)
        verification.update(lean="not_run", lean_axioms="not_run",
                            short_proof="not_run", long_proof="not_run")
        try:
            check_example(record)
            source_for([record])
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            verification.update(lean="error", lean_axioms="not_run", reason=f"invalid_certificate: {exc}")
            counts["error"] += 1
            continue
        verification.update(python_certificate="verified", render_roundtrip="verified")
        if lean is None:
            verification.update(reason="lean_executable_not_found")
            counts["not_run"] += 1
        else:
            good.append(record)
    if not good:
        return counts
    try:
        version_process = subprocess.run([lean, "--version"], capture_output=True,
                                         text=True, timeout=min(timeout, 10), check=False)
        lean_version = (version_process.stdout + version_process.stderr).strip()
    except (OSError, subprocess.TimeoutExpired):
        lean_version = "unavailable"
    for record in good:
        record["verification"]["lean_version"] = lean_version
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lemmaferry_lean_") as temporary:
        workdir = Path(temporary)
        for offset in range(0, len(good), batch_size):
            batch = good[offset:offset + batch_size]
            stem = f"batch_{offset // batch_size:05d}"
            passed, output, reason = run_batch(batch, lean, timeout, workdir, stem)
            if log_dir is not None:
                (log_dir / f"{stem}.log").write_text(output, encoding="utf-8")
            # Isolate failed batches so one invalid or slow theorem cannot
            # incorrectly invalidate otherwise verified individual examples.
            results = [(record, passed, output, reason, stem) for record in batch]
            if not passed and len(batch) > 1:
                results = []
                for index, record in enumerate(batch):
                    single_stem = f"{stem}_{index:03d}"
                    single_ok, single_output, single_reason = run_batch([record], lean, timeout, workdir, single_stem)
                    if log_dir is not None:
                        (log_dir / f"{single_stem}.log").write_text(single_output, encoding="utf-8")
                    results.append((record, single_ok, single_output, single_reason, single_stem))
            for record, ok, _output, result_reason, result_stem in results:
                status = "verified" if ok else "error"
                verification = record["verification"]
                verification.update(lean=status, lean_axioms="none" if ok else "not_verified",
                                    reason=result_reason, lean_executable=lean,
                                    short_proof=status, long_proof=status)
                if log_dir is not None:
                    verification["lean_log"] = str(log_dir / f"{result_stem}.log")
                counts[status] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="generated JSONL records")
    parser.add_argument("--output", type=Path, required=True, help="JSONL records with fresh verification status")
    parser.add_argument("--lean", default="lean", help="Lean executable, invoked without a shell")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--log-dir", type=Path, default=Path("results/lean_logs"))
    args = parser.parse_args()
    if args.timeout <= 0 or args.batch_size <= 0:
        parser.error("timeout and batch size must be positive")
    if args.input.resolve() == args.output.resolve():
        parser.error("use a separate output path to preserve the original records")
    with args.input.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        parser.error("input must contain at least one record")
    ids = [record.get("id") for record in records]
    if len(set(ids)) != len(ids):
        parser.error("record IDs must be unique")
    lean = shutil.which(args.lean)
    counts = check_records(records, lean, args.timeout, args.batch_size, args.log_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"counts": counts, "output": str(args.output)}, indent=2))
    if counts["error"]:
        return 1
    if lean is None or counts["not_run"]:
        print("Lean was not run. No records are Lean verified.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
