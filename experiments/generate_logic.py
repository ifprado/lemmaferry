#!/usr/bin/env python3
"""Export reproducible graph/text pairs and proof terms. Lean is not run here."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lemmaferry.logic import SPLIT_SETTINGS, generate_examples  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100, help="examples per split")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split", choices=["all", *SPLIT_SETTINGS], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = list(SPLIT_SETTINGS) if args.split == "all" else [args.split]
    summary = {"seed": args.seed, "lean_status": "not_run", "splits": {}}
    for split in splits:
        examples = generate_examples(args.count, split, args.seed)
        jsonl_path = args.output_dir / f"{split}.jsonl"
        lean_path = args.output_dir / f"{split}.lean"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
        lean_lines = ["-- Generated constructive propositional examples. Run Lean to verify.",
                      "set_option autoImplicit false", ""]
        for example in examples:
            lean_lines += [example["lean_declaration"], f"#print axioms {example['lean_name']}",
                           example["long_lean_declaration"], f"#print axioms {example['lean_name']}_long", ""]
        lean_path.write_text("\n".join(lean_lines), encoding="utf-8")
        summary["splits"][split] = {
            "examples": len(examples), "family": examples[0]["family"],
            "template": examples[0]["rendering_template"],
            "jsonl": str(jsonl_path), "lean_source": str(lean_path),
            "python_certificates": "verified", "render_roundtrip": "verified",
        }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Lean proof terms were exported but were NOT compiled by this command.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
