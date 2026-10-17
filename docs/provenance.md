# Provenance and release policy

Original code and documentation in this starter use Apache-2.0. This does not change the licenses of Python dependencies, Lean, models, source corpora, or borrowed code. No external model weights, private messages, scans, or paper PDFs are included.

## Required record for every new dataset

Store the generator version/commit and seed, source family and lineage, formal statement, assumptions and dependency identifiers, proof, semantic view, answer certificate, Lean toolchain/import revisions, checker status and axiom audit, and any human review status. Preserve failed attempts separately from accepted data.

Use distinct fields for `python_certificate`, `lean`, `lean_axioms`, and rendering/semantic checks. Do not rename `not_run` to `verified` to make a dataset convenient. A generated formal proof and a controlled Python certificate are valuable, but they are different from having executed Lean.

## Required record for every model experiment

Record the public model ID, immutable revision, tokenizer hash, model/data license link, layer, token position or pooling, normalization fitted on training only, random seeds, compute budget, training steps/tokens, split hashes, prompt/decoding specification, and all baseline outputs. Record actual hardware and peak memory rather than relying on estimates.

## Import checklist

- [ ] Find the actual upstream LICENSE file at the pinned revision.
- [ ] Record a source URL and commit, plus any required attribution.
- [ ] Check code, model weights, and dataset terms separately.
- [ ] Keep upstream notices when required. Do not present third-party material as original.
- [ ] Keep private conversations, API keys, employer information, and personal uploads out of the repository.
- [ ] Store large files outside Git and link the exact version and checksum.

## Current contents

| Item | Origin | Treatment |
| --- | --- | --- |
| Starter Python and docs | Created for this project | Apache-2.0 |
| Numeric toy outputs | This starter's seeded generator | Released with the starter; artificial data |
| Propositional graph tasks | This starter's seeded generator | Released with explicit verification statuses |
| NumPy and Matplotlib | Installed dependencies | Not vendored; retain their own terms |
| Lean | Separate installation | Not bundled |
| Retained elan installer | Downloaded from `https://elan.lean-lang.org/elan-init.sh`; exact SHA-256 retained in `results/lf-002/SHA256SUMS` | Unmodified installation evidence at `results/lf-002/install/elan-init.sh`; original Rust Project copyright and dual-license notice retained; distributed under its Apache-2.0 option, with attribution in `NOTICE` |
| Qwen checkpoints | Optional future downloads | Not bundled; exact licenses linked in the experiment plan |
| Research papers | Primary-source links | Not redistributed as full-text copies |

Before a public release, update the citation metadata, make the results match a specific commit, and have another contributor reproduce the main commands.
