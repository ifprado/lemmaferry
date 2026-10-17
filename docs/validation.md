# Preparation checks and measured results

These records describe what was actually executed while preparing the starter on October 16, 2026, Pacific time. They are not results from a language model.

## Follow-up runs

The [LF-001 reproduction record](../artifacts/lf-001/reproduction.json) contains fresh seed-7 and seed-19 JSON results, plots, environment versions, commands, source hashes, and the 23-test passing log. Every result field agrees with the bundled reference at relative tolerance `1e-8` and absolute tolerance `1e-10`; the largest absolute numeric difference is below `2.2e-13`. Both predefined seeds were retained, and preprocessing and recipient decoding used training data only. This run used Python 3.14.7, NumPy 2.5.4, and Matplotlib 3.11.2 on macOS arm64.

The separate [LF-002 Lean-verification record](lf-002-lean-verification.md) supersedes the preparation-time missing-Lean limitation below: real Lean 4.35.0 verified all 60 generated examples and both proof variants, with 120 recognized zero-axiom audits. Corrupted inputs failed, missing Lean remained `not_run`, and all 12 logic/checker tests passed. Inputs, checked records, and unedited logs are retained under [`results/lf-002/`](../results/lf-002/). No generator or checker fixes were needed.

## Executed successfully

- Editable package build and installation with existing dependencies.
- **23 Python unit and integration tests passed** on Python 3.12.14, NumPy 2.3.5, and Matplotlib 3.10.8.
- Python compilation of source, experiments, and tests.
- Artificial alignment experiments with seeds 7 and 19.
- Deterministic generation of three examples per formal-data split, nine examples total, with Python graph certificates and controlled renderer round trips checked.
- Missing-Lean behavior: checker returned exit code 2, zero Lean-verified records, three `not_run` records, and no fabricated compiler success.
- Optional activation collector and fitter help commands and static Python checks; numeric fitter integration tests use fake collected arrays.
- Interactive report build, JavaScript syntax check, internal-anchor checks, and HTML structure checks. Browser rendering was unavailable, so no screenshot-based visual review is claimed.

Tests cover held-out affine recovery, wrong anchors, training-only preprocessing, rigid alignment, semantic ambiguity, graph/proof certificates, renderer mismatches, split leakage, corrupted archives, missing Lean, stale compiler metadata, and exit-zero output without an axiom audit.

## Artificial alignment results

The toy uses 256 training and 256 test points, 12 source dimensions, 8 target dimensions, 4 planted shared factors, and Gaussian noise with standard deviation 0.01. Normalized RMSE uses per-coordinate target scales fitted on training data only.

| Seed | Paired affine nRMSE | Shuffled-anchor nRMSE | Training-mean nRMSE | Paired task accuracy | Shuffled-anchor task accuracy |
| --- | --- | --- | --- | --- | --- |
| 7 | 0.009702 | 1.068365 | 1.060576 | 100% | 68.36% |
| 19 | 0.006754 | 1.068165 | 1.002256 | 100% | 31.25% |

The task is the sign of a planted linear combination of latent factors. The target-space decoder is fitted on training data. The paired 100% result is unsurprising in this deliberately favorable construction. The changing shuffled accuracy illustrates why multiple seeds and the primary regression metric should be retained.

The symmetric-cloud counterexample has effectively zero unpaired geometric error and exact round trips under both identity and half-turn maps. Planted semantic-label accuracy is 100% for identity and 0% for the half-turn. These maps are hand-specified; this is an exact ambiguity example, not a learned-model benchmark.

Reproduce from the repository root:

```bash
python -m unittest discover -s tests -v
python experiments/toy_alignment.py --seed 7 --output artifacts/toy_alignment_seed7.json
python experiments/toy_alignment.py --seed 19 --output artifacts/toy_alignment_seed19.json
python experiments/generate_logic.py --count 3 --seed 7 --output-dir examples/logic
```

## Not executed during preparation

| Item | Actual status | Next action |
| --- | --- | --- |
| Real Lean compilation | Lean not installed in preparation environment | Run LF-002 and retain axiom audits |
| Actual model downloads/inference | No weights downloaded; torch/transformers unavailable | Run the small collector pilot from the model plan |
| Real-model alignment | Numeric integration tests only | Collect source/recipient activations, then fit |
| Activation intervention | Not implemented | Implement identity/oracle intervention before translation |
| Student training or distillation | Not run | Establish the dataset and matched baseline first |
| Adaptive self-play | Not implemented | Compare only after a useful fixed baseline exists |
| GitHub-hosted CI | Workflow supplied, not run on a remote repository | Inspect the first run after upload |
| GitHub repository/domain purchase | No remote created and no domain purchased | User launch checklist |

The source pair suggested for the next milestone is a small mathematical/general model pair, not a certified Lean prover. Actual proof checking always remains separate.
