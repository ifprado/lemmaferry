# LF-002: first real Lean verification

All 60 generated examples passed real Lean 4.35.0 checking: 20 train, 20
validation, and 20 test examples, generated with seed 7. Both proof variants
were checked for every example, and all 120 `#print axioms` audits reported
that the theorem depends on no axioms. The corruption and missing-executable
controls behaved as required. No generator or checker changes were needed.

The run took place on 2026-10-16 in America/Los_Angeles (2026-10-17 UTC), on
macOS arm64 using Python 3.14.7. The actual compiler reported:

```text
Lean (version 4.35.0, arm64-apple-darwin24.6.0, commit eb6cafb7290e8bd5f2937522d8d804f58f02bafc, Release)
```

The delegated cousin agent performed the run and inspected the example pair.
Ian's separate review of that pair remains to be done.

## Method and results

The hypothesis was that the generated short and longer proofs compile against
the pinned toolchain and have recognized zero-axiom audits. No training,
parameter selection, model inference, or external dataset was involved. The
ground truth was successful compiler execution plus a separate audit for each
named declaration; Python graph and renderer checks were also rerun.

| Split | Examples verified | Proofs checked | Recognized zero-axiom audits | Short / longer edge applications |
| --- | ---: | ---: | ---: | --- |
| Train | 20 / 20 | 40 | 40 | 2 / 4 |
| Validation | 20 / 20 | 40 | 40 | 3 / 5 |
| Test | 20 / 20 | 40 | 40 | 4 / 7 |

All three checker commands returned exit code 0. Lean emitted unused-variable
warnings because the shared theorem assumptions include edges unused by one
of the alternative routes. The unedited compiler logs retain those warnings.

The 12 existing graph, renderer, and Lean-status unit tests passed. In
particular, the existing tests reject exit-zero compiler output without
recognized axiom audits and clear stale success metadata when Lean is absent.
Those subprocess mocks were unit tests; the 120 audits above came from the
installed real compiler.

## Reproduction

Lean was installed using the [official manual installation instructions](https://lean-lang.org/install/manual/).
The installer was downloaded and retained before execution; `--no-modify-path`
kept shell profiles unchanged. From the repository root:

```bash
mkdir -p results/lf-002/install
curl --fail --location --silent --show-error https://elan.lean-lang.org/elan-init.sh --output results/lf-002/install/elan-init.sh
sh results/lf-002/install/elan-init.sh -y --no-modify-path --default-toolchain none
"$HOME/.elan/bin/elan" toolchain install leanprover/lean4:v4.35.0
.venv/bin/python experiments/generate_logic.py --count 20 --seed 7 --output-dir results/lf-002/generated
.venv/bin/python -m unittest discover -s tests -p test_logic.py -v
cd lean
"$HOME/.elan/bin/lean" --version
../.venv/bin/python ../experiments/check_lean.py ../results/lf-002/generated/train.jsonl --lean "$HOME/.elan/bin/lean" --output ../results/lf-002/checked/train.checked.jsonl --log-dir ../results/lf-002/logs/train
../.venv/bin/python ../experiments/check_lean.py ../results/lf-002/generated/validation.jsonl --lean "$HOME/.elan/bin/lean" --output ../results/lf-002/checked/validation.checked.jsonl --log-dir ../results/lf-002/logs/validation
../.venv/bin/python ../experiments/check_lean.py ../results/lf-002/generated/test.jsonl --lean "$HOME/.elan/bin/lean" --output ../results/lf-002/checked/test.checked.jsonl --log-dir ../results/lf-002/logs/test
cd ..
```

The existing shared `.venv` contained the project's editable installation and
its plots extra. The artifact directory intentionally differs from the default
`data/generated` and `results/lean_logs` paths so the inputs, statuses, and logs
stay together and are not excluded by the default ignore rules.

## Inspected pair and failure controls

The pair [`train_000000`](../results/lf-002/controls/inspected-pair.lean) has
the same statement after normalizing the two declaration names. It assumes
`h_start : p4` and the same seven implication rules, and concludes `p0`.

- Short route: `p4 → p5 → p0`, using edges `e2, e5` and proof
  `h_e5 (h_e2 (h_start))`.
- Longer route: `p4 → p6 → p7 → p3 → p0`, using edges `e0, e4, e6, e3`
  and proof `h_e3 (h_e6 (h_e4 (h_e0 (h_start))))`.

The unused edge `e1 : p2 → p1` is a distractor. These are proofs of the same
conditional proposition. Their explicit premises are theorem arguments, not
global axioms. A separate real compiler invocation of this pair exited 0 and
produced both zero-axiom audits.

| Control | Observed result | Retained log |
| --- | --- | --- |
| Replace the short proof body in a copied `.lean` file with `h_start` | Compiler exit 1: `p4` does not match `p0`; the failed declaration audit includes `sorryAx` | [corrupted-source.log](../results/lf-002/controls/corrupted-source.log) |
| Make the same change to a copied JSONL record | Checker exit 1, `lean: "error"`, reason `invalid_certificate: Lean source does not match graph certificate`; rejected before compiler invocation | [corrupted-record.log](../results/lf-002/controls/corrupted-record.log) |
| Recheck all 20 previously verified train records with an intentionally absent Lean executable | Checker exit 2; all 20 become `not_run`; old compiler version, executable, and log references are cleared | [missing-lean.log](../results/lf-002/controls/missing-lean.log) |

The exact control argument lists, working directory, and expected/actual exit
codes are retained in [controls/results.json](../results/lf-002/controls/results.json).
The corrupted copies are intentionally invalid and are kept apart from the
60 accepted records.

## Artifacts and scope

The machine-readable [summary](../results/lf-002/summary.json) includes tool
versions, source SHA-256 hashes, counts, and control outcomes. No Git repository
existed at run time, so source hashes identify the code used. The artifact
[SHA256SUMS](../results/lf-002/SHA256SUMS) covers inputs, outputs, and logs.

- Generated JSONL and exported Lean sources: [`results/lf-002/generated/`](../results/lf-002/generated/).
- Fresh checked JSONL: [`results/lf-002/checked/`](../results/lf-002/checked/).
- Compiler audits: [train](../results/lf-002/logs/train/batch_00000.log),
  [validation](../results/lf-002/logs/validation/batch_00000.log), and
  [test](../results/lf-002/logs/test/batch_00000.log).
- Installation logs: [`results/lf-002/install/`](../results/lf-002/install/).
- Logic unit-test output: [logic-tests.log](../results/lf-002/logic-tests.log).

This establishes real Lean verification for these 60 constructive implication
examples and the two supplied proofs of each. It does not establish novel
mathematics, learned translation, model capability transfer, or shortest proofs
outside the generated graph's edge-application metric. The generator's original
manifest correctly retains `lean_status: "not_run"`; only the separately checked
records carry the compiler result.
