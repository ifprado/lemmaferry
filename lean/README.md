# Lean verification

The generated examples use only propositions, assumptions, implication, and
function application. They introduce no custom axioms, `sorry`, or `admit`, and
require no Mathlib dependency. Each theorem is constructive in this small domain.
Generating source is not the same as compiling it.

The pinned toolchain is `leanprover/lean4:v4.35.0`. Install Lean through the
[official Lean installation instructions](https://lean-lang.org/install/).
Run the following from the repository root, after installing Python 3.10 or newer:

```bash
python experiments/generate_logic.py --count 20 --seed 7
cd lean
lean --version
python ../experiments/check_lean.py ../data/generated/train.jsonl --output ../results/train.checked.jsonl --log-dir ../results/lean_logs/train
python ../experiments/check_lean.py ../data/generated/validation.jsonl --output ../results/validation.checked.jsonl --log-dir ../results/lean_logs/validation
python ../experiments/check_lean.py ../data/generated/test.jsonl --output ../results/test.checked.jsonl --log-dir ../results/lean_logs/test
cd ..
```

The directory containing `lean-toolchain` selects the pinned Lean when using
`elan`. The checker initially runs batches and falls back to individual examples
after a failed batch. It checks both the short and longer proof, imposes a process
timeout, rejects placeholders, and requires each `#print axioms` audit to report
that the theorem depends on no axioms. Missing Lean yields exit code 2 and
`lean: "not_run"`; invalid certificates, compilation failures, or unrecognized
audit results yield an error. A Python certificate never earns a Lean status.

`results/*.checked.jsonl` preserves per-example verification results. Keep audit
logs with reported experiments. The short-path claim concerns the number of graph
edge applications. It does not claim shortest Lean source, shortest possible proof
in another logic, or minimized discovery compute.

The controlled language renderer has an exact parser round trip to the same graph.
This validates the template implementation. It does not establish the reliability
of a learned translator or the truth of any real-world causal claim.
