# Constructive learning loop

This branch implements an executable version of the proposed experimental loop:

```text
verified library C_t
    -> library-conditioned proposals (statement, proof)
    -> constructive-policy checks and real Lean verification
    -> accepted library C_(t+1)
    -> paired formal and language representations
    -> learned T and S, with held-out geometric diagnostics
    -> verified language targets or translated-state training
    -> updated recipient L_(t+1)
    -> training-curriculum feedback for the next proposals
```

The strongest conjecture is that these cycles could sustain increasingly broad
reasoning improvements. Implementing the mechanism does not establish that
conjecture. The existing negative experiments remain part of the record.

## Foundation, admission, and library growth

`Foundation` specifies `constructive_prop_v1`, Lean 4.35.0, and the exact Lean
compiler commit. Its restricted formula/proof language permits proposition
parameters, implication application, conjunction introduction/elimination, and
calls to previously accepted lemmas. It does not accept arbitrary Lean source,
new axioms, classical proof operators, placeholders, or caller-supplied approval.
Permitted premises are explicit theorem arguments.

`VerifiedLibrary.accept` reconstructs the theorem from that language, checks the
declared dependency IDs, compiles the cumulative library with real Lean, and
requires recognized zero-axiom audits for every theorem and generated definition.
Missing Lean cannot admit a proposal. The compiler, launcher, source, logs,
foundation, record chain, and immutable library snapshots have recorded hashes.
Reopening a library performs a fresh Lean replay without changing its files.
Admission uses an exclusive writer lock and an atomic index replacement.

The generator produces parameterized implication chains and conjunction
construction/projection tasks. Later proposals actually invoke earlier accepted
theorems. Structural hashes discard cosmetic names and proof spelling; changing
arity, route depth, or projection position can add a new supported structure.
This is a domain-specific structural filter, not a general test of mathematical
novelty or learning value. Generation has explicit proposal, proof-node, and
compiler-time budgets; indefinitely useful experience is a hypothesis, not an
operational guarantee on finite hardware.

## Representations and the two maps

The mathematical view contains the formal statement, proof, definitions, and
minimal prior-theorem closure. The language view renders the corresponding
assumptions, conclusion, proof operations, substitutions, and dependency closure.
The two views have shared record IDs and independently hashed contents. Separate
collector inputs permit different tokenizers and token lengths.

`fit_library_rosetta` extracts at the final normalized hidden state, after the
recipient adapter when present, at the last prompt token. The checkpoint,
tokenizer revision, boundary, position, input hashes, and parameter hashes are
recorded. Extraction runs before the current recipient update and cannot change
model parameters.

Both maps use paired supervision, with independent validation selection:

```text
T(x) = W x + b
objective(T) = sum_i ||W x_i + b - y_i||^2 + lambda ||W||_F^2
S(y) = V y + c
F(x, y) = (S(y), T(x))
F(F(x, y)) = (S(T(x)), T(S(y)))
```

The raw-coordinate sum objective leaves the intercept unpenalized. The earlier
standardized-coordinate implementation remains available and unchanged for
reproducing the published baseline. Each direction has shuffled-pair and
training-mean controls. Fixed reference vectors can fill missing slots in `F`;
they supply no semantic correspondence. Round-trip accuracy is a diagnostic,
not evidence that the learned pairing preserves meaning.

The current library calibration splits by theorem family. Its held-out geometry
examples may subsequently enter recipient training, and their proofs can reuse
earlier family results. They are not an independent capability benchmark.
Lean validates the formal theorem; correspondence of the language rendering
still requires a separate semantic audit.

## Persistent recipient learning and controls

`--experience-mode language` trains on admitted language choice examples. Its
control receives correct ordinary-language factual supervision. Both start from
identical weights and retain their own optimizer moments across iterations.

`--experience-mode translated` adds a hidden-state loss against `T(E_M(phi,p))`
to answer supervision. Five matched training arms run:

| Arm | Experience and auxiliary target |
| --- | --- |
| Verified translation | Verified language questions/answers and the paired translated mathematical state |
| Same-information text | The exact donor formal information supplied as text, with its own pre-update state as the auxiliary target |
| Shuffled correspondence | Identical verified questions/answers, with targets from the shuffled-pair translator |
| Mean target | Identical verified questions/answers, with the recipient calibration-training mean as target |
| Ordinary-language control | Correct ordinary fact questions/answers, with its own pre-update state as target |

Updates, examples seen, supervised answer tokens, padded token shapes, optimizer
settings, and initial weights are checked across arms. Nonpadding input lengths
can differ, especially for the text baseline. This matches training budgets;
it does not establish equal total collection, fitting, or inference cost. The
ordinary factual control is intentionally simple and is not a substitute for
every useful alternative curriculum.

The CPU backend is a small randomly initialized causal byte language model.
It is an engineering fixture. The pretrained backend uses a local, immutable
Hugging Face causal checkpoint and a trainable residual adapter at the output
head input. The recipient base is frozen and shared across arms; adapters,
gradients, and optimizer states remain separate. The mathematical checkpoint
has a separate frozen base. Neither backend downloads checkpoints by default.

## Independent evaluation and feedback

The chosen evaluation distribution is frozen and hashed before initialization
and training. All arms use the same questions, answers, and scoring rule. The
registered risk is mean negative log likelihood of the complete declared answer,
including EOS; accuracy chooses the lowest-loss declared answer. This loss
includes answer formatting/termination, so lower risk need not mean better
reasoning. Accuracy and control comparisons must also be inspected.

`benchmarks/independent-v1.jsonl` contains 24 explicitly authored arithmetic,
ordering, string, and everyday-rule questions outside the theorem generator.
It is a pipeline fixture, not a broad capability benchmark. Use an independently
selected, larger `--evaluation` dataset for research. IDs, prompts, and explicit
evaluation-role markers are rejected by training and curriculum scoring.

The next generator receives only post-update losses on accepted training
examples. Weak families receive more proposals; mastered families can advance
in constructor depth. The model checkpoint hash and exact feedback are retained.
External evaluation results never select problems, regularization, iteration
counts, layers, or stopping conditions. High training loss is a heuristic for
curriculum selection, not a measure of information gain.

## Run and inspect

Use Python 3.10+ and the pinned Lean installation described in `lean/README.md`.
The optional learning runtime is required:

```bash
python3 -m venv .venv-loop
source .venv-loop/bin/activate
python -m pip install -e '.[llm]'
python -m unittest discover -s tests -v
python experiments/run_constructive_loop.py \
  --output runs/my-loop --experience-mode translated \
  --evaluation benchmarks/independent-v1.jsonl \
  --iterations 2 --steps 2 --torch-threads 1
python experiments/verify_constructive_run.py runs/my-loop
```

Choose a fresh output directory for every registered experiment. Completed runs
retain the library, proposals, admission logs, formal/language pairs, raw
activations, both maps, initial and per-iteration checkpoints, optimizer states,
training budgets, evaluation rows, feedback, protocol, and complete file hashes.
The independent verifier replays Lean, reloads checkpoints, checks optimizer
continuity, recomputes evaluation, and validates translated/control targets.

For already-cached pretrained checkpoints, the same command supports actual
general-model adapter learning:

```bash
python experiments/run_constructive_loop.py \
  --output runs/my-pretrained-loop --backend pretrained \
  --experience-mode translated --iterations 2 --steps 1 \
  --recipient-model Qwen/Qwen2.5-1.5B \
  --recipient-revision 8faed761d45a263340a0528343f099c05c9a4323 \
  --math-model Qwen/Qwen2.5-Math-1.5B \
  --math-revision 4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2 \
  --evaluation benchmarks/independent-v1.jsonl \
  --device mps --max-length 1024 --torch-threads 1
```

The command rejects overlong inputs instead of truncating mathematical context.
Larger libraries need larger context/resource budgets or a separately evaluated
representation policy. Pinned model licenses remain separate; pretrained base
weights are not bundled.

## Evidence and remaining research

The [backend smoke](../artifacts/constructive-loop/hf-adapter-smoke.json) checks a
real cached Qwen adapter update, frozen base parameters, and exact checkpoint
reload. The [new semantic audit](../artifacts/constructive-loop/semantic-audit.json)
checks five examples from each constructive family, including dependency reuse
and 15 corresponding wrong constructions rejected by Lean. This is agent review;
Ian's earlier human approval covers only the original 15 baseline examples.

The [independent run review](../artifacts/constructive-loop/independent-review.json)
recomputes extraction, maps, auxiliary targets, controls, feedback, and checkpoint
evaluations. Execution measurements are retained alongside this document in
the published branch artifacts and release bundle. Earlier three-arm runs were
exploratory; shuffled and mean training controls were added before the final
five-arm executions, without changing the seed, layer, optimizer, or budget.

The next scientific gate is reproducible improvement over informative,
equal-budget controls on a larger fixed independent distribution, with multiple
seeds and structural/task-family uncertainty. Sustained improvement requires
further cycles where the learned capability demonstrably improves the value of
subsequent generated experience. Neither a growing proof archive, a low vector
error, nor successful execution of this loop establishes that result.
