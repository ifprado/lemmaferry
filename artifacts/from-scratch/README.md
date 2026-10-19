# From-scratch constructive-learning study

This directory publishes compact evidence for the registered study described in
[the implementation guide](../../docs/from-scratch-swap.md). Full checkpoints,
activation arrays, generated corpora, and raw logs remain in the local archive;
`evidence-manifest.json` records their hashes and location.

The completed 36-condition run found **no useful paired-translation advantage**
and no monotonic improvement from more distinct problems at the fixed recipient
budget. Question-only A/B likelihood accuracy, averaged over seeds 7, 19, and 31:

| Split | Distinct training problems | Text only | Paired | Shuffled | Mean |
|---|---:|---:|---:|---:|---:|
| Ordinary | 256 | 50.72% | 48.57% | 50.33% | 49.93% |
| Ordinary | 1,024 | 49.09% | 48.89% | 49.74% | 49.15% |
| Ordinary | 4,096 | 49.67% | 48.83% | 50.00% | 50.00% |
| Deeper structure | 256 | 51.24% | 50.91% | 49.54% | 49.80% |
| Deeper structure | 1,024 | 48.37% | 47.85% | 49.22% | 48.50% |
| Deeper structure | 4,096 | 50.39% | 49.35% | 50.00% | 50.00% |

Each seed uses the same 512 held-out questions in each split. The 1,536 summed
evaluations therefore repeat 512 theorem structures; they are not 1,536
independent theorem samples. Per-seed counts and descriptive sample standard
deviations are in the CSV, summary, and figures. Greedy generation is evaluated
separately and also stays near chance. These observations concern this small
model, representation interface, grammar, and training budget.

Ordinary-text validation loss decreases during pretraining, and real proof
vectors improve byte reconstruction relative to shuffled/zero vectors. Some
bridge geometry improves over the mean control. Those are distinct findings:
neither establishes useful deduction. QA token NLL averages a predictable space
and EOS with the A/B letter; it can approach `log(2)/3` while A/B choice remains
at chance. `answer-diagnostics.json` reports conditional choice NLL and constant
label predictions so format learning is visible.

The proof encoder remains lossy (about 39–44% selected validation byte accuracy),
and Lean verifies its input certificates rather than its learned vectors. Its
canonical input also omits query-specific option order and surface atom mapping
that occur in the recipient solution state. Whole-state regression can therefore
penalize nuisance information unavailable to the donor. The objective supplies
64 proof-byte targets and three QA targets per exposure, with only one variable
answer letter. The negative result does not test all possible representation
interfaces, objectives, or compute budgets.

The recipient has 5,002,752 parameters: six causal Transformer blocks, width 256,
four heads, a 512-byte-token context, and a fixed UTF-8 byte vocabulary. All
recipient weights start without pretrained parameters and train locally. An
independent proof autoencoder supplies a frozen 64-dimensional mathematical
representation. The crossed swap bridge maps between that representation and
the current recipient's 256-dimensional solution state.

The registered grid contains three seeds (7, 19, 31), three nested pools (256,
1,024, 4,096 distinct problems), and four arms (text, paired, shuffled, mean).
Each condition receives 1,024 recipient updates in two stages: 8,192 problem
exposures, 548,864 supervised byte tokens, and 8,388,608 padded input tokens.
The largest pool is consumed once in each stage. All arms receive the same
answers and complete canonical proofs as text; auxiliary targets are the only
difference in recipient supervision. Encoder and bridge costs are separate.

The frozen main dataset contains 4,096 training, 128 validation, 512 ordinary
test, and 512 deeper-structure test problems. Actual Lean 4.35.0 checks all
certificates with zero-axiom audits. Atom-renaming and premise-order identities
stay in one partition. The ordinary/harder split uses intrinsic formula depth,
with an additional proof-depth requirement; it does not establish shortest
proof difficulty. Every A/B pair has the same connective tree and printed byte
length, and every negative has a Boolean countermodel.

An initial dataset was abandoned before main synthetic training after an audit
found an answer-length shortcut. The corrected data and development pilot were
regenerated. The corrected pilot selected auxiliary weight 0.1 by validation
answer NLL from candidates 0.01 and 0.1. Its 80 development theorems are excluded
from the main splits, and its weights are discarded. The reporting-only source
amendment also predates main synthetic fitting and training; its receipt retains
the earlier plan hash and confirms unchanged experimental code and settings.

The final test suite completed 140 cases: 137 executed successfully and three
optional pretrained-runtime cases skipped. Independent CPU replay passed all 36
conditions, the continuation, and all three starting baselines; fresh Lean checks
passed all 5,504 main and extension certificates. The active experiment clock was
about 75.45 minutes. The separately recorded verification/postprocessing interval
brings the total to about 78.11 minutes at the overall-budget receipt; earlier
regression tests and fixtures are separate.

Read the evidence in this order:

1. `report.html` and `results.csv`: all seed/size/arm scores and exact counts.
2. `plots/`: standalone SVG/PNG accuracy, geometry, and reconstruction figures.
3. `summary.json`, `plan.json`, and `pilot*.json`: protocol and measured settings.
4. `independent-verification.json`: full integrity checks plus explicitly sampled
   neural and question-only prediction replay, with tolerances and counts.
5. `baseline-independent-verification.json` and `collapse-diagnostics.json`:
   additional baseline replay and validation-only variance/rank measurements.
6. `example-audit.json`: an AI audit of five examples per split, explicitly
   distinguished from human confirmation.
7. `continuation-summary.json`: exploratory continued learning, kept separate
   from the independent scaling comparison.
8. `evidence-manifest.json`: published-file and raw-evidence SHA-256 values.

Likelihood accuracy and unconstrained greedy answer generation are distinct
metrics. Seed standard deviations describe three runs; they are not confidence
intervals. Proof reconstruction and bridge alignment do not establish useful
reasoning transfer. These tasks concern controlled propositional deduction;
sustained feedback and broader reasoning remain untested.

To reproduce the publication after completing and independently verifying a run:

```bash
.venv-scratch/bin/python -m lemmaferry.scratch --run "$LF_RUN" --device cpu verify
.venv-scratch/bin/python experiments/verify_scratch_baselines.py --run "$LF_RUN"
.venv-scratch/bin/python experiments/inspect_scratch_collapse.py \
  --run "$LF_RUN" --output "$LF_RUN/collapse-diagnostics.json" \
  --recipient-samples 128 --require-complete
.venv-scratch/bin/python -m lemmaferry.scratch --run "$LF_RUN" --device cpu report
.venv-scratch/bin/python experiments/summarize_scratch_answers.py --run "$LF_RUN"
.venv-scratch/bin/python experiments/plot_scratch_results.py \
  --run "$LF_RUN" --output artifacts/from-scratch/plots
.venv-scratch/bin/python experiments/export_scratch_evidence.py \
  --run "$LF_RUN" --output artifacts/from-scratch
```

The exporter requires a complete passed 36-condition run and checks that the
report contains the final verification receipt. It never copies raw model
weights, vector arrays, generated corpora, or full prediction dumps.

The published collapse report summarizes coordinate-variance and singular-energy
arrays by count, canonical hash, minimum, maximum, and mean. Scalar variance and
rank diagnostics, provenance, budgets, and traces remain complete. Its publication
note binds the full local receipt, where the original arrays remain available.

The answer summarizer uses only the Python standard library and completed
evaluation JSON files. It writes `answer-diagnostics.json`, binding the final
report, plan, evaluation files, and its own source by SHA-256. Its conditional
A/B NLL, constant-label counts, and matched-seed deltas are descriptive
postprocessing; it does not load models or select training settings. Run it
again after regenerating the canonical report to refresh the summary binding.
