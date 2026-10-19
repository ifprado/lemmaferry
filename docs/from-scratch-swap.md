# A local study trained from random initialization

This opt-in pipeline asks whether more distinct Lean-verified deduction problems
help a small language model, and whether translated proof vectors help beyond
the same proof information supplied as text. It trains every neural weight
locally from random initialization. It does not use pretrained weights, hosted
inference, Qwen activations, or a tokenizer fitted on synthetic test problems.

The benchmark measures controlled constructive deduction in ordinary-language
questions. It does not establish broad reasoning or general intelligence. A null
or negative result is a completed research outcome. A resource-limited partial
run remains explicitly incomplete.

The executed three-seed, 36-condition study is complete. Its paired arm did not
improve question-only reasoning over the text-only control at any tested pool
size. All arms remain close to 50% A/B accuracy. See the
[compact results and limitations](../artifacts/from-scratch/README.md), standalone
report, and scientific figures for exact counts, variation, geometry, and costs.

Generation uses a fixed bounded typed grammar. Extension tests continued learning
from additional admitted structures; this study does not test whether an improved
recipient can guide a better mathematical generator. Sustained feedback and
transfer outside propositional deduction remain separate research questions.

## Models and the information path

The recipient is a standard decoder-only Transformer: UTF-8 byte embeddings,
learned positions, pre-normalized causal multi-head attention, feed-forward
blocks, residuals, a final layer norm, and a byte output head. The initial
configuration is six blocks, width 256, four heads, and context 512; the
throughput pilot selects the registered configuration before test evaluation.
All recipient parameters train. The byte vocabulary is fixed:
`PAD=0`, `BOS=1`, `EOS=2`, and byte values plus three. No vocabulary fitting can
leak a larger synthetic pool into a smaller-data condition.

The mathematical encoder is a separately trained, randomly initialized masked
convolutional proof autoencoder. It compresses canonical proof bytes through a
64-dimensional vector. Its parallel decoder receives only that vector and
positions: there is no copied input, attention to input tokens, teacher forcing,
or token bypass. Validation selects the reconstruction checkpoint, which is
then frozen. Real, shuffled, and zero-vector reconstruction losses and embedding
variance are reported to expose collapse or an ineffective bottleneck.

For each bridge refresh, the proof encoder produces `x`; a frozen snapshot of
the current recipient reads a complete solution to produce `y`. The solution
contains the question, correct answer, and canonical proof term. The recipient
state is taken after the final layer norm at the last solution byte, before EOS.
The bridge fits on training pairs, selects a checkpoint on validation pairs,
and evaluates held-out geometry separately.

The solution view appends the raw Lean proof term. Its hypothesis names refer
to the associated formal record; the rendered premise order can differ. This
view is not a self-contained natural-language proof explanation. The separate
proof-as-text examples include the complete canonical premises, goal, and term
for every arm. Random atom names and option order add nuisance variation to the
language states, which can limit alignment.

```text
T(x) = D_L(E_M(x))
S(y) = D_M(E_L(y))
F(x,y) = (S(y),T(x))

bridge loss = paired + 0.25*self reconstruction + 0.10*cycle + 0.01*latent agreement
recipient loss = supervised byte CE + alpha * mean(((h(question)-T(x))/training_std)^2)
```

The bridge has fixed cross-routing and explicit slot-presence masks, as described
in [the swap translator documentation](swap-autoencoder.md). Its four small
neural maps are separate from both large text paths. Reconstruction and cycle
consistency are geometric diagnostics, not evidence of semantic transfer.

At recipient training time, `h(question)` is the final question byte, before any
answer. Translated vectors are detached auxiliary **training targets**. At
inference, the evaluator reloads only the recipient and receives the question;
it cannot access a proof, proof encoder, solution rendering, or bridge. This is
not an inference-time model communication interface.

## Data, verification, and held-out structures

The generator constructs typed implication/conjunction proofs from explicit
premises and a small fixed common lemma library. It checks the proof certificate,
creates a false alternative with an explicit Boolean countermodel, and renders
an A/B question. Actual Lean admission and axiom audits are mandatory before an
example is called verified. The fixed common lemmas are dependencies, not extra
training examples.

The two answer options have the same complete connective-tree skeleton,
connective counts, formula depth, and printed byte length. Atomic names all
occupy four bytes. A false alternative is formed by independently reassigning
leaves within that skeleton and is accepted only with a countermodel satisfying
all premises. Choosing by option length, operator counts, or unlabelled tree
shape therefore ties; a fixed-label tie break scores 50% on balanced labels.
The deeper theorem holdout increases the structure of both options together;
answer-option shape does not identify the correct alternative.

This matching follows a pretraining data audit: the discarded first dataset had
an option-length shortcut (about 72% ordinary and 86% harder accuracy). No main
recipient training used that dataset. Generation and the disjoint development
pilot are repeated for the matched-option dataset before freezing the study.
These controls remove the identified cues; they do not establish that every
possible surface heuristic has been excluded.

Canonical theorem structures are deduplicated across train, validation, test,
and the harder split. Hash buckets reserve ordinary held-out structures;
ordinary theorem formula depth is at most three, while harder theorem formula
depth is at least four and its generated proof depth is at least five. These
are intrinsic formula-shape and proof-certificate distribution shifts, not lower
bounds on the shortest possible proof or a claim of minimum difficulty. Tests
and their labels never select model size, loss weights, bridge checkpoints, or
curriculum updates. The independent verifier checks certificate validity,
negative countermodels, identities, dependency isolation, and real Lean receipts.

The default ladder contains 256, 1,024, and 4,096 unique training problems, with
three seeds. The training pools are nested. Each size restarts from the same
ordinary-text pretrained checkpoint for its seed, including optimizer state;
the scaling curve is not a cumulative continuation curve. The frozen plan and
selected configuration, rather than this default, govern an executed run.

The common starter corpus is *Alice's Adventures in Wonderland* by Lewis Carroll,
Project Gutenberg ebook 11, Millennium Fulcrum Edition 3.0. It is public domain in
the USA; retain the Gutenberg license in the downloaded source. Its pinned
edition, exact SHA-256, byte count, URL, and 90%/10% byte split are in
[`configs/from-scratch-corpus.json`](../configs/from-scratch-corpus.json). The
corpus is split before tokenizer/model training. Its weights are learned here;
no external model supplies labels or representations.

## Set up once, then work offline

From the repository root, create an isolated environment with a native compatible
Python and install project dependencies. A working environment can be locked:

```bash
python3 -m venv .venv-scratch
.venv-scratch/bin/python -m pip install -e '.[swap]'
.venv-scratch/bin/python -m pip freeze > scratch-environment.lock.txt
```

The executed environment lock is an artifact of the actual run. Python and Torch
must support the local machine; provisioning a compatible Python in isolation
must not replace the system interpreter. A CPU option is retained. MPS allocator
protections remain enabled.

Create a fresh run directory and download the single pinned corpus file:

```bash
LF_RUN="../lemmaferry-archive/local-runs/from-scratch-20261019-study"
mkdir -p "$LF_RUN/corpus"
curl --fail --location https://www.gutenberg.org/ebooks/11.txt.utf-8 \
  --output "$LF_RUN/corpus/alice.txt"
```

`preflight` refuses a changed corpus hash. A later Gutenberg revision must be
handled as a separately documented corpus version, not silently accepted.
Install the real toolchain specified in `lean/lean-toolchain` before generation.
With elan already installed, the pinned toolchain command is:

```bash
"$HOME/.elan/bin/elan" toolchain install leanprover/lean4:v4.35.0
(cd lean && "$HOME/.elan/bin/lean" --version)
```

The proof checker discovers the installed pinned compiler through elan and verifies
its version; no global default toolchain is required. A compiler exit code without
the axiom audit is insufficient.
After dependencies, the Lean toolchain, and this corpus are installed, the
experiment needs neither a network connection nor Codex.

## Run the registered study

These commands use one CLI. Configuration is a global option before the
subcommand. The default configuration is
[`configs/from-scratch.json`](../configs/from-scratch.json).

```bash
.venv-scratch/bin/python -m lemmaferry.scratch --run "$LF_RUN" --device mps \
  --config configs/from-scratch.json preflight
.venv-scratch/bin/python -m lemmaferry.scratch --run "$LF_RUN" --device mps generate
.venv-scratch/bin/python -m lemmaferry.scratch --run "$LF_RUN" --device mps pretrain
.venv-scratch/bin/python -m lemmaferry.scratch --run "$LF_RUN" --device mps study
.venv-scratch/bin/python -m lemmaferry.scratch --run "$LF_RUN" --device mps verify
.venv-scratch/bin/python experiments/verify_scratch_baselines.py --run "$LF_RUN" --device cpu --samples 8
.venv-scratch/bin/python -m lemmaferry.scratch --run "$LF_RUN" --device mps report
```

Use `--device cpu` for the tested CPU path. The development pilot discards its
weights. It measures forward/backward throughput, overfitting a development
batch, memory, and feasible budgets before test scores are opened. A separate
development-only pilot determines the registered auxiliary setting. It uses 80
reserved theorems (64 train, 16 validation) that are excluded from the main data.
Both alpha candidates receive the same fixed 64 recipient updates. The chosen
alpha is recorded in `development.json` after validation answer NLL is compared
on the matched-option development data; the pilot weights are discarded. The
earlier dataset’s selected alpha is not carried forward. This selection does
not open the main test sets. Check
`pilot.json`, `development.json`, `selected-config.json`, and `plan.json` for the
actual decision and evidence. The six-hour experiment cap excludes coding and
installation; it is a configurable ceiling, not a promised completion time.
Arms run sequentially. Memory observations are labeled as measured or sampled,
and guards are not disabled to make a configuration fit.

`study` already evaluates every completed condition. The separate `evaluate`
subcommand reruns all recipient-only evaluations when a full replay is desired;
`verify` performs the independently implemented, explicitly sampled replay.

Every size and seed runs four recipient conditions:

| Condition | Supervision |
|---|---|
| `text_only` | Correct question answers and the same proof bytes as ordinary text |
| `swap_paired` | Identical text supervision plus correctly paired translated-vector targets |
| `swap_shuffled` | Identical text supervision plus a bridge fitted on shuffled correspondences |
| `swap_mean` | Identical text supervision plus the frozen training-mean target |

Recipient update counts, problem order, fixed padded-token budgets, and
supervised-token budgets match. Each problem exposure supplies three QA tokens
(space, answer, EOS) and 64 selected proof tokens. Auxiliary target arrays are
frozen before updates. Encoder/bridge fitting and extraction costs are reported
separately; matched recipient budgets do not imply equal end-to-end compute.
At documented stage boundaries, each arm refreshes its bridge against its own
current recipient snapshot. Weights and optimizer moments persist within that
condition. Early conditions never see a larger pool's training examples.

## Read and verify the results

The report command produces these local standalone artifacts:

- `report.html`: inline SVG scaling curves, seed variability, exact counts,
  generation, NLL, vector diagnostics, geometry, budgets, costs, and provenance.
  It contains no remote scripts, fonts, stylesheets, or assets.
- `summary.json`: compact per-seed/size/arm metrics and execution evidence.
- `results.csv`: ordinary and harder split scores, exact correct/total counts,
  likelihood accuracy, free-generation accuracy, NLL, and paired deltas from text.

For separate SVG/PNG scientific figures, install the optional plotting tools
during setup and export only from the report-generated summary:

```bash
.venv-scratch/bin/python -m pip install -e '.[swap,plots]'
.venv-scratch/bin/python experiments/plot_scratch_results.py \
  --run "$LF_RUN" --output artifacts/from-scratch/plots
```

The accuracy figure shows ordinary/harder results for all four arms, observed
pretraining baselines, and seed variability. Its 50% A/B chance line is a
mathematical reference, not an empirical baseline. A separate two-direction
geometry figure holds the producer fixed to the `text_only` recipient's final
bridge stage; it compares paired, shuffled, and mean maps without mixing target
spaces from different recipient arms. The proof-encoder figure shows selected
validation reconstruction with real/shuffled/zero vectors and vector variance.
`plot-manifest.json` retains exact plotted values, counts, source/figure hashes,
and omitted figures. Partial summaries produce explicitly partial plots; absent
scores are never replaced by zero.

Likelihood accuracy scores A/B from question plus a fixed space. Free generation
starts from the question and uses exact answer extraction. Answer NLL includes
space, letter, and EOS; termination calibration must not be described as an
accuracy gain. Bridge geometry has its own plot and cannot substitute for
recipient scores. Seed standard deviations are descriptive dispersion, not
confidence intervals or a claim of generalization beyond these structures.

After the completed run's final report, reproduce the posthoc answer analysis
with the standard-library-only command:

```bash
.venv-scratch/bin/python experiments/summarize_scratch_answers.py --run "$LF_RUN"
```

It writes `answer-diagnostics.json` with normalized conditional A/B NLL,
per-seed prediction counts, constant-label rates, matched-seed paired/control
deltas, and input/source hashes. `--output PATH` changes the destination. This
is descriptive analysis with no model loading or tuning; regenerate it after
the final canonical report so its summary hash remains current.

The publication exporter summarizes the collapse report's coordinate-variance
and singular-energy arrays by count, canonical hash, minimum, maximum, and mean.
It retains all scalar variance/rank diagnostics, provenance, budgets, and traces;
an explicit publication note hashes the full local receipt containing the arrays.

Starting recipient checkpoints are `starting/{seed}.pt`. Each condition saves
`study/{seed}-{size}/{arm}/stage-{stage}/recipient.pt`; proof encoders, bridge
bundles, vector snapshots, logs, and integrity receipts stay beside them in the
local run. Checkpoints retain configuration, tokenizer, complete parameters,
AdamW moments, cumulative updates, and RNG state. Hashes detect changed files;
they are integrity receipts, not signatures by an external authority.

The verifier independently reloads saved checkpoints, recomputes target hashes
and budgets, checks all metric rows and split boundaries, and replays a declared
subset of expensive neural extractions and predictions. Its report states the
exact sample counts and tolerances. It must not describe sampled neural replay
as exhaustive. `report.html` remains labeled incomplete when conditions or
required evidence are missing; an interrupted study is not a negative result.

The report includes an actual training example traveling through all stages:
canonical Lean proof bytes → proof bottleneck `x` → translated `T(x)`, paired
with the frozen recipient's solution state `y` → auxiliary update at the
question state → recipient-only held-out answers. This traces information flow;
it does not imply that any individual coordinate has an identified meaning.

## Resume or extend

Rerun `study` with the same run directory and frozen configuration to resume
without replaying completed conditions. `progress.json` records completed stages
and the next continuation command. Partial checkpoints retain optimizer moments;
changing a frozen plan requires a separate run.

Generate and train on more verified data through the extension command:

```bash
.venv-scratch/bin/python -m lemmaferry.scratch --run "$LF_RUN" --device mps \
  extend --additional 256 --seed 7 --size 4096 --arm swap_paired --updates 128
.venv-scratch/bin/python -m lemmaferry.scratch --run "$LF_RUN" --device mps \
  sample --prompt 'Alice ' --tokens 32
```

An extension rejects duplicates and reserved held-out structures, refreshes the
proof/bridge representation at its declared boundary, resumes the recipient and
optimizer, and appends its own history. Its cumulative curve remains separate
from the registered independent scaling study. CLI output identifies exact
report and checkpoint paths. Extension artifacts must never replace the frozen
study's dataset, test labels, starting weights, or reported historical results.

Run the relevant checks with:

```bash
.venv-scratch/bin/python -m unittest discover -s tests -p 'test_scratch*.py' -v
.venv-scratch/bin/python -m unittest discover -s tests -v
```

Commit code, environment locks, configuration, tests, documentation, small
fixtures, compact results, plots/report, and hashes. Large corpora, environments,
checkpoints, raw logs, and vector dumps belong in the documented gitignored run
directory. Distinguish observed learning, any additional paired-translation
advantage, and unresolved limitations in the final results.
