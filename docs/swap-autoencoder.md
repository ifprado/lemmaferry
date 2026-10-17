# Two-port swap translator

The opt-in `--translator swap` backend integrates the existing Apache-2.0
LemmaFerry swap prototype into the constructive learning loop. The sibling
prototype is not imported at runtime. `--translator affine` remains the default;
existing affine bundles and fitting commands retain their behavior.

## Architecture and slots

The small bridge has two encoders and two decoders, separate from the large
models that produced its input activations:

```text
T(x) = D_L(E_M(x))
S(y) = D_M(E_L(y))
F(x, y) = (S(y), T(x))
F(x_i, y_j) approximately equals (x_j, y_i)
```

Cross-routing is fixed wiring. The output at each port depends only on the
opposite input. Domain dimensions may differ. `Slot` stores a numeric matrix and
one boolean presence flag per row. Empty slots have zero payloads and false
flags; a present zero vector has a true flag. The output swaps the flags, masks
empty payloads to exact zeros, and supports:

```text
F(x, empty_L) = (empty_M, T(x))
F(empty_M, y) = (S(y), empty_L)
F(F(x,y)) approximately equals (x,y)
```

The last equality measures learned round-trip consistency. It is not a guarantee
of semantic equivalence. The optional `ReversibleSwap` retains the equal-domain
dimension restriction and algebraic inversion; it is not another loop backend.
The historical affine `F(None, y)` reference-vector behavior is preserved for
compatibility and differs from the explicit masked neural slot interface.

## Loss, normalization and controls

All neural errors use per-coordinate training-standardized coordinates:

```text
loss = paired + 0.25*self + 0.10*cycle + 0.01*latent
paired = MSE(T(x), y) + MSE(S(y), x)
self = MSE(D_M(E_M(x)), x) + MSE(D_L(E_L(y)), y)
cycle = MSE(S(T(x)), x) + MSE(T(S(y)), y)
latent = MSE(E_M(x), E_L(y))
```

The weights are configurable fixed defaults. Training fits normalization once.
Validation selects the best bidirectional translation checkpoint on a fixed
schedule; every arm completes its full update budget. Test and the independent
capability benchmark do not select checkpoints or settings.

The shuffled neural control uses identical initial weights, optimizer settings,
batch indices, update budget and validation schedule. A recorded derangement
changes training pairs to `(x_i, y_perm[i])` consistently in both translation
directions and every correspondence-dependent term. Self and cycle terms still
reconstruct their own inputs. Training-mean controls use the same frozen domain
statistics. Affine comparisons are descriptive and are not compute matched to
neural training.

## Audited inputs and saved checkpoints

`fit_activations.paired_collections` retains the legacy same-input contract:
equal tokenizer fingerprints, original input hashes, ordered IDs, structural
families, token hashes and lengths. `fit_rosetta.paired_domains` handles distinct
formal/language views through a bound correspondence manifest. The latter can
have different tokenizers and lengths. Both retain immutable model revisions,
representation positions, dimensions, archive hashes and disjoint split checks.
Neither loader relaxes its protocol for the neural backend.

`lemmaferry.translators` supplies `fit_translator` and `load_translator`. Both
backends expose `to_language` and `to_math` with `control="paired"`, `"shuffled"`
or `"mean"`, plus evaluation and save/load. Importing the factory and using affine
does not import Torch. Install `.[swap]` for saved-vector neural fitting or
`.[llm]` for the pretrained recipient backend.

The versioned `lemmaferry.swap.bundle.v1` bundle contains `manifest.json` and
numeric `maps.npz`; loading never unpickles Python objects. The manifest records
architecture and dimensions, configuration and losses, seed, split fingerprints,
normalization, selected checkpoints and validation traces, training budget,
pairing permutation, model/library extraction provenance and integrity hashes.
Both arm weights and normalization buffers are restored with strict keys,
shapes, dtypes and finite-value checks. Hashes detect changed artifacts; they
are integrity checks, not signatures from an external authority.

## Reproduce a bounded toy fit

Use a fresh output directory each time. From the repository root, an existing
compatible learning environment can run the registered CPU configuration:

```bash
.venv-llm/bin/python experiments/compare_translators.py \
  --synthetic --output ../lemmaferry-archive/local-runs/my-swap-toy \
  --seeds 7 --swap-latent-dim 8 --swap-hidden-dim 32 \
  --swap-steps 512 --swap-batch-size 32 \
  --swap-learning-rate 0.002 --swap-validation-every 64 --torch-threads 1
```

This uses 512/96/128 independently drawn samples of favorable planted nonlinear
views, dimensions 12 and 10. It is not a theorem-family generalization test.
The command writes a protocol before fitting, raw evidence and checkpoints,
bidirectional paired/shuffled/mean metrics, affine comparisons, costs, and a
summary. Saved checkpoint predictions must match their in-memory originals.

## Fit audited actual-model vectors without loading the models

The existing collections were produced by pinned Qwen2.5 Math/general 1.5B
models at layer 14. They contain 256/64/128 pairs of identical input text, with
1,536 coordinates in each domain. The registered comparison is:

```bash
.venv-llm/bin/python experiments/compare_translators.py \
  --source ../lemmaferry-archive/local-runs/transfer-v1/source \
  --target ../lemmaferry-archive/local-runs/transfer-v1/target \
  --output ../lemmaferry-archive/local-runs/my-swap-actual \
  --seeds 7 19 31 --swap-latent-dim 16 --swap-hidden-dim 64 \
  --swap-steps 256 --swap-batch-size 32 \
  --swap-learning-rate 0.002 --swap-validation-every 32 --torch-threads 1
```

For one same-input fit, use `experiments/fit_activations.py` with the same
`--source`, `--target`, `--output`, `--translator swap`, `--seed 7` and
`--swap-*` options. For independently collected formal/language views, use:

```bash
.venv-llm/bin/python experiments/fit_rosetta.py \
  --source /path/to/formal-collection --target /path/to/language-collection \
  --correspondence /path/to/bound-views --output /path/to/new-fit \
  --translator swap --seed 7 --swap-steps 128 --swap-validation-every 16
```

`compare_translators.py` also accepts `--correspondence` for that separate
protocol. Arrays from different prompts cannot be passed through the legacy
same-input loader by dropping checks.

## Execute and independently verify two learning cycles

Install the real Lean toolchain pinned in `lean/lean-toolchain`. The registered
tiny recipient run is:

```bash
.venv-llm/bin/python experiments/run_constructive_loop.py \
  --output ../lemmaferry-archive/local-runs/my-swap-loop \
  --evaluation benchmarks/independent-v1.jsonl \
  --backend tiny --experience-mode translated --translator swap \
  --iterations 2 --proposals-per-iteration 6 --seed 1729 \
  --steps 2 --batch-size 1 --max-length 2048 --auxiliary-weight 0.1 \
  --swap-latent-dim 8 --swap-hidden-dim 16 --swap-steps 32 \
  --swap-batch-size 4 --swap-learning-rate 0.002 --swap-validation-every 8 \
  --device cpu --torch-threads 1 --lean "$HOME/.elan/bin/lean"
.venv-llm/bin/python experiments/verify_constructive_run.py \
  ../lemmaferry-archive/local-runs/my-swap-loop --torch-threads 1
.venv/bin/python -m unittest discover -s tests -v
.venv-llm/bin/python -m unittest discover -s tests -v
```

Each iteration extracts from the frozen donor and current recipient before its
update, records both states and fits a fresh bridge. Targets are detached and
copied before recipient training. All five arms remain: paired translation,
same-information text, shuffled correspondence, mean target, and ordinary
language supervision. Recipient weights and optimizer moments persist; only
training-curriculum losses guide subsequent generation. The fixed independent
evaluation remains outside fitting and feedback.

The independent verifier dispatches by backend, reloads neural checkpoints,
reconstructs targets from saved activations and checks their provenance. Real
Lean admission is replayed. Neural comparisons use dtype-aware tolerances;
the existing tighter affine comparisons remain unchanged. CI exercises both
the default affine loop and the opt-in swap loop.

## Interpretation

The loop consumes translated states as auxiliary **training targets**. Fitting
saved vectors measures geometric alignment. Recipient learning measures a
separate update. Inference assistance requires a separately tested intervention,
and external capability requires an independently evaluated task distribution
and budget controls. This feature does not introduce an inference-time model
communication interface or establish sustained broad reasoning improvements.

Fixed budgets and acceptance criteria are in
[`swap-validation-plan.json`](swap-validation-plan.json). Small executed summaries
are retained in `results/swap-integration/`; raw arrays, checkpoints and logs stay
in the fresh archive run named by the summary.

## Executed validation

The [JSON summary](../results/swap-integration/summary.json) and
[exact command receipts](../results/swap-integration/commands.json) retain these
results. All 109 tests passed with Torch 2.14.1 and NumPy 2.5.4 on Python 3.14.7.
The NumPy-only environment passed 73 tests and skipped 36 optional-runtime tests.
The prototype's 43 source and historical artifact files were hash-checked and
remained unchanged.

Held-out normalized RMSE, lower is better:

| Data and method | Math → language | Language → math |
| --- | ---: | ---: |
| Synthetic swap, paired, seed 7 | 0.0952 | 0.1018 |
| Synthetic swap, shuffled | 0.9641 | 0.9758 |
| Synthetic training mean | 0.9872 | 0.9963 |
| Synthetic affine, paired | 0.3578 | 0.4981 |
| Actual saved Qwen vectors, swap paired, mean of seeds 7/19/31 | 0.7550 | 0.7586 |
| Actual saved Qwen vectors, swap shuffled, mean of seeds | 1.2121 | 1.1354 |
| Actual saved Qwen vectors, training mean | 1.1257 | 1.0957 |
| Actual saved Qwen vectors, affine paired | 0.6768 | 0.6913 |

The synthetic acceptance criterion passed. On the actual saved activations, the
neural bridge beat shuffled and mean controls, but **affine had lower error**.
No settings were changed in response to test results. The affine comparison is
descriptive: its fitting algorithm and computational budget differ from neural
training. The actual inputs are identical-text collections, not newly collected
formal/language pairs.

Each actual neural arm had 400,672 trainable parameters, completed 256 optimizer
updates and saw 8,192 pairs. Both arms within each seed had matching initial and
batch-order hashes, update counts and validation schedules. Synthetic arms had
2,598 parameters, 512 updates and 16,384 pair exposures each. The measured CPU
comparison commands took 1.58 seconds for synthetic and 10.61 seconds for actual
vectors, including writing and reloading evidence. Actual paired/shuffled neural
fitting totaled 3.16 seconds across the three seeds; descriptive affine fitting
totaled 1.47 seconds. These local timings are not portable performance claims.

The registered loop admitted six lemmas in each of two iterations using Lean
4.35.0; all six second-iteration lemmas reused earlier results. Each of five arms
retained four cumulative recipient updates. Independent verification checked 88
hashed files, replayed real Lean, reloaded ten recipient checkpoints, re-extracted
both calibration domains and reproduced neural/control targets and metrics. A
copy with an altered target was rejected even after its envelope hash was
updated. Loop execution took 4.97 seconds; independent verification took 1.40
seconds.

All five arms scored 14/24 (58.3%) on the small fixed evaluation after two
iterations. Paired swap had mean answer NLL 16.3090, versus 16.2321 for
same-information text, 16.3084 shuffled, 16.3088 mean and 16.3132 ordinary-language
control. Swap therefore showed no accuracy advantage and did not beat text on
answer loss. This tiny randomly initialized recipient is an engineering check,
not evidence about broad general-model reasoning.
