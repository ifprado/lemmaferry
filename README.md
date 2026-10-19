# LemmaFerry

**Verified mathematics, measurable transfer.**

LemmaFerry is an experimental workbench for growing a verified constructive library, learning correspondences between mathematical and language representations, and testing whether that experience improves a language model.

The hypothesis is that constructive mathematics can supply expandable, verifiable learning experience, and that a learned correspondence can make it useful beyond the generated proofs. This branch implements bounded **generation → verification → translation → learning → feedback-guided generation** cycles. Sustained broad improvement remains a conjecture to test.

## Start here

- Read the [constructive learning-loop implementation](docs/constructive-loop.md), including the foundation, cumulative library, two translators, persistent recipient updates, independent evaluation, and feedback controls.
- Use the optional [two-port swap autoencoder](docs/swap-autoencoder.md) with `--translator swap`; affine remains the default. The guide includes audited saved-vector fitting, controls, checkpoints, and two-cycle verification commands.
- Run the [local study from random initialization](docs/from-scratch-swap.md): a byte Transformer, independent proof-vector autoencoder, real Lean admission, matched text/paired/shuffled/mean training, structural holdouts, checkpoint continuation, and an offline HTML report. This is a controlled deduction experiment; broad transfer remains unestablished.
- Inspect its [completed three-seed results](artifacts/from-scratch/README.md) and [standalone report](artifacts/from-scratch/report.html): all 36 conditions ran, with no paired-translation advantage and question-only accuracy near chance.
- Download the [constructive-loop branch release](https://github.com/ifprado/lemmaferry/releases/tag/constructive-loop-v0.2.0), with source, execution evidence, and SHA256 checksums.
- The [earlier frozen baseline and first-model release](https://github.com/ifprado/lemmaferry/releases/tag/research-2026-10-17) remains available; its source snapshot predates the separate human audit confirmation linked below.
- Read the [first actual-model results](docs/validation.md#first-actual-model-experiment): vector alignment improved, but the translated state did not beat ordinary answers or supplying the same proof as text in answer accuracy.
- Inspect the [latest alignment reproduction](artifacts/lf-001/reproduction.json) and [real Lean-verification report](docs/lf-002-lean-verification.md). The accepted dataset is in [`results/lf-002/checked/`](results/lf-002/checked/), with 20 verified examples per split; `examples/logic/` retains the original three-example preparation samples.
- Read the [actionable roadmap](docs/roadmap.md), or open [the interactive roadmap](docs/index.html) in your browser.
- Run the first two CPU experiments below.
- Pick one [issue from the backlog](docs/issues.md), then use an [experiment card](docs/experiment-card.md).
- Follow [the GitHub launch checklist](docs/launch.md) to publish your copy.

## Quick start

Python 3.10 or newer is required; the supplied CPU runs used Python 3.12, NumPy 2.3.5, and Matplotlib 3.10.8. NumPy is the only mandatory third-party dependency. No model downloads, API keys, or GPU are required for these commands.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python experiments/toy_alignment.py --seed 7 --output artifacts/my_alignment.json
python experiments/generate_logic.py --count 20 --seed 7
```

The last command generates 20 examples **per split**, including formal proof terms, exact rule-chain answers, controlled fictional-language renderings, and provenance. It does not execute Lean. To draw an optional plot:

```bash
python -m pip install -e '.[plots]'
python experiments/toy_alignment.py --seed 7 --output artifacts/my_alignment.json --plot artifacts/my_alignment.svg
```

## What is implemented

| Component | Status and scope |
| --- | --- |
| Cumulative constructive library | Content-addressed, append-only records, explicit foundation and premises, dependency reuse, structural deduplication, real Lean admission, and fresh replay on reopening. |
| Library-conditioned generation | Parameterized implication chains and conjunction introduction/elimination. Later proofs invoke prior accepted lemmas; recipient training losses guide subsequent family/depth allocation. |
| Affine ridge translator | Implemented and tested on artificial numeric vectors with planted shared factors. Includes training-only preprocessing and held-out data. |
| Misalignment controls | Shuffled-anchor and mean-output baselines; symmetric-cloud counterexample with correct geometry and wrong semantics. |
| Formal task generator | Implemented constructive propositional implication graphs. Short and longer valid rule chains share the same theorem. |
| Fictional-language renderer | Implemented from the same graph, with a checked controlled-format round trip. It is not a learned semantic translator. |
| Lean runner | Implemented timeouts, per-example statuses and zero-axiom audit. A [follow-up run](docs/lf-002-lean-verification.md) verified 60 generated examples and both proofs of each with Lean 4.35.0 and recognized zero-axiom audits. |
| Actual LLM collector | Executed on pinned Qwen2.5 Math/general 1.5B checkpoints: CPU smoke checks, a 30-pair pilot, and a separate 448-pair collection with matching tokenizers and input hashes. |
| Real-model affine fitter | Executed with 256 training, 64 validation, and 128 test pairs. Validation selected ridge alpha `1.0`; test MSE was 0.290 times the training-mean baseline. |
| Bidirectional Rosetta maps | Learned `T` and `S`, the exact raw-coordinate ridge objective, separate formal/language inputs, validation-selected regularization, round trips, fixed reference slots, shuffled-pair and mean controls. |
| Two-port swap autoencoder | Optional crossed encoders/decoders, explicit empty-slot presence flags, unequal dimensions, train-only normalization, matched shuffled neural controls, validation-selected numeric checkpoints, and constructive-loop auxiliary targets. |
| Local training from random initialization | Executed a 5,002,752-parameter causal Transformer and separate proof encoder on 256/1,024/4,096 verified training problems, three seeds and four matched recipient arms. Saved-state continuation and independent replay passed; no useful reasoning transfer was observed. |
| Activation intervention | Implemented last-prompt-token patching at layer 14 with self-patch and shuffled/mean/zero controls. Accuracy was 51.6%, versus 53.9% ordinary and 64.1% with identical proof information as text. |
| Small student training | Executed a 539,437-parameter causal student on fixed verified data, with three seeds and exactly matched label-permutation controls. No held-out improvement; this is a restricted from-scratch student, not general-model distillation. |
| Persistent recipient learning | Tiny causal model and local pinned pretrained-model adapters; weights and AdamW state persist through iterations. Language-only and translated-state training modes have matched controls. |
| Bounded feedback loop | Executable two-cycle workflow with a frozen independent evaluation set kept out of curriculum feedback. Mechanism execution does not establish sustained capability improvement. |

The implication generator always chooses a reachable target. Its useful answer is a rule chain. Binary reachable/not-reachable accuracy on these records would be trivial and must not be used as evidence of reasoning.

## Run the constructive learning loop

Install the pinned Lean toolchain, then use the optional learning environment:

```bash
python -m pip install -e '.[llm]'
python experiments/run_constructive_loop.py --output runs/my-loop --experience-mode translated --iterations 2 --steps 2
python experiments/verify_constructive_run.py runs/my-loop
```

This default is a small CPU mechanism test. The [full protocol](docs/constructive-loop.md) provides the cached-pretrained-model command, five training arms, fixed evaluation format, artifact layout, and scientific limits. No checkpoint download occurs implicitly.

## Run the formal checker

Install Lean using the [official instructions](https://lean-lang.org/install/). The core toy has no Mathlib dependency. `lean/lean-toolchain` pins Lean 4.35.0. From the repository root, generate data, then run:

```bash
cd lean
lean --version
python ../experiments/check_lean.py ../data/generated/train.jsonl --output ../results/train.checked.jsonl --log-dir ../results/lean_logs/train
```

Repeat for `validation.jsonl` and `test.jsonl`. A missing Lean executable returns exit code 2 and `not_run`; a failed proof or axiom audit returns exit code 1. Only compiler success **and** recognized zero-axiom output can produce `verified`. See [Lean details](lean/README.md). The separate activation collector uses `val.jsonl`; its inputs are a deliberately prepared view, not an automatic synonym for the formal generator's filenames.

## The translator

For paired source and recipient activations, fit `T(x) = W x + b` using training data, select regularization on validation data, then evaluate once on test data. The models can have different hidden dimensions. Matching pairs must express corresponding content at specified layers and positions.

The [actual-model milestone](docs/llm-experiment.md) starts with related Qwen mathematical/general models and identical input text. It separates four outcomes:

1. Low representation error.
2. Improved answers when translated activations are supplied.
3. Improved answers after training, without the mathematical teacher present.
4. Repeated improvements from an adaptive generation loop at a fixed budget.

Each requires separate evidence. The repository includes artificial-vector checks, actual frozen-model interventions, a fixed-student experiment, and the new bounded constructive learning-loop implementation.

## First reproducible results

The bundled [run record](docs/validation.md) contains measured CPU results and verification limits. For seed 7, the artificial paired map's normalized held-out RMSE was about 0.00970, compared with 1.068 for shuffled anchors and 1.061 for predicting the training mean. These are planted-factor sanity checks, not mathematical or LLM capability results.

A [follow-up reproduction](artifacts/lf-001/reproduction.json) retained both predefined seeds, 7 and 19, and matched the bundled alignment results within floating-point tolerance; all 23 Python tests passed. The separate [first real Lean verification](docs/lf-002-lean-verification.md) checked 20 examples per split, retained compiler logs, and confirmed corrupted-input and missing-executable controls.

The [hosted CI reproduction](docs/ci-reproduction.md) passed on clean Linux checkouts after fixing the missing Lake manifest. A [15-example correspondence audit](docs/example-audit.html) records five examples per split; coding-agent review passed, and Ian confirmed agreement on all 15 in the [human review record](results/lf-004/human-review.json).

In the [actual-model experiment](docs/validation.md#first-actual-model-experiment), the affine map aligned representations but did not improve answer accuracy. The recipient benefited more from the supplied proof as ordinary text. The same fixed dataset also failed to improve a small student's held-out accuracy over a label-permutation control. These are bounded negative results, with commands, controls, hashes, and per-example outputs retained.

## Research map and collaboration

The [annotated source map](docs/sources.md) covers theorem generation, proof optimization, model stitching, actual transfer results, and implementation tools. The [backlog](docs/issues.md) suggests tasks for a research lead and an implementation lead. Contributors can start with one small pull request. Negative results with clean controls are welcome.

The static report is already built. To rebuild it after editing Markdown, use Node 20 or newer, run `npm install`, then `npm run docs`. Node is optional and is not required for the Python/Lean experiments.

## License and provenance

Original starter code and documentation use Apache-2.0; see [LICENSE](LICENSE). No third-party model weights or training corpora are bundled. Their licenses remain separate. The original private conversations and scanned concept sketch are deliberately absent from this public-ready package. Check the [provenance policy](docs/provenance.md) before adding external data or code.
