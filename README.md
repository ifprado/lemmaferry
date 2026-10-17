# LemmaFerry

**Verified mathematics, measurable transfer.**

LemmaFerry is an open research starter for generating formal reasoning tasks, fitting translators between representation spaces, and testing whether mathematical experience improves language-model reasoning.

The first question is deliberately small: **can a learned affine map carry useful structure between two representations, and can we tell that apart from a misleading geometric match?** The larger hypothesis is that an adaptive supply of verified mathematics can become useful training experience for a general model. This repository does not establish that hypothesis.

## Start here

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
| Affine ridge translator | Implemented and tested on artificial numeric vectors with planted shared factors. Includes training-only preprocessing and held-out data. |
| Misalignment controls | Shuffled-anchor and mean-output baselines; symmetric-cloud counterexample with correct geometry and wrong semantics. |
| Formal task generator | Implemented constructive propositional implication graphs. Short and longer valid rule chains share the same theorem. |
| Fictional-language renderer | Implemented from the same graph, with a checked controlled-format round trip. It is not a learned semantic translator. |
| Lean runner | Implemented timeouts, per-example statuses and zero-axiom audit. A [follow-up run](docs/lf-002-lean-verification.md) verified 60 generated examples and both proofs of each with Lean 4.35.0 and recognized zero-axiom audits. |
| Actual LLM collector | Optional script and instructions supplied. No model weights downloaded or real-model inference run during preparation. |
| Real-model affine fitter | Connects collected arrays to validation-selected ridge regression and controls. A numeric integration test does not substitute for a model run. |
| Activation intervention, student training, adaptive self-play | Roadmap experiments, not implemented claims. |

The implication generator always chooses a reachable target. Its useful answer is a rule chain. Binary reachable/not-reachable accuracy on these records would be trivial and must not be used as evidence of reasoning.

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

Each requires a separate experiment. The current starter demonstrates artificial-vector experiments and bounded formal-data verification.

## First reproducible results

The bundled [run record](docs/validation.md) contains measured CPU results and verification limits. For seed 7, the artificial paired map's normalized held-out RMSE was about 0.00970, compared with 1.068 for shuffled anchors and 1.061 for predicting the training mean. These are planted-factor sanity checks, not mathematical or LLM capability results.

A [follow-up reproduction](artifacts/lf-001/reproduction.json) retained both predefined seeds, 7 and 19, and matched the bundled alignment results within floating-point tolerance; all 23 Python tests passed. The separate [first real Lean verification](docs/lf-002-lean-verification.md) checked 20 examples per split, retained compiler logs, and confirmed corrupted-input and missing-executable controls.

## Research map and collaboration

The [annotated source map](docs/sources.md) covers theorem generation, proof optimization, model stitching, actual transfer results, and implementation tools. The [backlog](docs/issues.md) suggests tasks for a research lead and an implementation lead. Contributors can start with one small pull request. Negative results with clean controls are welcome.

The static report is already built. To rebuild it after editing Markdown, use Node 20 or newer, run `npm install`, then `npm run docs`. Node is optional and is not required for the Python/Lean experiments.

## License and provenance

Original starter code and documentation use Apache-2.0; see [LICENSE](LICENSE). No third-party model weights or training corpora are bundled. Their licenses remain separate. The original private conversations and scanned concept sketch are deliberately absent from this public-ready package. Check the [provenance policy](docs/provenance.md) before adding external data or code.
