# Contributing

Start with one small experiment from [the issue backlog](docs/issues.md). State the question, baseline, dataset split, seed, compute budget, and acceptance criterion before running it. A careful negative result is a useful contribution.

## Development

Use Python 3.10 or newer; Python 3.12 was used for the supplied CPU runs. Install with `python -m pip install -e .`. Run `python -m unittest discover -s tests -v`. The NumPy-only experiments do not require an LLM or GPU. Lean verification is a separate mandatory gate before calling an item Lean-verified.

Use a branch such as `experiment/shuffled-anchors`. Keep each pull request focused on one question. Include a command that reproduces the result and a small JSON result file. Avoid committing model weights, activation dumps, credentials, private conversations, or large generated datasets.

## Research standards

- Distinguish artificial-vector tests, Python certificate checks, Lean-checked proofs, representation alignment, assisted inference, and standalone learned capability.
- Never treat a compiler exit code alone as proof completeness. Retain the axiom audit and reject incomplete proofs.
- Keep train, validation, and test examples separate by theorem or graph lineage, not only paraphrase.
- Do not choose layers, regularization, or prompts based on test performance.
- Publish wrong-pair, no-translation, and text-input controls alongside the proposed method.
- Record upstream model/data licenses. The project's license does not relicense dependencies.
- Do not label this starter an AGI or ASI system. Broader ambitions belong in hypotheses, with explicit tests.

## Pull requests

Describe the concrete question, what changed, exact reproduction command, observed result, and limitations. Note whether Lean and any actual language model were run. Ask a second contributor to inspect one generated example manually and rerun the principal control before merging an experimental claim.
