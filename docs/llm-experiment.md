# First experiment with actual language models

The first real-model milestone asks a limited question: can an affine map align
the activations of a small mathematical model and a small general model on
unseen theorem families? A later intervention experiment asks whether the
translated activations improve the recipient's predictions. These are distinct
claims. A low alignment error does not establish reasoning transfer, permanent
learning, recursive self-improvement, or AGI.

## Choose this pair first

| Role | Public checkpoint | Why use it | Published license |
|---|---|---|---|
| Source | [Qwen/Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) | Small, math-specialized, directly inspectable weights | [Apache 2.0](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B/blob/main/LICENSE) |
| Recipient | [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) | Related general base model with compatible architecture | [Apache 2.0](https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/LICENSE) |

The official configurations currently specify 28 layers and a 1,536-dimensional
hidden state for each checkpoint. They differ in positional encoding and
context limits. Use short prompts and record both configurations. These are
base completion models, so start with literal completion prompts rather than
chat templates. References:
[math configuration](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B/blob/main/config.json),
[general configuration](https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/config.json).

The math checkpoint is a mathematical language model, not a certified Lean
proof checker. Lean remains a separate executable verifier. This model pair
tests the representation interface; it does not yet instantiate a
Lean-trained, self-improving theorem prover.

Do not assume related names guarantee identical tokenization. The collector
records a full tokenizer fingerprint and token IDs for every input as hashes.
Require equal fingerprints and token hashes for the same-input baseline. If
they differ, inspect the actual tokenization or choose a different checkpoint
pair. Matching vocabularies alone would be insufficient.

## Checklist A: prepare a small, auditable dataset

- [ ] Run the repository's deterministic toy experiments before downloading any models.
- [ ] Create `data/anchors/train.jsonl`, `val.jsonl`, and `test.jsonl`.
- [ ] Give each example a stable `id`, `family`, and `text`.
- [ ] Start with 30 prompts to debug collection, then aim for about 3,000 anchors
  across several genuinely different families. A possible budget is 2,000 train,
  500 validation, and 500 test examples. These are proposed experiment sizes.
- [ ] Assign complete generator families to splits before rendering text. Keep
  all variable renamings, numeric variants, alternate proofs, paraphrases, and
  applications of one underlying theorem in the same split.
- [ ] Use at least two held-out families. Examples include implication chains,
  parity composition, finite-set membership, modular arithmetic, and finite
  graph reachability. Record the particular grammar used for each.
- [ ] Deduplicate normalized theorem/proof structures, not just literal strings.
- [ ] Store the original formal statement, proof, Lean toolchain version,
  verification result, source, and generator seed in the main dataset. The
  collector's compact JSONL is a view of that dataset.
- [ ] Select validation/test families once. Write down the decision and avoid
  inspecting test results while choosing layers, regularization, or prompts.

Each JSONL line is one JSON object:

```json
{"id":"train-implication-0001","family":"implication_chains","text":"Facts: every A is B; every B is C; x is A. Therefore x is"}
```

An example from another split needs a different family, for example
`parity_composition`. Numbers changed within one implication template do not
constitute a new family. The collector checks disjoint IDs, exact-text
duplicates across splits, and the declared family partition. It cannot
automatically detect disguised structural duplicates; the generator must
enforce that stronger rule.

For this milestone, both models see **identical prompt text**. This removes
formal-language translation and token-position alignment from the first test.
The selected prompt must end before the answer. Do not give the source the
correct solution while presenting the recipient only the question when
claiming same-information reasoning transfer.

## Checklist B: collect frozen-model activations

An optional collector is provided at `experiments/collect_activations.py`.
Its model inference path has not been executed in the starter package.
The initial workspace did not contain torch or transformers, and no weights
were downloaded. Treat the first run as a documented engineering milestone.

Create a separate environment so the toy package stays lightweight:

```bash
python3 -m venv .venv-llm
source .venv-llm/bin/activate
python -m pip install -e '.[llm]'
python -m pip freeze > llm-environment.lock.txt
```

For hardware-specific PyTorch installation, follow the
[official installation selector](https://pytorch.org/get-started/locally/).
The optional dependency group keeps this prototype on Transformers 4.x. The command above intentionally does not claim a tested dependency lock.
Commit the exact lock from a successful run together with the run's manifest.

Resolve and inspect immutable model revisions before collecting. This command
queries metadata rather than downloading weights:

```bash
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
for model in ["Qwen/Qwen2.5-Math-1.5B", "Qwen/Qwen2.5-1.5B"]:
    print(model, api.model_info(model).sha)
PY
```

Paste each 40-character commit SHA into its command. The collector rejects
mutable references such as `main`:

```bash
python experiments/collect_activations.py \
  --model Qwen/Qwen2.5-Math-1.5B \
  --revision PASTE_THE_MATH_COMMIT_SHA \
  --train data/anchors/train.jsonl \
  --val data/anchors/val.jsonl \
  --test data/anchors/test.jsonl \
  --output runs/llm-source-layer14 \
  --layer 14 --max-length 256 --batch-size 1 \
  --device cpu --dtype float32

python experiments/collect_activations.py \
  --model Qwen/Qwen2.5-1.5B \
  --revision PASTE_THE_GENERAL_COMMIT_SHA \
  --train data/anchors/train.jsonl \
  --val data/anchors/val.jsonl \
  --test data/anchors/test.jsonl \
  --output runs/llm-recipient-layer14 \
  --expected-tokenizer-sha PASTE_SHA_FROM_SOURCE_MANIFEST \
  --layer 14 --max-length 256 --batch-size 1 \
  --device cpu --dtype float32
```

Run the commands sequentially. Each process loads one checkpoint, finishes,
and exits before the other starts. CUDA and MPS can be requested explicitly;
try the CPU float32 baseline for a few examples first. Float16 on CUDA may
reduce memory use but changes the numerical experiment. MPS support and
bfloat16 support depend on the installed runtime and hardware.

The collector:

- Uses built-in Transformers model classes with `trust_remote_code=False` and
  safetensors weights.
- Loads tokenizer and model from the same immutable revision.
- Sets a fixed seed, `eval()` mode, and `torch.inference_mode()`.
- Records package versions, model configuration, input/output hashes, device,
  dtype, tokenizer fingerprint, layer, and selection rule.
- Supplies attention masks, pads on the right, and selects the last **nonpadding**
  token. It does not average padding tokens or assume the last array column is valid.
- Rejects overlong inputs rather than silently truncating them.
- Saves a status manifest; partially failed runs must not be treated as results.

For each split, the NPZ contains `vectors` of shape `[examples, hidden_size]`,
`ids`, `families`, `token_sha256`, and `lengths`. Arrays are non-pickle NumPy
types. Pair rows by IDs and assert identical input hashes and token hashes
before fitting. Do not pair rows merely by their current array order.

The layer argument is the literal index in the Transformers `hidden_states`
tuple. Index zero is the embedding output; subsequent indices identify model
boundaries. The final entry may include final normalization. Confirm the
precise boundary in the installed model implementation before writing hooks.
[Transformers documents these output conventions](https://huggingface.co/docs/transformers/en/main_classes/output).

## Checklist C: fit and evaluate the affine translator

Use the same regression idea as the toy alignment experiment:

\[
\min_{W,b}\sum_{i\in\mathrm{train}}\|x_iW+b-y_i\|_2^2
+\lambda\|W\|_F^2.
\]

This uses row-vector notation. Its matrix is transposed relative to the
column-vector expression `Wx + b`; the operation is the same.

After both collection manifests report `completed`, run the included fitter
from the repository root:

```bash
python experiments/fit_activations.py \
  --source runs/llm-source-layer14 \
  --target runs/llm-recipient-layer14 \
  --output runs/affine-layer14 \
  --alphas 0.0001 0.001 0.01 0.1 1.0 \
  --seed 1729
```

Choose a new output directory for each run. The fitter verifies archive hashes,
completed statuses, matching input/token hashes, and identical paired IDs in
the same order. It rejects reordered collections rather than silently pairing
their rows. It fits standardizers on training data, selects the ridge strength
using validation normalized RMSE, and then evaluates the selected map on test.
Its objective averages squared error in standardized coordinates, so the
numeric `alpha` values depend on that convention. It does not refit on the
validation set.

Read `runs/affine-layer14/results.json` for validation scores, test metrics,
per-family metrics, and shuffled/mean/identity controls. The raw-coordinate
map is saved in `translator.npz`; apply it as
`source_vectors @ coef.T + intercept`. PCA, an unrelated-input control, layer
selection, and activation interventions remain follow-up tasks. The fitter's
numeric integration tests use artificial activations; an actual checkpoint
collection is still required for model results.

- [ ] Fit source centering, target centering, any standardization, and PCA
  **only on the training split**. Transform validation/test with those fitted values.
- [ ] Begin with ridge regression. The 1,536 by 1,536 map has over two million
  parameters, so low training error with a few hundred anchors is uninformative.
- [ ] Optionally use a training-fitted PCA or low-rank map with 32 or 64 components.
  Fit ridge in the smaller space and reconstruct full recipient activations.
- [ ] Compare a small, predetermined validation grid for ridge strength and
  layers, such as indices 7, 14, and 21. Select one configuration before test.
- [ ] Include recipient-training-mean prediction, identity when dimensions
  match, a shuffled-pair map, and an unrelated-input control.
- [ ] Report test MSE, MSE divided by the recipient-mean baseline MSE, cosine
  similarity, and matched-example retrieval accuracy. State chance retrieval
  accuracy and candidate-pool size. For MSE, smaller is better; a ratio below one
  beats the mean baseline.
- [ ] Report per-family metrics so a majority template cannot hide failure.
- [ ] Publish unsuccessful runs and data sizes. A usable result includes the
  manifests and enough scripts to reproduce it, not just a scatter plot.

This experiment is motivated by
[Chen et al., Transferring Linear Features Across Language Models With Model Stitching](https://papers.neurips.cc/paper_files/paper/2025/hash/4569a868e7aa891248832ec08445d071-Abstract-Conference.html).
That work supports affine feature transfer between language models. It does
not establish that this proposed math-to-general interface will improve
general reasoning. The related checkpoints make alignment easier, so success
here should not be presented as evidence for arbitrary unrelated models.

## Checklist D: test an actual intervention

This part is a future implementation task. The collector does not replace
activations, run the recipient's decoder, or evaluate answers.

1. Load the recipient as `AutoModelForCausalLM`, keep its parameters frozen,
   and implement a hook at the exact boundary used during collection.
2. Feed both models the same held-out prompt. Translate the source's final
   prompt-token activation with the fitted affine map.
3. Replace only the recipient activation at that boundary and position.
   Continue through the remaining recipient layers and its output head.
4. Start by scoring the **next answer token**. This avoids claiming that a
   single-position intervention has implemented coherent multi-token generation.
   Check tokenization of answer labels; do not assume `A`, `B`, `0`, or `1`
   is a single token in every context.
5. Compare ordinary recipient, translated source, shuffled source, zero source,
   recipient mean, and identity-hook controls on identical examples. Include
   source-model performance as a reference, not as an automatically attainable
   upper bound. A self-patch must reproduce the ordinary recipient's logits
   within a declared numerical tolerance.
6. Measure held-out answer accuracy and answer-token negative log likelihood.
   Later, score full answers with teacher forcing while masking prompt and
   padding positions out of the loss. Report the exact intervention policy
   for each continuation token.
7. Bootstrap paired differences at the **underlying theorem or family level**,
   rather than treating paraphrases as independent observations. Use a fixed
   test set and confidence intervals. Report regressions as well as gains.

Before spending on a large run, verify that the source is better on the target
toy task or carries a complementary usable signal. Mapping its activations
toward the weaker recipient's existing activations can erase exactly the
information you hoped to transfer. If geometric alignment succeeds but
functional improvement fails, preserve that result and try a task-loss-trained
adapter as the next hypothesis.

When testing a source that receives an entire proof and a recipient that sees
only a question, also give a baseline recipient the proof as ordinary text.
Improvement in this setting might be evidence compression or communication
of extra context. It cannot by itself establish transfer from equal information.

## Later: formal-to-natural correspondence and permanent learning

Formal statements, proof steps, and ordinary-language explanations have
different token counts. A token-wise matrix does not solve this alignment
problem automatically. Begin with one statement-level summary activation on
each side or explicit AST/proof-step anchors. If that bottleneck loses needed
information, test variable-length cross-attention or a query-based adapter.
Keep every rendition of an underlying proof in the same split. Separate
semantic preservation checks from whether Lean accepts the original proof.

An inference-time interface leaves the recipient's weights unchanged. To
study permanent learning, create a verified dataset with the connected system,
train or distill a separate recipient on that data, remove the source and
adapter, then evaluate again. Compare against the same number of ordinary
proof-text examples and match training tokens, optimizer steps, and compute.

The first recursive experiment needs at least two complete cycles and fresh
evaluation families: mathematical-system improvement, new accepted examples,
recipient retraining, and measured improvement that helps the next cycle.
Additional stored vectors alone are not such a cycle.

## Resource planning and stop conditions

The following are calculations and planning assumptions, not measured runtimes:

| Item | Approximate scale or decision |
|---|---|
| One 1.5B checkpoint's weights | About 3 GB at two bytes per parameter, or 6 GB at four bytes, before runtime overhead |
| Loading strategy | One model per process, sequential collection; never require both resident initially |
| Machine target | A 16 to 24 GB machine may support short float32 runs sequentially; runtime and loading peaks must be measured on the actual machine |
| Stored last-token vectors | `N * 1536 * 4` bytes per model; 3,000 examples are about 18.4 MB before archive overhead |
| Raw all-token activations | Larger by roughly average token length; do not save them in the first experiment |
| CPU timing | Time 30 examples and extrapolate; publish the measured rate rather than guessing a GPU rental budget |
| First spending gate | Finish the CPU toy and 30-example collector checks before paid compute |
| Second spending gate | Beat the mean/shuffled alignment controls on held-out families before implementing a large adapter |

Retain existing model-license and attribution requirements if redistributing
weights; the repository code license does not relicense model checkpoints or
third-party datasets. Downloading public weights locally is sufficient for
this experiment; do not commit them to GitHub.

Fixed seeds improve reproducibility but do not guarantee identical results
across hardware or package releases. See
[PyTorch's reproducibility guidance](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/randomness.md).

Sources checked on 2026-10-17 UTC. Pin the exact model commits and runtime
versions used by the actual experiment.
