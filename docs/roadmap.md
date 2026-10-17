# LemmaFerry: an actionable research roadmap

**Working name:** LemmaFerry. **Working question:** Can verified mathematical experience be translated into useful training signals for a general language model?

Prepared for Ian Prado and a prospective collaborator on October 16, 2026. The schedule assumes each person can contribute roughly four to six hours per week, an existing laptop, and no paid compute for the first milestone. These are planning assumptions, not requirements. The project can proceed more slowly without changing its scientific question.

## 1. What you are actually starting

You have a plausible research direction and several separable hypotheses. The credible first open-source contribution is a small benchmark, generator, translator baseline, and evaluation protocol that other people can reproduce. You do not need to claim that you can build AGI to publish that. Nor do you need to master statistical mechanics or topology before running an affine regression experiment.

The broad idea has close predecessors. The [source map](sources.md) explains where this project overlaps existing work. The contribution to investigate is the combination: correct-by-construction tasks, explicit representation translation, and tests that distinguish geometric alignment from useful reasoning transfer. Whether this combination adds value is an empirical question.

| Layer | What it does | First implementation |
| --- | --- | --- |
| Formal world | Defines objects, premises, inference rules, and target claims | Tiny propositional implication graphs |
| Generator | Constructs tasks and proof certificates | Deterministic Python generator, later a learned proposer |
| Checker | Checks that a formal proof proves the stated claim under stated assumptions | Lean, separate from the neural model |
| Semantic view | Expresses the same tiny world as a language task | Controlled renderer from the same graph |
| Representation translator | Maps source activations to a recipient model's chosen interface | Affine ridge regression, then a small nonlinear adapter if justified |
| Learner | Uses translated states or trains on generated examples | Small related open-weight models |
| Evaluator | Tests unfamiliar problems, costs, and general-capability retention | Fixed family splits and explicit controls |

### Keep these meanings separate

- A Lean proof is a formal object. An embedding of a proof is a learned representation of that object. Lean does not certify what an arbitrary embedding means.
- A neural prover proposes proof text or actions. Lean checks the formal result. They have different jobs.
- New proofs create new activation vectors when a model processes them. Those activations do not automatically become persistent knowledge.
- Moving vectors with a matrix is a representation change. Updating a model's parameters is learning. Supplying a teacher at inference is assistance.
- A generated theorem can be a new training example without being a mathematical discovery new to humanity.
- A Lean-accepted proof can use classical axioms. This starter's tiny implication fragment aims for no axiom dependencies; broader constructive restrictions require an explicit policy.
- Formal correctness establishes a consequence of premises. A natural-language description still needs a separate correspondence check, and an empirical premise still needs evidence from the world.

## 2. Your first 90 minutes

Do these in order. A domain purchase and an LLM download are not prerequisites.

- [ ] **Open the starter's README.** Read the status table before running anything. Know which parts were tested and which remain unrun.
- [ ] **Create a Python environment and run the tests.** Use Python 3.10 or newer. Follow the exact quick-start commands in the README.
- [ ] **Run the artificial alignment experiment with seed 7.** Open the output JSON. Find the paired, wrong-pair, and mean-output errors.
- [ ] **Change the seed to 19.** Explain why the wrong-pair task accuracy changes while its regression error remains poor. Do not select the nicer seed for a public claim.
- [ ] **Read the ambiguity counterexample.** A half-turn of a symmetric point cloud has excellent geometric matching and the wrong semantic labels. This is why pairing and task evaluation matter.
- [ ] **Generate 20 implication problems per split.** Open one JSONL record and its Lean declaration. Trace its short proof by hand.
- [ ] **Create an empty public GitHub repository called `lemmaferry`.** Follow [launch.md](launch.md), review the staged files, and push the starter. The supplied package has not been pushed to your account.
- [ ] **Create three initial issues:** run Lean verification, reproduce both alignment seeds, and choose the first actual-model experiment. Invite your cousin after you confirm their GitHub username yourself.

Finish line: a repository another person can clone, a reproducible CPU result, one understood formal example, and an honest list of untested claims.

## 3. What this starter already gives you

| Included item | What you can do now | Limit |
| --- | --- | --- |
| `alignment.py` | Fit a train-standardized affine ridge map with different source/target dimensions | Artificial tests do not establish neural-model transfer |
| `toy_alignment.py` | Reproduce paired versus shuffled anchors and the symmetric-cloud counterexample | Shared factors are deliberately planted |
| `logic.py` and `generate_logic.py` | Generate graphs, rule-chain answers, Lean proof terms, and controlled language views | Small fixed grammar; every selected target is reachable |
| `check_lean.py` | Run Lean with timeouts and axiom audits; keep per-example results | Lean was absent during preparation, so generated proofs are initially `not_run` |
| `collect_activations.py` | Collect actual-model activations when you install optional dependencies and download chosen checkpoints | Model inference has not been executed in the supplied environment |
| `fit_activations.py` | Fit a validation-selected affine map to collected arrays and compare controls | It evaluates representation error, not downstream model capability |
| Documentation and templates | Publish the project, divide work, register experiments, and record provenance | The future experiments still need to be implemented and evaluated |

See [validation.md](validation.md) for the exact completed checks. No model weights or large source corpora are bundled. The private July conversation and scanned sketch are not in the public-ready repository.

## 4. The milestones and the evidence each requires

### Milestone A: establish the mechanics

**Question:** Does the affine translator implementation recover a known relationship on held-out numeric vectors?

- [ ] Run seeds 7 and 19 with the default settings.
- [ ] Run with noise zero, then increase noise to 0.1.
- [ ] Compare paired anchors, shuffled anchors, and training-mean prediction.
- [ ] Confirm that normalization uses training rows only.
- [ ] Draw the optional plot and commit the small result JSON plus the command.

**Done when:** another person reproduces the numbers within documented numerical tolerance. A default paired normalized RMSE below 0.05 is an engineering sanity check for this planted toy, not a research success threshold for real models. If the toy fails, fix the implementation before introducing model complexity.

### Milestone B: generate genuinely checked formal data

**Question:** Can the pipeline create tasks, formal proofs, and matching language views while keeping their verification status accurate?

- [ ] Install the pinned Lean toolchain through the official instructions.
- [ ] Check the generated train, validation, and test files with the provided runner.
- [ ] Keep compiler version, diagnostics, and zero-axiom output with each accepted record.
- [ ] Deliberately corrupt one proof. Confirm it fails and cannot retain an old success status.
- [ ] Deliberately insert an incomplete proof. Confirm it is rejected even if a compiler process exits successfully.
- [ ] Have both collaborators inspect at least five examples from each split.

**Done when:** the intended accepted sample passes Lean and its axiom audit, altered examples fail, and the renderer preserves the same graph. If Lean has not run, keep the `not_run` status. A Python certificate check is useful and does not replace this gate.

### Milestone C: align actual-model representations

**Question:** Can an affine map predict a recipient model's activations on unseen task families better than simple controls?

- [ ] Follow [llm-experiment.md](llm-experiment.md).
- [ ] Begin with identical short input prompts for both models. This isolates coordinate alignment from language translation.
- [ ] Record the exact model commits, tokenizer hashes, layers, positions, input hashes, and packages.
- [ ] Debug on about 30 examples before collecting thousands.
- [ ] Fit regularization on validation data only. Keep the test set untouched until the choice is fixed.
- [ ] Compare shuffled anchors and training-mean outputs; add an identity map when dimensions match.

**Done when:** a fresh collection and fitting run produces a complete manifest and a clear held-out result, positive or negative. Do not require a positive result to publish. This milestone establishes representation alignment only.

### Milestone D: establish functional transfer

**Question:** Does the translated representation help the recipient answer questions, beyond what text or simple controls provide?

- [ ] Choose a single insertion point and document whether the state is before or after normalization.
- [ ] Implement an identity/oracle intervention using the recipient's own states. It should preserve the recipient's predictions within numerical tolerance.
- [ ] Implement the translated-source intervention at that exact interface.
- [ ] Test the recipient alone, the source alone, plain-text communication, translated states, shuffled states, and zero/mean states.
- [ ] Keep input information and decoding budgets matched. If the source sees a solution, compare with an equally solution-informed text teacher and label that condition.
- [ ] Evaluate exact rule-chain correctness or next-token loss on held-out problem families, not only embedding distance.

**Done when:** the intervention is reproducible and its result is interpreted narrowly. A clean functional failure despite good alignment is an informative result. Increase adapter complexity only if the failure points to a specific missing capability.

### Milestone E: teach a standalone student

**Question:** Does training on checked synthetic examples improve the student when the mathematical teacher and translator are absent at evaluation?

- [ ] Create a small fixed text dataset from the formally checked tasks and audited renderings.
- [ ] Fine-tune a small student with a modest parameter-efficient method or an explicitly bounded full-tuning run.
- [ ] Compare equal training-token/step budgets: controlled renderer data, learned-translator data if available, and a simpler nonformal synthetic baseline.
- [ ] Evaluate unseen structures and a small frozen general-language retention set.
- [ ] Repeat training with three seeds before advertising a capability gain.

**Done when:** the student is evaluated alone, with all controls and compute reported. A suggested later confirmatory criterion is at least five percentage points of held-out task improvement with a paired uncertainty interval excluding zero, chosen before the run. That is a proposed local criterion, not a field standard or a guaranteed achievable result.

### Milestone F: test adaptive improvement

**Question:** Does a curriculum adapting to the current learner outperform a fixed generator at the same total budget?

- [ ] Implement only after a useful standalone or connected-system baseline exists.
- [ ] Keep a fixed-generator control. Give both conditions the same total generated candidates, proof attempts, training tokens, and compute accounting.
- [ ] Score candidate usefulness using the current learner on training/validation tasks, never the held-out test set.
- [ ] Prefer tasks near the learner's current frontier while enforcing diversity and nontriviality.
- [ ] Retain some older data to measure and reduce forgetting; compare a replay condition explicitly.
- [ ] Evaluate after a fixed number of rounds and report unsuccessful candidates as part of the cost.

**Done when:** adaptation beats the fixed curriculum under the registered budget, or a reproducible failure reveals why it does not. More generated proofs alone do not demonstrate recursive capability improvement.

## 5. Concrete toy problems to work through

### Toy 1: change coordinates with known ground truth

Take the 2D point `(3, 1)`, rotate it 90 degrees to `(-1, 3)`, then add `(2, -1)` to obtain `(1, 2)`. Generate many points transformed by the same rule. Fit a matrix plus offset and predict held-out transformed points. Then use different input/output dimensions with a shared lower-dimensional factor, as the included code does.

**Exercise:** recover the transformation, add noise, shuffle the anchors, and compare errors. For an equal-dimension rigid case, compare the included orthogonal Procrustes baseline with the full affine map. Distances should be preserved only in the rigid case.

### Toy 2: perfect geometry with the wrong meaning

Place points equally around a circle. Label points by whether their horizontal coordinate is positive. Both identity and a half-turn map the unpaired cloud onto itself and undo themselves on a second application. The half-turn reverses the labels.

**Exercise:** reproduce the included counterexample, then add three known anchors and see which ambiguities remain. The lesson concerns correspondence, not whether unsupervised alignment can ever work.

### Toy 3: a theorem with two valid proofs

Use a starting proposition `P0` and rules `P0 -> P1`, `P1 -> P4`, `P0 -> P2`, `P2 -> P3`, `P3 -> P4`. The short route uses two rules; another valid route uses three. Formal proof terms apply the implication hypotheses to the starting fact.

**Exercise:** inspect the generated equivalent in `data/generated/train.jsonl`. Reconstruct the short chain, then the long chain. Lean should validate both under the same assumptions. Shortening here is shortest-path search over a fixed supplied rule graph. It is not a general algorithm for finding globally shortest mathematical proofs.

### Toy 4: render the exact same graph as a fictional task

Render the previous example as processes becoming ready: process Atlas is ready; if Atlas is ready then Birch is ready; if Birch is ready then Cedar is ready. Ask for a valid shortest rule chain establishing the target. Store both rule IDs and the atom chain as ground truth.

**Exercise:** rename every process, reorder rules, and add irrelevant rules. The answer's semantic structure should remain valid. The included templates have an exact round trip to the underlying graph; they do not establish that an arbitrary free-form explanation is faithful.

Every target in the initial generator is reachable. Asking only whether it is reachable would yield a trivial all-yes dataset. For a balanced future task, ask finite Horn entailment and generate non-entailment certificates under an explicit finite-world semantics. An unreachable graph node does not, by itself, prove the negation of a free proposition in Lean.

### Toy 5: separate the different generalization challenges

The initial splits change graph topology, naming vocabulary, and rendering template together. This is a useful stress test but does not isolate the cause of failure.

**Exercise:** make a factorial evaluation: familiar/unfamiliar structure crossed with familiar/unfamiliar naming and familiar/unfamiliar template. Keep each graph lineage within one split. This yields eight conditions. Report each condition instead of one blended accuracy.

### Toy 6: learn where affine translation fails

Generate source vectors and targets that include nonlinear functions such as a squared coordinate. Fit the affine map and observe the residual. Compare a two-layer MLP on exactly the same train/validation/test split. Change only the translator.

**Exercise:** distinguish insufficient model capacity from insufficient anchors. A nonlinear adapter winning this artificial task does not justify using a huge translator for real LLMs. It only demonstrates the intended capacity difference.

### Toy 7: write ten Lean exercises yourself

Prove implication application, implication composition, conjunction introduction, conjunction commutativity, projection from a conjunction, disjunction introduction, disjunction case analysis, an explicit existential witness, a simple natural-number identity, and an induction statement. Record the axiom dependencies.

**Exercise:** for each one, write a plain-English statement that preserves quantifiers and assumptions. Change one quantifier or reverse an implication and explain why the statement is different. These examples can become a manually inspected seed set for future translation.

### Toy 8: inspect actual activations without training a model

Give related general and mathematical models the same short prompts ending before the answer. Extract a fixed hidden-state layer and the last nonpadding position. Fit an affine map. The optional collector and fitter support this stage once the model dependencies and checkpoints are available.

**Exercise:** compare a fixed middle layer with the final representation using validation data, then freeze the choice. Changing layers based on test performance would contaminate the evaluation. Do not insert an intermediate residual vector into an input-embedding interface as if they were interchangeable.

### Toy 9: test whether the recipient can use a translated state

Begin with the recipient's own state as an oracle. For a simple next-token decoding test, a final normalized state may be supplied to the matching output head after inspecting the model implementation. For an intermediate-layer test, implement a proper forward intervention at that layer so remaining layers run normally.

**Exercise:** first show that the oracle path reproduces the original output. Then replace it with the mapped source state. A low MSE and a bad task result are compatible. Record both.

### Toy 10: a tiny learning-progress curriculum

After establishing a working learner, create tasks at several rule-chain depths. Compare uniform sampling against sampling from depth ranges where the learner is improving. Reject duplicate normalized tasks. Use an independent validation pool for selection and a frozen test pool for reporting.

**Exercise:** use the same number of learner updates and examples in both conditions. Add generation, checking, and teacher inference costs to the comparison. This is the first bounded test of the adaptive part of your original proposal.

## 6. An eight-week plan with dependencies

These are effort-sized phases, not deadlines. If a phase takes three weeks, move the later phases rather than skipping its checks.

| Phase | Ian's concrete work | Cousin's concrete work | Reviewable deliverable |
| --- | --- | --- | --- |
| First evening | Run CPU toys; create repository; choose first issues | Clone and reproduce one result | Public starter plus reproduction comment |
| Week 1 | Trace ten graph/proof pairs; learn basic Lean terms | Verify test harness and continuous integration | Lean audit results and deliberately failing example |
| Week 2 | Specify task semantics and held-out graph families | Add factorial split/renderer controls | Dataset card, 100 audited examples, frozen evaluation |
| Week 3 | Read model-stitching methods and write experiment card | Collect a 30-example activation pilot, measure memory/time | Exact model revisions and complete pilot manifest |
| Week 4 | Inspect errors and competing explanations | Fit affine map and controls on a larger anchor set | Held-out alignment report, including failed cases |
| Week 5 | Choose a functional task and plain-text baseline | Implement oracle and translated-state intervention | Connected-system comparison at matched information |
| Week 6 | Curate a small verified text dataset | Run one small student-training pilot if earlier gates justify it | Student-alone evaluation and retention results |
| Week 7 | Design fixed versus adaptive curriculum comparison | Implement a bounded curriculum if a useful baseline exists | Registered comparison and cost accounting |
| Week 8 | Write a short methods/results/limitations note | Reproduce from a fresh environment and package release | Versioned v0.1 research result, positive or negative |

Have a weekly 30- to 45-minute review. Each person brings one artifact, one uncertainty, and one proposed next experiment. Rotate who reproduces the other's result. Ian can lead the question, semantics, interpretation and documentation without pretending to be an expert in every implementation detail. His cousin can lead model internals and training while reviewing the mathematical claims together.

## 7. Learn only the prerequisites needed for the next gate

### Linear algebra and regression

- [ ] Derive the shape of `W` when the source has 12 coordinates and the target has 8.
- [ ] Explain why the intercept matters and why coordinate scales affect regularization.
- [ ] Fit using least squares or a ridge solver rather than an explicit matrix inverse.
- [ ] Compute the number of anchors relative to the input dimension and effective rank. A wide, underdetermined fit can interpolate training data without useful generalization.
- [ ] Explain the circle ambiguity and why determinant matching would not resolve semantics.

Finish when you can describe the planted toy and its controls from the output JSON without relying on a model's reassurance. Read the PyTorch/SciPy references in the source map as needed.

### Lean and formal semantics

- [ ] Work through the logic chapters of Theorem Proving in Lean 4 and the corresponding Mathematics in Lean exercises.
- [ ] Understand propositions as types and proofs as terms at the level needed for implication chains.
- [ ] Check actual dependencies with `#print axioms`.
- [ ] Keep assumptions, theorem statements, proof certificates, and language views distinct in the data schema.

Finish when you can intentionally break a statement, a proof, and a renderer separately and predict which check should catch each error.

### Transformer representations

- [ ] Identify the difference between token embeddings, residual-stream states, normalized states, logits, and KV caches.
- [ ] Print the shape of one actual model output and the token positions selected.
- [ ] Verify padding masks and deterministic evaluation mode on a small pilot.
- [ ] Document the recipient insertion point before training a translator for it.

Finish when you can trace one prompt through collection, mapping, and recipient evaluation with every shape and interface written down.

### Experimental inference

- [ ] Choose a primary outcome before running the test.
- [ ] Use validation for choices and hold out independent graph/theorem families.
- [ ] Keep paraphrases, variable renamings, alternative proofs, and numeric variants of one source within the same split.
- [ ] Report per-family results and paired differences. Bootstrap independent problem families rather than treating all paraphrases as independent samples.
- [ ] Repeat stochastic training seeds, retain failed runs, and avoid selecting the best seed.

Finish when a skeptical reader can distinguish leakage, extra context, extra compute, and useful transfer from your report.

## 8. Evaluation rules that prevent convincing yourself too early

### Before collecting data

Write the task, allowed inference information, generator grammar, split unit, selection rule, and budget into an experiment card. Version the generator and record a normalized structure hash or lineage identifier. Do not make a test theorem available as an imported library lemma while claiming the system discovered its proof.

### Before claiming translation

Use paired content anchors, selected layers, consistent token positions, and training-only preprocessing. Compare mean outputs, shuffled pairs, and a simple affine map before adding neural capacity. A small nonlinear MLP is the next capacity test. A cross-attention/Q-Former-style bridge is a later candidate for variable-length proof representations when simple mapping is demonstrably insufficient.

### Before claiming useful connected-model reasoning

Use end-task correctness and the same-information controls. Compare the mathematical model's raw text against its mapped representations. If the teacher read the complete proof, describe the result as proof-informed assistance. Account for the teacher's compute and the recipient's context length. An oracle with more information can define a ceiling, but it is not a fair baseline for an unaided model.

### Before claiming permanent learning

Evaluate the student without the source model or adapter. Compare matched training tokens, steps, and data counts. Keep a small frozen retention set so specialized improvement is not mistaken for broader improvement. Show example-level failures, not just an average score.

### Before claiming recursive improvement

Compare multiple adaptive rounds against a fixed generator with equal budget. Measure gain per checked example and per unit of total compute. Track unique normalized statements, proof families, attempted candidates, checking time, search time, training tokens, and replay. A larger data file or a shorter proof is not itself a capability improvement.

## 9. Compute, tools, and spending

The default is to keep the CPU/Lean phase free apart from existing hardware. A domain is optional. A standard text-only LLM API is useful for coding assistance or drafting explanations, but it generally does not expose the hidden states needed for your vector-regression experiment. Use locally inspectable open-weight models for that part.

Start the real-model stage with a small pilot and measure memory. Raw weights for a 1.5-billion-parameter model occupy about 6 GB in float32 or 3 GB in two-byte precision, before framework overhead and activations. Sequential collection avoids keeping both models in memory simultaneously. These are arithmetic estimates, not fit guarantees. The detailed plan includes activation-storage calculations and short-context defaults.

If the laptop is too slow, use a bounded one-off compute job after the pilot is specified. A suggested first discretionary cap is $25, chosen before launching anything; it is a budget decision, not a provider price estimate. Do not begin a large training run while debugging data semantics or padding. Record time and memory on the first batch and extrapolate cautiously.

Use Git, Python, NumPy, Lean, and optional PyTorch/Transformers. Prefer simple local files and JSONL over infrastructure. A vector database, web backend, and distributed training system are not needed for the first result. Full STP reproduction is a reference target with much larger compute requirements, not a sensible first milestone.

## 10. Open-source launch and collaboration

- [ ] Use the supplied Apache-2.0 license for original starter material and record external terms separately.
- [ ] Follow the launch commands in `launch.md`; the account name and remote URL are intentionally yours to choose.
- [ ] Keep the original private conversation and scanned notes outside the public repository.
- [ ] Add topics: `lean4`, `synthetic-data`, `representation-learning`, `theorem-proving`, `reproducible-research`.
- [ ] Add an honest short description: "Verified synthetic reasoning tasks and experiments in representation transfer."
- [ ] Enable continuous integration and inspect its first run. A workflow file being present does not mean it has passed on GitHub.
- [ ] Invite your cousin yourself and assign one small issue to each person.
- [ ] Enable GitHub Pages from `main` and `/docs` if you want the interactive roadmap as the project page.
- [ ] Publish the first experiment note only after another person reproduces it.
- [ ] Tag a release after commands, provenance, and results correspond to one commit. Use `CITATION.cff`; add a DOI only if you later actually obtain one.

Working domain: `lemmaferry.com`. A read-only official registry check found no matching record at October 16, 2026, 8:22 p.m. Pacific. This is not a reservation or a registrar checkout result; availability and price can change. GitHub Pages can host the project without buying a domain. See the timestamped observations in `launch.md`.

Useful first public contribution requests are concrete: reproduce an environment, inspect five proof/language pairs, improve a split, or implement one control. Avoid a broad request for someone to "help build AGI" before the toolkit has an independently checkable result.

## 11. Possible later connection to Light Dose

Keep the first benchmark domain-neutral so other people can understand and reproduce it. If the toolkit becomes useful, a later physics branch could generate and check small symbolic or finite-model exercises: dimensional consistency, matrix evolution in a specified kinetic model, probability identities, or energy accounting under explicit assumptions.

Such exercises could help audit theoretical derivations or teach a model to respect stated constraints. A formal proof would certify the mathematics inside the chosen model. It would not establish that the model describes biological effects or validate a radiation-health claim. The bridge back to Light Dose should come from a specific computational question, not from renaming a general benchmark.

## 12. Choose the next step from the result

| Observation | Next action |
| --- | --- |
| Artificial affine recovery fails | Fix numeric implementation, scaling, pairing or test generation |
| Python certificate passes but Lean fails | Inspect formal export and actual checker diagnostics |
| Controlled text view fails a round trip | Fix rendering before adding a language model |
| Real-model alignment is poor | Inspect layers, tokenization, rank, anchors and family shift before increasing capacity |
| MSE is low but answers do not improve | Investigate the recipient interface and lost task-relevant information; report the failure |
| Teacher-assisted answers improve | Compare plain-text assistance and total compute |
| Standalone student improves on familiar templates only | Strengthen structural splits and simplify the claim |
| Adaptive curriculum fails to beat a fixed one | Publish the comparison and diagnose diversity, difficulty and forgetting |
| Several results are reproducible | Release the benchmark and invite focused contributions before expanding scope |

Your next concrete deliverable is small: a public repository with a tested numeric translator, an accurately labeled formal-data pipeline, and one experiment another person can reproduce. Everything larger can be earned one result at a time.
