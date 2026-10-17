# Field map and experiment-reading guide

Prepared for Ian Prado, October 16, 2026. Primary sources checked during this session. This is a research map, not a claim that the complete proposal is already implemented or that it yields general intelligence.

## Where this project sits

The proposal combines four established research areas: verified synthetic theorem generation, adaptive theorem-proving curricula, representation alignment between neural models, and distillation into a general language model. The integration question remains open: does a verified mathematical curriculum improve a general model, and does a learned vector interface beat simply supplying well-designed text?

Lean is a programming language and proof assistant. Its kernel checks proof terms against formal statements and assumptions. A neural encoder can turn those terms, statements, or proof states into vectors. Lean itself does not continuously generate neural embeddings or automatically add knowledge to another model's weights.

The most useful first contribution is a small, reproducible experiment distinguishing alignment quality, connected-system performance, and permanent student improvement. These are three different outcomes.

## Five nearest related projects

| Project | Closest part of Ian's proposal | Gap this project can investigate |
|---|---|---|
| STP | Conjecture, prove, select problems near current ability, retrain, repeat | Transfer beyond formal theorem proving and the benefit of a vector interface |
| Synthetic Theorem Generation in Lean | Generate theorem and proof together by forward inference | A small open reproduction with controlled task families and target-domain transfer |
| MUSTARD | Formal verification plus paired formal and informal training data | Lean 4 pipeline, controlled rendering, stronger out-of-family tests |
| Cache-to-Cache | Learn a direct internal interface between specialist and general language models | Verified theorem curriculum as the specialist's experience and separate distillation assessment |
| ProofOptimizer | Iteratively shorten verified proofs and reuse them for training | Whether shorter, clearer, or more diverse proofs produce more transferable learning |

These are related implementations to learn from and compare against. An exhaustive novelty determination would require a continuing literature review.

## Read these first, alongside the experiment

| Order | Read | Produce while reading |
| --- | --- | --- |
| 1 | Lean textbook logic chapters, source 15 | Ten explicit proofs and an axiom audit |
| 2 | Model stitching methods, source 9 | An affine-map experiment card and paired anchors |
| 3 | Forward theorem generation, source 2 | One graph/theorem/proof example traced by hand |
| 4 | ProofOptimizer, source 5 | A table separating proof length, checking time and search cost |
| 5 | Transferability study, source 11 | A held-out functional task and a retention set |

The remaining references are a map to consult when a milestone needs them. Completing all of them is not a prerequisite for publishing the first CPU experiment.

## Annotated primary-source reading list

License notes refer only to the component explicitly named. A repository's code license does not automatically license its models, datasets, or dependencies. A missing note means terms were not verified in this pass.

### Verified generation and self-improvement

**1. STP: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving, Kefan Dong and Tengyu Ma, ICML 2025.** [Paper](https://proceedings.mlr.press/v267/dong25h.html), [official code, models and data links](https://github.com/kfdong/STP). A conjecturer learns from problems barely solvable by its prover; verified proofs train subsequent rounds. Demonstrates stronger Lean and Isabelle theorem proving. It does not demonstrate general-world capability transfer. The Lean run generated 51.3 billion tokens. The repository's reproduction example requests 256 TPU-v4 cores and does not directly support GPUs. Borrow the curriculum design before reproducing its scale. Code MIT; bundled `levanter` Apache 2.0; model subject to the separate DeepSeek model agreement.

**2. Synthetic Theorem Generation in Lean, Joseph Rotella, Zhizhen Qin and colleagues, 2025 preprint.** [Author page and publication metadata](https://josephrotella.com/), [paper](https://josephrotella.com/files/synthm-lean.pdf). Uses forward reasoning from existing Lean proof states and assembles theorem and proof together, with optional proof minimization. Very close to correct-by-construction generation. Reported Falcon2-11B miniF2F improvement is 37.3% to 38.5% in one setting. It demonstrates theorem-proving utility, not broad transfer. No public implementation or reusable dataset license was verified from the consulted primary pages. Reimplement a tiny inference-rule subset rather than assuming code is available.

**3. MUSTARD: Mastering Uniform Synthesis of Theorem and Proof Data, Yinya Huang and colleagues, ICLR 2024.** [Paper](https://arxiv.org/abs/2402.08957), [official code and data](https://github.com/Eleanor-H/MUSTARD). Creates informal problems, informal proofs, formal statements and checked formal proofs. Its 5,866 validated examples improve theorem proving and math word problems in reported experiments. This is the closest clear text-data bridge. The current repository explicitly targets Lean 3.51.1, so treat it as a design reference for a new Lean 4 project. Do not assume its code or data has your chosen project's license; terms were not verified here.

**4. ImProver: Agent-Based Automated Proof Optimization, Riyaz Ahuja, Jeremy Avigad, Prasad Tetali and Sean Welleck, ICLR 2025.** [Conference paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4864005cfdea7ebd07086ed1b9846825-Abstract-Conference.html), [official code](https://github.com/riyazahuja/ImProver). Rewrites Lean proofs while maintaining correctness and optimizing user-defined criteria such as length or declarative structure. Useful for choosing a measurable proof-improvement objective. It does not imply that shorter proofs teach general intelligence better. Implementation license was not verified in this pass.

**5. ProofOptimizer: Training Language Models to Simplify Proofs without Human Demonstrations, Alex Gu and colleagues, October 17, 2025 preprint.** [Paper](https://arxiv.org/abs/2510.15700), [author project page](https://proof-optimizer.github.io/). Learns Lean proof simplification using expert iteration and reinforcement learning. Reports faster checking and better downstream prover training from simplified proofs. Strong evidence for the proof-optimization feedback component. It remains formal-math evidence. The project page was verified; a dedicated released code repository and its license were not verified. Separate proof discovery cost, checking cost, token length and readability in your own metrics.

**6. DeepSeek-Prover-V1.5, Huajian Xin and colleagues, 2024.** [Official implementation and paper](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5). Combines Lean feedback, supervised learning, reinforcement learning and search. Provides practical proof-attempt and checker integration patterns. It is a neural prover with an external formal checker, not a proof checker that learns truth by itself. Code MIT, model under separate model terms. It is a substantial specialist baseline; a small project need not train it from scratch.

**7. DeepSeek-Prover-V2, DeepSeek, 2025.** [Official repository, paper and models](https://github.com/deepseek-ai/DeepSeek-Prover-V2). Uses recursive subgoal decomposition, synthesized formal proofs and reinforcement learning. Demonstrates stronger formal mathematical reasoning. Recursive task decomposition here is different from demonstrated open-ended recursive improvement of the whole AI system. Read to understand how formal and informal reasoning can coexist. Models have a separate model license.

**8. Absolute Zero: Reinforced Self-play Reasoning with Zero Data, 2025.** [Paper](https://arxiv.org/abs/2505.03335), [official implementation](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner). A pretrained model proposes and solves executable code reasoning tasks, using execution as verification. Relevant to task generation and learning-progress curricula without human examples during this post-training phase. It does not start from a randomly initialized general language model, does not use Lean as its main verifier, and does not establish AGI. Implementation/model license terms were not verified here.

### Representation translation and transfer

**9. Transferring Linear Features Across Language Models With Model Stitching, Alan Chen, Jack Merullo, Alessandro Stolfo and Ellie Pavlick, NeurIPS 2025.** [Paper](https://arxiv.org/html/2506.06609v3), [conference record](https://proceedings.nips.cc/paper_files/paper/2025/hash/4569a868e7aa891248832ec08445d071-Abstract-Conference.html). Learns affine maps between model residual streams; transfers sparse autoencoders, probes and steering vectors. The central experiments use within-family pairs trained on the same data. This supports affine regression as a sensible baseline, not guaranteed transfer between arbitrary Lean and natural-language models. The paper reports that stitched downstream performance is bottlenecked by the weaker model. A dedicated official implementation license was not verified.

**10. Cache-to-Cache: Direct Semantic Communication Between Large Language Models, 2025 preprint, official repository identifies ICLR 2026.** [Paper](https://arxiv.org/abs/2510.03215), [official code and checkpoints](https://github.com/thu-nics/C2C). Projects and fuses attention caches, with gating to select useful layers. It demonstrates connected-model gains rather than permanent modification of a standalone receiver. Current repository supports differing dimensions, layers, heads and tokenizers and lists a Qwen3-0.6B receiver with a Qwen2.5-Math-1.5B specialist. This is a relevant later comparison to affine hidden-state translation. Repository Apache 2.0; constituent models have their own terms.

**11. Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning, Maggie Ziyu Huan and colleagues, ICML 2026.** [Final conference paper](https://proceedings.mlr.press/v306/huan26a.html), [official code](https://github.com/ReasoningTransfer/Transferability-of-LLM-Reasoning). Studies more than 20 models and controlled math-only post-training. Most observed math gains do not transfer broadly; controlled experiments identify on-policy training as important to transfer. Use it to design general-capability retention tests. The conclusion is empirical for studied setups, not a theorem that off-policy training always fails. License terms were not verified here.

**12. Learning from Synthetic Data Improves Multi-hop Reasoning, Anmol Kabra and colleagues, March 2, 2026; accepted to ICLR 2026 per author record.** [Paper](https://arxiv.org/abs/2603.02091). Rule-generated fictional knowledge improves multi-hop question answering after reinforcement learning. Relevant evidence that synthetic structure can teach transferable composition. The evaluation supplies relevant passages, so it does not show learning previously unknown real-world facts. It does not require Lean or vector translation. This is a useful simple alternative to compare with your formal-math pipeline. A dedicated official code/data link was not verified in this pass.

**13. Theorem Prover as a Judge for Synthetic Data Generation, Joshua Ong Jun Leang, Giwon Hong, Wenda Li and Shay B. Cohen, ACL 2025.** [Paper and official metadata](https://aclanthology.org/2025.acl-long.1448/). Uses iterative formalization and theorem-prover feedback to assess generated mathematical reasoning, improving several mathematical word-problem benchmarks. Relevant to filtering synthetic examples, but a formally valid translated statement need not faithfully express the original natural-language problem. Preserve that distinction in dataset status fields. A dedicated code/data license was not verified in this pass.

**14. Self-Play Pretraining with Zero Data, Aditya Cowsik and colleagues, September 24, 2026 preprint.** [Paper](https://arxiv.org/html/2609.30063v1). Randomly initialized program generator and learner adapt their synthetic curriculum; next-byte prediction improves on natural text and other modalities. The study is below 25 million parameters with 4K context. Natural validation data informs hyperparameter selection, although it is excluded from gradient updates. It uses executable programs, not Lean proof generation. Its most actionable lesson is that an adaptive curriculum beats merely having access to an unlimited program space. No official full reproduction repository was verified in this pass.

### Tools and prerequisite materials

**15. Theorem Proving in Lean 4, Jeremy Avigad, Leonardo de Moura, Soonho Kong and Sebastian Ullrich with the Lean community.** [Living textbook](https://leanprover.github.io/theorem_proving_in_lean4/). Start with propositions/proofs, tactics, induction and axioms/computation. Current online pages and an older model's Lean environment can use different versions. Pin a project's toolchain and dependencies; do not upgrade a reproduction merely because the textbook is newer.

**16. Mathematics in Lean, Lean community.** [Living textbook and exercises](https://leanprover-community.github.io/mathematics_in_lean/). Best bridge from Ian's mathematics background to practical Lean proof writing. Do the logic, algebra and induction exercises before attempting a general theorem generator. Existing mathematical knowledge lowers the conceptual burden; formalization syntax remains a separate skill.

**17. Lean REPL, Lean community.** [Code and JSON protocol](https://github.com/leanprover-community/repl). Programmatic command execution, proof states, diagnostics and `sorry` information. The simplest likely starting interface for a Python orchestration layer. Apache 2.0. The README demonstrates commands containing `sorry` for exploration; your accepted training records must reject incomplete results and audit proof dependencies. A successful process exit alone is insufficient.

**18. LeanDojo-v2, LeanDojo team.** [Current implementation](https://github.com/lean-dojo/LeanDojo-v2), [license file](https://github.com/lean-dojo/LeanDojo-v2/blob/main/LICENSE). End-to-end tracing, data management, retrieval agents, fine-tuning and evaluation. The original LeanDojo repository now points new projects to v2. More infrastructure than the minimal toy requires. The actual license file is Apache 2.0 despite an inconsistent README sentence saying MIT; use the license file and pinned commit. This is a reason to check actual files rather than inherit assumptions from a paper.

**19. mathlib4, Lean community.** [Repository](https://github.com/leanprover-community/mathlib4). Mathematical definitions, lemmas and tactics for Lean 4. Apache 2.0. Useful for later arithmetic, graph and algebra tasks; a core-Lean logic toy can avoid this dependency initially. Pin a revision and record imported theorem dependencies so the held-out theorem cannot be imported as an available lemma.

**20. PyTorch least squares.** [Official `torch.linalg.lstsq` documentation](https://docs.pytorch.org/docs/stable/generated/torch.linalg.lstsq.html). Implements the central affine-regression baseline. Add a constant column to inputs for an intercept, or center using training-only means. Use an SVD/QR-based least-squares solver or ridge solve instead of explicitly forming an inverse. This is a numerical tool, not evidence that an arbitrary pair of neural spaces is alignable.

**21. SciPy orthogonal Procrustes.** [Official reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orthogonal_procrustes.html). Finds the best orthogonal alignment for matched, same-shaped matrices. Useful to test whether rotation/reflection suffices before permitting scaling and shear. It does not itself include translation or scaling. Center using training data if appropriate and compare to full affine regression. Toy control: rotate random vectors with a known orthogonal matrix and recover it from anchors.

**22. Hugging Face Transformers model outputs.** [Official documentation](https://huggingface.co/docs/transformers/main_classes/output). Explains `output_hidden_states=True` and the batch-by-token-by-hidden-dimension shapes. Essential for defining exactly which vectors are translated. Distinguish token embeddings, layer activations, pooled representations, logits and KV caches. A pooled whole-proof vector is an experimental choice, not a standard universal representation of a theorem.

**23. Hugging Face PEFT LoRA guide.** [Official documentation](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora). A practical way to update a small fraction of a student model's parameters after creating checked text examples. This belongs to the permanent-learning experiment and should be separate from an adapter-only inference experiment. Use a pinned stable package version and identical data/step budgets across compared training conditions.

**24. Qwen2.5-0.5B model card, Qwen team, 2024 model family.** [Official base-model card](https://huggingface.co/Qwen/Qwen2.5-0.5B). A manageable open-weight starting point for inspecting activations or training a tiny specialist/student pair from the same initialization. Card lists Apache 2.0. It is a general pretrained model, not an already competent Lean prover; begin with small logic tasks. Pin model revision and tokenizer. Report actual memory use instead of guessing that an entire experiment fits on a laptop.

**25. Language Model Evaluation Harness, EleutherAI.** [Official code and usage](https://github.com/EleutherAI/lm-evaluation-harness). Standard evaluation runner, MIT. Useful later for a small fixed set of general-language and reasoning regression tests. The tiny generated benchmark should have its own exact verifier first. Record task version, prompt template, decoding parameters and sample budget. Benchmark datasets can have separate licenses and pretraining contamination.

## Five prerequisite blocks with concrete finish lines

These can be learned while building. Ian can own task semantics, Lean examples, experiment interpretation and documentation while his cousin owns model internals and training. Both should review each other's work.

1. **Lean proof terms and trust.** Read sources 15 and 16 on logic. Write ten core-Lean proofs using implication, conjunction and existential witnesses. Example: from `P -> Q`, `Q -> R` and `P`, construct `R`. Then deliberately replace a proof with `sorry` and make the pipeline reject it. Inspect `#print axioms` for accepted declarations. Finish when a script distinguishes checked, incomplete and failed attempts.
2. **Linear algebra and regression.** Use sources 20 and 21. Generate 1,000 random 16D vectors, transform with known `W` and `b`, fit on 800 and evaluate on 200. Repeat with noise, shuffled anchors and an unseen nonlinear transformation. Report held-out MSE and compare affine, orthogonal and mean-prediction baselines. Finish when Ian can explain what each failure means without referring to an LLM.
3. **Transformer activations.** Use sources 22 and 24. Extract one layer from ten inputs, document its exact shape and token positions, and reproduce it twice in evaluation mode. Verify that changing a premise changes activations. Compare a last-token vector with masked mean pooling. Finish with a small activation cache plus model, tokenizer, layer and pooling metadata.
4. **Training versus inference.** Use source 23. Freeze a student, train only an adapter, then verify the student's parameters have not changed. Separately train a tiny LoRA on checked text and compare parameters and held-out performance. Finish with two clearly named experiments: connected-system performance and standalone student learning.
5. **Experimental design.** Read sources 9, 11 and 14 selectively. Create train, validation and test splits by underlying task family or graph structure before rendering text. Generate three paraphrases of one structure and show they remain in the same split. Implement bootstrap intervals over independent problems and three training seeds. Finish with a frozen evaluation specification and an explicit unsuccessful-result interpretation.

## Evaluation controls that make the result informative

| Control | What it rules out |
|---|---|
| Student alone, same questions and decoding budget | Extra context or changed evaluation explaining the gain |
| Plain text theorem/premises and a deterministic verbal rendering | Translator only recreating information text already supplies cheaply |
| Same examples and training tokens, unverified generation condition | Whether proof checking adds measurable value at this scale |
| Random map, zero map, training-mean output, shuffled anchor pairs | Gains from generic perturbation, bias or accidental leakage |
| Source statement only versus source statement plus proof | Whether proof structure matters or the statement already gives the answer |
| Oracle target activation injection | Whether the selected target layer/interface can support the task at all |
| No-answer source features and explicitly labeled answer-visible oracle | Teacher representations secretly containing the held-out answer |
| Same-family aligned token experiment before cross-format proofs | Separate basic mapping failure from tokenization/semantic mismatch |
| Lexical baseline such as bag-of-tokens or fixed random projection | Apparent semantic alignment caused by shared syntax or token identity |
| Structure/family split, renaming and paraphrase kept together | Near-duplicate and template leakage |
| Fixed proof attempts, maximum tokens and wall-clock budgets | Pass@many or search compute being confused with learned ability |
| End-task exact correctness in addition to MSE/cosine/retrieval | Attractive vector alignment without useful computation |
| Frozen general-domain retention set after each training round | Narrow math gain that damages other capabilities |
| Three seeds, per-problem paired comparison and uncertainty | A noisy small-sample difference being advertised as progress |
| Keep old data replay versus new data only | Forgetting being confused with recursive improvement |

More proof bytes do not necessarily mean more learning. Track unique normalized statements, proof families, proof costs, student improvement per generated example and per unit of compute. An adaptive curriculum should beat a fixed generator at an equal budget before expanding it.

## Trust and semantic checks for the first dataset

- Pin Lean and all imports. Store the exact formal statement, assumptions, proof, source rules and dependency list, toolchain, checker diagnostics and generation seed.
- Reject incomplete proofs and any occurrence of `sorryAx` in the accepted theorem's dependency closure. For a constructive-only subset, use an explicit allowed-axiom policy and inspect dependencies; a proof accepted by Lean can use classical principles.
- Prevent generated declarations from introducing arbitrary new axioms. Recheck in a clean environment with allowlisted imports.
- Keep finite toy-world premises satisfiable by construction. A valid implication from contradictory assumptions is not useful evidence about the intended problem.
- Generate formal and natural-language views from the same small abstract syntax tree. Lean verifies the formal side; separately unit-check the renderer against that tree. Free-form autoformalization does not inherit correctness merely because its output compiles.
- If a source model sees a full solution at inference, call that condition a solution-informed teacher. Compare it to an equally solution-informed text teacher. Do not market it as the student solving a problem unaided.

## An appropriate future scientific claim after the evaluations

After actual Lean checking, recipient interventions, and student-alone evaluations have run, a bounded claim could be: "We release a reproducible benchmark and baselines for translating representations of Lean-verified synthetic reasoning tasks into a small language model. We measure alignment, exact downstream answers, and standalone student learning separately." The starter does not yet establish that result.

An improvement on this bounded experiment would justify the next experiment. A negative result with clean controls would still give other contributors a useful open-source artifact.
