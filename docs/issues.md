# Actionable issue backlog

Copy the first three cards into GitHub issues after creating the repository. Estimates are focused work time for someone with the stated prerequisites, not deadlines. Use the experiment issue template for new research claims. The research lead and implementation lead are suggested roles, not fixed ownership.

## First three issue bodies

### LF-001: Reproduce both artificial alignment runs

**Owner:** Ian. **Estimate:** 45 to 90 minutes. **Dependencies:** none. **Labels:** good first issue, reproduction.

Run the test suite and `toy_alignment.py` with seeds 7 and 19. Save the two JSON outputs and optional plots. Compare paired-anchor, wrong-pair, and mean-output errors. Read the symmetric-cloud control. Write five sentences describing why it is a numerical sanity check and why geometric matching alone cannot establish semantic translation.

**Acceptance:** another person reruns both commands; normalized RMSE agrees to a sensible floating-point tolerance; no test-data preprocessing or best-seed selection; results explicitly say artificial vectors, no LLM or Lean involved.

### LF-002: Execute the first real Lean verification

**Owner:** cousin, with Ian reviewing examples. **Estimate:** 1 to 3 hours plus installation. **Dependencies:** none. **Labels:** good first issue, verification.

Install the pinned Lean toolchain using official instructions. Generate 20 examples per split. From `lean/`, check all three JSONL files, retaining logs and actual version. Inspect a short and a longer proof for the same theorem. Deliberately corrupt one copy and confirm failure. Read the existing tests for missing executables and misleading compiler success.

**Acceptance:** every accepted example has a recognized zero-axiom audit; corrupted input fails; missing Lean remains `not_run`; checked files and log locations appear in the experiment record. If the compiler rejects exported syntax, fix and retest the generator rather than changing statuses manually.

### LF-003: Publish the initial repository and reproduction instructions

**Owner:** Ian. **Estimate:** 45 to 90 minutes. **Dependencies:** review LF-001 and provenance policy. **Labels:** documentation, launch.

Create an empty public `lemmaferry` repository. Review the license, README, contents and staged diff, then push. Do not upload private notes, scans, credentials, activations, or weights. Invite the cousin after confirming their account. Open LF-001 and LF-002 and add one reproduction comment when they pass. Enable GitHub Pages from `/docs` if desired.

**Acceptance:** a fresh clone can run CPU tests; the README states actual verification limits; the first CI run is inspected; the project page has working links. No claim of Lean or model success is based only on a workflow file being present.

## Next issues

| ID | Concrete task | Suggested owner / effort | Depends on | Done when |
| --- | --- | --- | --- | --- |
| LF-004 | Audit five graph/proof/language triples from each split | Both, 1 hour | LF-002 | Each formal assumption and answer chain matches the rendered world; audit recorded |
| LF-005 | Add factorial structure, naming, and template splits | Implementation lead, 3 to 6 hours | LF-004 | Eight evaluation conditions isolate each shift; graph lineage stays within one split |
| LF-006 | Add normalized graph/statement hashes and deduplication | Implementation lead, 2 to 4 hours | LF-005 | Renamings and reordered-edge variants are grouped or rejected across splits |
| LF-007 | Write and check ten hand-authored Lean seed examples | Ian, 3 to 6 hours | LF-002 | Assumptions, proof terms, English views, and axiom audits reviewed |
| LF-008 | Define finite Horn non-entailment semantics and balanced tasks | Both, 3 to 6 hours | LF-004 | Negative labels mean not derivable in the finite rule world; never confused with a proof of negated free propositions |
| LF-009 | Add nonlinear planted-map capacity test | Ian or cousin, 2 to 4 hours | LF-001 | Affine and small MLP use identical splits; expected failure/success explained |
| LF-010 | Record a first actual-model experiment card | Both, 1 hour | LF-005, LF-007 | Model IDs/revisions, layer, position, tokens, controls and budget chosen before test |
| LF-011 | Prepare disjoint activation anchor views | Ian, 2 to 4 hours | LF-010 | IDs/families/text and stronger structural lineage checks pass; no answers in same-information prompts |
| LF-012 | Run 30-example activation collector pilot | Implementation lead, 2 to 4 hours plus download | LF-011 | Complete manifests, exact revisions, tokenizer hashes, timing and memory recorded |
| LF-013 | Collect larger anchors and fit affine baseline | Implementation lead, 3 to 6 hours | LF-012 | Validation selects alpha, test untouched, shuffled/mean controls and per-family metrics released |
| LF-014 | Add identity/oracle recipient intervention | Implementation lead, 3 to 8 hours | LF-013 | Recipient's own activation reproduces original predictions at the specified interface |
| LF-015 | Test translated-state intervention and text baseline | Both, 3 to 8 hours | LF-014 | Exact task results and equal-information/equal-budget controls, not MSE alone |
| LF-016 | Compare a small MLP against affine mapping | Implementation lead, 3 to 6 hours | LF-015 diagnosis | Capacity and training cost isolated; no test-set architecture selection |
| LF-017 | Prototype cross-attention bridge only if variable-length mismatch warrants it | Implementation lead, 1 to 2 weeks | LF-015, LF-016 | Same task/budget comparison shows whether complexity helps; evidence is not only training loss |
| LF-018 | Build fixed verified-text student dataset | Ian, 3 to 6 hours | LF-002, LF-006, LF-007 | Every accepted formal item checked; rendering audit and provenance intact |
| LF-019 | Run a small standalone student-training pilot | Implementation lead, 1 to 2 weeks | LF-018 | Student evaluated without teacher/translator; equal-token baseline and retention set included |
| LF-020 | Add paired uncertainty estimates over independent families | Both, 2 to 4 hours | Any task result | Confidence procedure respects dependence; three training seeds retained when applicable |
| LF-021 | Compare fixed and learning-progress curricula | Both, 1 to 2 weeks | LF-019 useful baseline | Same total candidates/updates/compute and frozen test; unsuccessful rounds reported |
| LF-022 | Compare proof costs with a fixed library | Ian, 2 to 4 hours | LF-002 | Edge applications, source tokens, checking time and search effort reported separately |
| LF-023 | Add a carefully scoped Mathlib domain | Both, 1 to 2 weeks | LF-007 and useful baseline | Toolchain/imports pinned; target theorem cannot be imported as its own solution |
| LF-024 | Write the first public methods/results note | Ian, 3 to 6 hours | LF-013 or later clean result | Claim limited to completed experiment; failures, controls, commands and revisions included |
| LF-025 | Reproduce a release from a fresh environment | Cousin, 2 to 4 hours | LF-024 | Clean clone commands work; citation, status table and results match one release commit |
| LF-026 | Explore one explicitly modeled physics task | Ian, later | A useful released baseline | Mathematical assumptions explicit; no biological inference claimed from formal correctness |

## Stop and simplify rules

- If LF-001 fails, do not troubleshoot large models yet.
- If LF-004 fails, repair the renderer or formal semantics before training.
- If LF-013 has low MSE but LF-015 gives no useful task improvement, publish that distinction and inspect the interface.
- If a plain-text baseline wins at lower cost, keep it as the recommended method and investigate why.
- If LF-019 learns familiar wording only, strengthen the evaluation instead of expanding training compute.
- If LF-021 does not beat the fixed curriculum, the generator may still be a useful benchmark; retain the negative result.

For every issue, attach the smallest artifact that lets the other collaborator check it: a JSON result, five inspected examples, a failing input, a shape trace, a plot, or a reproduction command.
