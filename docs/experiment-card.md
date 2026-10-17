# Experiment card

Copy this into an issue before running a new experiment.

| Field | Fill in before the run |
| --- | --- |
| ID and question | One specific claim |
| Hypothesis | Expected direction and meaningful effect |
| Inputs allowed at inference | Theorem, proof, question, source model, or no assistance |
| Ground truth | How the answer and formal meaning are checked |
| Training data | Origin, license, example count, family or lineage IDs |
| Validation/test split | What remains unseen and why |
| Method | Map, loss, layer, pooling, regularization, insertion point |
| Controls | Mean/random/shuffled/text/oracle, as applicable |
| Budget | Tokens, candidate attempts, wall time, hardware, memory |
| Selection rule | What can be tuned using validation only |
| Primary metric | Task correctness first for capability claims |
| Uncertainty | Seeds and paired bootstrap over independent problem families |
| Stop rule | Conditions to stop, simplify, or escalate |
| Release | Commands, exact revisions, outputs, unsuccessful runs |

Afterward record the observed result, any deviations, and the narrowest claim justified by it. A low embedding error by itself is not a functional transfer result. A proof supplied at inference is assistance, not evidence of knowledge retained without it.
