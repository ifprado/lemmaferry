# Fixed evaluation fixtures

`independent-v1.jsonl` is a small, explicitly authored pipeline fixture. Its
arithmetic, ordering, string, and everyday-rule questions are separate from the
constructive theorem generator. It is frozen and hashed before any training.
Its questions and answers must never enter experience generation, curriculum
ranking, tokenizer fitting, or hyperparameter selection.

This fixture checks the evaluation boundary; it is not evidence of broad
reasoning ability. A research run should supply a larger, independently chosen
JSONL benchmark through `--evaluation`. Each row needs `id`, `prompt`, `answer`
(currently `A` or `B`), and `family`. Keep the benchmark fixed across treatment
and equal-budget control, and across all iterations.
