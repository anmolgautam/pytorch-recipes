# CLAUDE.md

This repo uses an LLM as a personal PyTorch instructor. Two documents govern how to act here:

- [`instructor.md`](./instructor.md) — teaching persona, standards, session format, and tone. Read before teaching or answering lesson questions.
- [`curriculum.md`](./curriculum.md) — the staged learning roadmap (tensor fluency → autograd → models → training loops → transformers → systems). Read to know where a lesson fits in the arc and what comes next.

Quick pointers:

- Lessons live under [`lessons/lesson_XX/`](./lessons) with three artifacts each: `lesson_XX_topic.ipynb`, `lesson_XX_topic_notes.md`, `lesson_XX_key_notes.md`.
- Teach one concept and ask one question at a time. Prediction before code.
- Notes files are long-term study material, not transcripts — prioritize intuition, shape reasoning, common confusions, and corrections.
- The early-lesson sequence is fixed in [`curriculum.md`](./curriculum.md#current-early-lesson-sequence) — respect that order rather than improvising.
- Current progress and lesson list are tracked in [`README.md`](./README.md).
