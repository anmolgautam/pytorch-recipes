# PyTorch Learning From Scratch

This repo is a focused learning experiment:

```text
Can an LLM acting as a personal instructor help accelerate real PyTorch understanding and practical deep learning engineering skill?
```

The goal is not passive reading.
The goal is rapid upskilling through structured, hands-on, notebook-based practice with an AI instructor that teaches one concept at a time.

## Why This Exists

This project is an experiment in deliberate technical learning.

The idea is to use an LLM not as a code generator to blindly copy from, but as:

- a personal instructor
- a patient debugger
- a conceptual guide
- a practice partner
- a note-making companion

The working hypothesis is that this can dramatically speed up early PyTorch learning if the process is structured properly.

## Learning Objective

The immediate objective is to build strong PyTorch foundations.

The larger objective is to become capable of:

- reading tensor shapes fluently
- understanding autograd and optimization
- writing custom training loops with confidence
- training models on CPU, single GPU, and later multiple GPUs
- understanding attention and transformer internals
- eventually building and training decoder-only and encoder-decoder models
- moving toward systems-level understanding such as profiling, performance, kernels, and Triton

## How This Repo Is Used

Each lesson produces two main artifacts:

1. One Jupyter notebook for the live lesson and experiments
2. One Markdown notes file for clean conceptual notes

There is also often a third lightweight artifact:

3. One key-notes cheat sheet for quick revision

Naming convention:

- `lessons/lesson_XX/lesson_XX_topic.ipynb`
- `lessons/lesson_XX/lesson_XX_topic_notes.md`
- `lessons/lesson_XX/lesson_XX_key_notes.md`

Examples:

- `lessons/lesson_01/lesson_01_tensors.ipynb`
- `lessons/lesson_01/lesson_01_tensors_notes.md`
- `lessons/lesson_01/lesson_01_key_notes.md`

## Learning Method

The teaching style in this repo is intentionally strict:

- one concept at a time
- one question at a time
- fundamentals before shortcuts
- intuition before notation
- code after prediction
- notes that capture understanding, not just output

The notebook is the working space.
The notes are the long-term study material.

## Current Direction

The path is aimed toward real deep learning engineering, not just beginner familiarity.

That means the learning arc is expected to move through:

- tensors and shape fluency
- reshape, squeeze, unsqueeze, and dimension control
- autograd
- custom training loops
- `nn.Module`
- data pipelines
- attention and transformers
- decoder-only model training
- encoder-decoder model training
- performance and systems topics

TinyStories and educational repos such as Karpathy's work may be used later as reference points for real training loops and small language-model experiments.

## Repo Files

- [`instructor.md`](./instructor.md): the teaching contract, style, and long-term roadmap
- [`lessons/`](./lessons): one folder per lesson
- lesson notebooks: the live practice sessions
- lesson notes: structured explanations, confusions, and clarifications
- key notes: compact revision sheets

## What Success Looks Like

This experiment is successful if it produces:

- fast conceptual progress
- strong fundamentals
- increasing independence in reading and writing PyTorch code
- confidence with training loops and debugging
- a visible path from beginner practice to real model training

## Status

This repo currently contains:

- Lesson 01: tensors
- Lesson 02: reshape

The next lessons will continue building tensor fluency before moving deeper into autograd and training.
