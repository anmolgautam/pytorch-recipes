# PyTorch Learning Curriculum

This document is the teaching roadmap for this repo.

It is meant to be read alongside [`instructor.md`](/Users/anmolgautam/Documents/learning/pytorch-learning-from-scratch/instructor.md), which defines the teaching style, standards, and session format.

## North Star

The goal is not just to "learn PyTorch."

The goal is to become someone who can:

- reason clearly about tensors, shapes, gradients, and devices
- write custom training loops with confidence
- train models correctly on CPU, single GPU, and multiple GPUs
- debug training failures and shape bugs without guessing
- understand performance, memory, numerical stability, and throughput
- grow into lower-level optimization topics such as kernels and Triton
- eventually build and train small transformer models from scratch

This should stay aligned with the north star in [`instructor.md`](/Users/anmolgautam/Documents/learning/pytorch-learning-from-scratch/instructor.md): build, train, debug, scale, and optimize deep learning systems from first principles.

## Curriculum Shape

The curriculum is intentionally staged:

1. Tensor fluency
2. Autograd fluency
3. Model-building fluency
4. Training-loop fluency
5. Transformer fluency
6. Systems and optimization fluency

Each stage is meant to support the next one.

## Stage 1: Tensor Fluency

Goal: become comfortable thinking in tensors instead of just writing tensor code.

Topics:

- scalars, vectors, matrices, and higher-dimensional tensors
- shapes, `ndim`, `numel`, `dtype`, and `device`
- indexing and slicing
- broadcasting
- `reshape`
- `squeeze` and `unsqueeze`
- `flatten`
- `view`
- `transpose` and `permute`
- contiguity and why `.contiguous()` appears
- `stack`, `cat`, `expand`, and `repeat`
- common shape and device errors

Important shape-manipulation cluster:

- `reshape` changes the requested shape while preserving the number of elements
- `view` is reshape-like but depends more directly on memory layout
- `permute` reorders axes
- `transpose` is a special case of axis swapping
- contiguity explains why some tensors can be viewed directly and others cannot

Why this matters:

If tensor layout and shape reasoning are shaky, everything later feels mysterious.

## Stage 2: Autograd And Optimization

Goal: understand how learning happens numerically.

Topics:

- what gradients mean
- `requires_grad`
- computation graphs
- `backward`
- gradient accumulation
- `torch.no_grad()`
- manual gradient descent
- loss functions
- optimizers such as SGD and AdamW
- gradient clipping
- learning-rate scheduling

Practice direction:

- fit a line with raw tensors
- inspect gradients manually
- move from manual updates to optimizer-based updates

## Stage 3: Building Models With PyTorch

Goal: become fluent with PyTorch's model-building conventions.

Topics:

- `nn.Module`
- parameters and buffers
- linear layers
- nonlinearities
- normalization basics
- embeddings
- loss modules
- model mode switching with `train()` and `eval()`
- saving and loading model state

Practice direction:

- build small MLPs
- inspect parameter shapes
- run forward passes and sanity checks

## Stage 4: Custom Training Loops

Goal: write training code from scratch with confidence.

At this stage, the loops can start with tiny in-memory tensors and toy datasets.
The full `Dataset` and `DataLoader` pipeline is formalized in Stage 5.

Topics:

- forward pass
- loss computation
- backward pass
- optimizer step
- gradient zeroing
- training vs evaluation loops
- metrics such as loss curves, accuracy, and later perplexity for language modeling
- checkpointing
- logging
- validation discipline
- debugging unstable runs

Practice direction:

- write a clean loop without trainer abstractions
- train on simple datasets first
- debug shape and optimization issues on purpose

## Stage 5: Data Pipelines

Goal: make data loading and batching feel normal, not magical.

Topics:

- datasets
- dataloaders
- batching
- shuffling
- transforms
- collate functions
- padding and masking basics
- throughput-aware input pipelines

## Stage 6: Decoder-Only Transformers First

Goal: build toward a real small language-model training loop.

Topics:

- tokenization basics and tokenization choices, including character-level vs subword tokenization
- why practical language modeling usually moves toward BPE-style tokenization
- choose a simple tokenizer first, then later move toward a more realistic subword setup
- embeddings and positional information
- causal masking
- self-attention
- multi-head attention
- MLP blocks
- residual connections
- layer normalization
- decoder-only transformer blocks
- next-token prediction training loop

Target:

- train a small decoder-only model on a manageable dataset such as TinyStories

Practical note:

- TinyStories can support a clean first language-model pipeline
- we should be explicit about the tokenization choice when we reach it rather than treating tokens as if they appear magically

Reference style:

- educational repos like Karpathy's can be used for orientation
- the aim is to understand and rebuild, not copy blindly

## Stage 7: Encoder-Decoder Models

Goal: extend transformer understanding beyond decoder-only models.

Topics:

- encoder vs decoder roles
- cross-attention
- sequence-to-sequence training structure
- teacher forcing
- masks for source and target sides
- inference differences

## Stage 8: GPU And Distributed Training

Goal: become comfortable moving from local experiments to serious training setups.

Topics:

- CPU vs GPU execution
- Apple `mps` and CUDA mental models
- mixed precision
- memory pressure and batch sizing
- gradient accumulation
- data parallel training
- distributed data parallel basics
- effective batch size
- checkpointing at scale

## Stage 9: Performance And Systems Depth

Goal: understand why some PyTorch code is elegant but slow, and how to improve it.

Topics:

- profiling
- memory layout awareness
- kernel launch overhead
- fusion ideas
- compilation and graph capture concepts
- custom kernels as a later topic
- Triton as a practical optimization tool

Important note:

We do not rush here. Systems depth makes far more sense after strong fluency with tensors, autograd, and training loops.

## Lesson-Level Workflow

Each lesson should usually create:

- one notebook for the live lesson
- one notes file for proper understanding and revision
- one key-notes file for fast review

Naming convention:

- `lessons/lesson_XX/lesson_XX_topic.ipynb`
- `lessons/lesson_XX/lesson_XX_topic_notes.md`
- `lessons/lesson_XX/lesson_XX_key_notes.md`

## Current Early-Lesson Sequence

The current shape-and-layout sequence should include:

1. tensors, shapes, indexing, slicing, broadcasting
2. `reshape`
3. `squeeze` and `unsqueeze`
4. `flatten`
5. `stack`, `cat`, `expand`, and `repeat`
6. `transpose` and `permute`
7. `view` vs `reshape`
8. contiguity and `.contiguous()`

This sequence is deliberate.

It builds from intuitive shape changes into memory-layout-aware tensor operations.
