# Lesson 04: Flatten Notes

## Lesson Goal

Understand what `flatten` really does, when it is useful, and why it is not the same thing as every other shape change.

## Core Idea

`flatten` combines dimensions into a single dimension.

That is the whole heart of the operation.

Sometimes that gives a 1D tensor.
Sometimes it keeps part of the shape unchanged and only merges a later chunk of dimensions.

So the clean rule is:

```text
flatten merges a chosen range of dimensions into one dimension
```

## Intuition

A tensor can have structure such as:

- rows and columns
- channels, height, and width
- batch and features

`flatten` is what we use when we want to collapse some of that structure into one axis.

Examples:

- `[2, 3] -> flatten() -> [6]`
- `[2, 2, 2] -> flatten() -> [8]`
- `[2, 2, 2] -> flatten(start_dim=1) -> [2, 4]`
- `[2, 2, 3] -> flatten(start_dim=0, end_dim=1) -> [4, 3]`

This is the lesson where it becomes important to see not just the final numbers, but which dimensions got merged.

## What To Watch For

- which dimensions are being combined
- whether batch should stay separate
- whether the next layer expects a vector or a structured tensor

That middle point is the one that matters most in actual model code.

## Common Confusions

### 1. `flatten` is not the same as `squeeze`

`squeeze` removes dimensions only when their size is 1.

`flatten` does something different:

it merges dimensions together.

### 2. `flatten` is not automatically "flatten everything in the whole world"

Sometimes we flatten all non-batch dimensions.

Sometimes we flatten only a range of dimensions.

The meaning depends on which dimensions we choose to collapse.

### 3. Same values, different meaning

Just like earlier lessons, the values stay the same.

What changes is the structure and therefore the meaning of the tensor for later operations.

### 4. `flatten` does not always mean "make it 1D"

This was the main correction in the lesson.

It is true that:

```text
flatten() -> usually 1D
```

But:

```text
flatten(start_dim=1)
```

does not make the tensor 1D.

It keeps the dimensions before `start_dim` unchanged and merges from that point onward.

So:

```text
[4, 3, 2] -> flatten(start_dim=1) -> [4, 6]
```

This is still a 2D tensor.

## `flatten()` vs `flatten(start_dim=1)`

This distinction is the real practical lesson.

### Plain `flatten()`

`flatten()` merges all dimensions into one.

Example:

```text
[2, 2, 2] -> [8]
```

That is often fine for small experiments, but in model code it is usually too aggressive because it destroys sample boundaries.

### `flatten(start_dim=1)`

This keeps dimension `0` unchanged and merges dimensions `1` onward.

Example:

```text
[2, 2, 2] -> [2, 4]
```

This is usually what we want for batched model inputs:

- dimension `0` stays batch
- the rest become flattened features per sample

## Why Batch Preservation Matters

If a tensor has shape:

```text
[batch, ...]
```

then plain `flatten()` merges the batch dimension with everything else.

Example:

```text
[2, 2, 2] -> flatten() -> [8]
```

Now the tensor no longer clearly represents:

- sample 1
- sample 2

That separation is gone.

But:

```text
[2, 2, 2] -> flatten(start_dim=1) -> [2, 4]
```

preserves the batch meaning:

- `2` = number of samples
- `4` = flattened features per sample

This is why `flatten(start_dim=1)` is much more common in neural network code.

## Relationship To `reshape`

`flatten` and `reshape` can sometimes produce the same result.

For example:

```text
[4, 3, 2] -> flatten(start_dim=1) -> [4, 6]
[4, 3, 2] -> reshape(4, 6) -> [4, 6]
```

So why use `flatten` at all?

Because the intent is clearer.

`reshape(4, 6)` says:

```text
make this tensor have shape [4, 6]
```

`flatten(start_dim=1)` says:

```text
keep the first dimension and collapse the rest
```

That second sentence often matches how we actually think about batched data.

## Why This Matters In Deep Learning

`flatten` shows up all the time when moving from:

- an image-like tensor into a linear layer
- a structured intermediate representation into a feature vector
- a higher-dimensional tensor into something easier to feed into loss or classifier code

Very often the pattern is:

```text
[batch, channels, height, width] -> [batch, features]
```

That is exactly the kind of job `flatten(start_dim=1)` is made for.

## Rules Of Thumb

- `flatten()` merges all dimensions into one
- `flatten(start_dim=1)` usually means keep batch, flatten the rest
- `flatten(start_dim=a, end_dim=b)` merges only dimensions `a` through `b`
- the values stay the same; the structure changes
- when batch exists, protect it unless you have a very specific reason not to

## Debugging Habit

Before flattening, ask:

1. what shape do I have now?
2. which dimensions should stay meaningful?
3. which dimensions should be merged?
4. what shape does the next operation expect?

This lesson especially sharpens one habit:

```text
Do I want one long vector, or one flattened vector per sample?
```

That question often tells you immediately whether you want:

- `flatten()`
- or `flatten(start_dim=1)`

## Lesson Takeaway

`flatten` is not just "make it simpler."

It is a deliberate shape operation:

- sometimes total collapse
- sometimes partial collapse
- often used to preserve batch and merge features

The best mental model is:

```text
flatten merges a range of dimensions into one
```
