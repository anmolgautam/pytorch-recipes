# Lesson 04: Key Notes

## Core Idea

- `flatten` merges a range of dimensions into one dimension

## Mental Check

- `squeeze` removes size-1 dimensions
- `unsqueeze` adds one size-1 dimension
- `flatten` merges dimensions together

## Main Shapes

- `[2, 3] -> flatten() -> [6]`
- `[2, 2, 2] -> flatten() -> [8]`
- `[2, 2, 2] -> flatten(start_dim=1) -> [2, 4]`
- `[2, 2, 3] -> flatten(start_dim=0, end_dim=1) -> [4, 3]`

## Important Distinction

- `flatten()` merges all dimensions, so result is 1D
- `flatten(start_dim=1)` keeps earlier dimensions and merges from dimension 1 onward

Very often:

- dimension `0` is batch
- flattening from `1` onward keeps batch intact

## Why It Matters

- `[2, 2, 2] -> flatten() -> [8]` loses explicit sample boundaries
- `[2, 2, 2] -> flatten(start_dim=1) -> [2, 4]` keeps batch and gives flattened features per sample

## Practical Question

Before using `flatten`, ask:

- what shape do I have now?
- what shape does the next operation expect?
- should batch stay separate?

## Readability Insight

- `reshape(4, 6)` says "make it this shape"
- `flatten(start_dim=1)` says "keep the first dimension, collapse the rest"

Both may work, but `flatten(start_dim=1)` often communicates model intent more clearly.
