# Lesson 03: Key Notes

## Terminology

- `squeeze`: remove dimensions of size 1
- `unsqueeze(dim)`: insert one new dimension of size 1 at axis `dim`

## Core Shape Moves

- `[3, 1] -> squeeze() -> [3]`
- `[1, 3, 1] -> squeeze() -> [3]`
- `[3] -> unsqueeze(0) -> [1, 3]`
- `[3] -> unsqueeze(1) -> [3, 1]`
- `[1, 3] -> unsqueeze(-1) -> [1, 3, 1]`

## Mental Checks

- `squeeze` does not mean "make it 1D"
- it only removes size-1 dimensions
- `unsqueeze` adds exactly one new dimension
- axis position changes meaning

## Important Distinction

- `[1, 3]` often means one sample, three features
- `[3, 1]` often means three rows, one feature each

Same values, different shape, different behavior.

## Negative Index Reminder

- `-1` means last position
- `-2` means second-last position

## Common Pitfalls

- thinking `squeeze()` always flattens
- forgetting that `squeeze(dim)` does nothing if that dimension is not size 1
- forgetting that `unsqueeze` adds only one dimension
- mixing up `[1, n]` and `[n, 1]`

## Quick Rule

Use `unsqueeze` when the next operation needs an explicit batch or column dimension.
Use `squeeze` when a useless size-1 dimension is getting in the way.
