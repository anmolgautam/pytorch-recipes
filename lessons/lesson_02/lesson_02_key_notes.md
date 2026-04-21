# Lesson 02: Key Notes Cheat Sheet

## Core Terms

- `reshape`: change the shape without changing the values
- `numel()`: total number of elements in the tensor
- `-1` in `reshape`: let PyTorch infer one dimension automatically

## Main Rules

- `reshape` changes structure, not content
- the total number of elements must stay the same
- valid reshape means the new dimensions multiply to `numel()`
- `[6]` is 1D
- `[1, 6]` is 2D
- `[6, 1]` is 2D
- only one dimension can be `-1`

## Common Mistakes

- thinking `reshape` changes the values
- thinking `[6]` and `[1, 6]` are basically the same
- using matrix language so loosely that dimension meaning gets blurry
- forgetting to check total element count
- expecting `reshape(-1, -1)` to work
- confusing `reshape` with transpose

## Best Short Summary

```text
reshape keeps the values and changes the organization
```

```text
the legality test is simple:
new shape dimensions must multiply to the same total number of elements
```

```text
always distinguish:
[6] vs [1, 6] vs [6, 1]
```
