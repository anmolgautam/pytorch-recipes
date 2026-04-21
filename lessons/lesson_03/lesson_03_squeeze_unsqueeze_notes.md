# Lesson 03: Squeeze and Unsqueeze Notes

## Lesson Goal

Understand how to remove and add size-1 dimensions on purpose.

This lesson is really about control over tensor shape.

`reshape` changed the overall arrangement of elements.
`squeeze` and `unsqueeze` are narrower tools:

- `squeeze` removes dimensions of size 1
- `unsqueeze` adds a new dimension of size 1

These operations matter because model code often expects very specific shapes.

## Core Ideas

### 1. `squeeze` does not mean "make it 1D"

This was the first important correction.

It is tempting to think:

```text
squeeze -> flatten or turn into a vector
```

That is not the rule.

The real rule is:

```text
squeeze removes dimensions whose size is 1
```

So:

- `[3, 1]` becomes `[3]`
- `[1, 3, 1]` becomes `[3]`
- `[2, 3]` stays `[2, 3]`

The result depends entirely on where the size-1 dimensions are.

### 2. `unsqueeze` adds one new size-1 dimension

`unsqueeze(dim)` inserts a new dimension of size 1 at the specified position.

Examples:

- `[3] -> unsqueeze(0) -> [1, 3]`
- `[3] -> unsqueeze(1) -> [3, 1]`
- `[1, 3] -> unsqueeze(-1) -> [1, 3, 1]`

Only one new dimension is added each time.

This was an important correction during the lesson: `unsqueeze` does not add multiple dimensions at once.

### 3. Axis position matters

The whole meaning of `unsqueeze` is tied to position.

For a tensor of shape `[3]`:

- `unsqueeze(0)` adds a dimension in front, giving `[1, 3]`
- `unsqueeze(1)` adds a dimension at the end, giving `[3, 1]`

Same values, different meaning.

That distinction becomes very important in model code.

## Why Size-1 Dimensions Matter

At first, size-1 dimensions can feel useless.
They are not useless.

They often encode meaning such as:

- batch size 1
- one channel
- one feature column
- one sequence step

A tensor with shape `[3]` is not the same thing as:

- `[1, 3]`
- `[3, 1]`

Even if the numbers are the same, the tensor can behave differently in:

- matrix multiplication
- broadcasting
- batching
- model input handling

## Main Confusions And Clarifications

### Confusion 1: "`squeeze` converts a tensor to 1D"

Clarification:

`squeeze` only removes size-1 dimensions.

So if a tensor has no size-1 dimensions, nothing changes.

Example:

```text
[2, 3] -> squeeze() -> [2, 3]
```

### Confusion 2: "`squeeze(dim)` errors if that dimension is not size 1"

Clarification:

In PyTorch, `squeeze(dim)` does not error in that case.
It simply returns the tensor unchanged.

That is important because it makes `squeeze(dim)` safer and more predictable than many beginners expect.

### Confusion 3: "`unsqueeze(-2)` adds a lot of dimensions"

Clarification:

Negative indices just tell PyTorch where to insert the one new dimension.

They do not mean "keep adding dimensions until the index makes sense."

So if:

```text
[1, 3] -> unsqueeze(-2)
```

the result is:

```text
[1, 1, 3]
```

because one size-1 dimension is inserted at the second-last position.

## Negative Axis Intuition

Negative indices count from the end.

Examples:

- `unsqueeze(-1)` means add a new size-1 dimension at the end
- `squeeze(-1)` means try to remove the last dimension if its size is 1

This is useful because in model code we often care about the last dimension semantically, such as:

- last dimension is features
- last dimension is channels
- last dimension is embedding size

## Real Model Intuition

Suppose a model expects shape:

```text
[batch, features]
```

If we have a vector of shape `[3]`, that does not yet explicitly say what is batch and what is feature.

To represent one sample with three features, we use:

```text
[1, 3]
```

This is why `unsqueeze(0)` matters.

To represent a column-style tensor, we may want:

```text
[3, 1]
```

This is why `unsqueeze(1)` or `unsqueeze(-1)` matters.

The numbers may be identical, but the meaning is different.

## Rules Of Thumb

- `squeeze` removes size-1 dimensions
- `unsqueeze` inserts one size-1 dimension
- `squeeze()` removes all size-1 dimensions
- `squeeze(dim)` only targets one dimension
- if that dimension is not size 1, PyTorch leaves the tensor unchanged
- `unsqueeze(0)` usually adds a batch dimension in front
- `unsqueeze(-1)` often creates a column-style last dimension

## Debugging Tips

When a tensor shape is confusing, ask:

1. Which dimensions have size 1?
2. Am I trying to remove one of them or insert one?
3. Do I want batch-in-front behavior like `[1, n]`?
4. Do I want column-style behavior like `[n, 1]`?
5. What does the next operation expect?

This last question matters most:

```text
What shape does the next layer or operation expect?
```

That is often the real reason `squeeze` or `unsqueeze` shows up.

## Lesson Takeaway

`squeeze` and `unsqueeze` are simple, but they sharpen shape control.

This lesson helps build the habit of seeing:

- where size-1 dimensions are
- what those dimensions mean
- how adding or removing them changes model behavior

That shape discipline is exactly what we need before moving into more advanced layout operations.

## Next Topic

The next natural topic is `flatten`, followed by axis-reordering operations such as `transpose` and `permute`.
