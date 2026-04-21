# Lesson 02: Reshape Notes

## Lesson Goal

Learn what `reshape` does, when it works, and how to reason about tensor shapes before and after a reshape.

## Core Idea

`reshape` changes the shape of a tensor without changing the underlying values.

The values stay the same.
The arrangement changes.

## Theory And Intuition

The cleanest way to think about `reshape` is this:

```text
same numbers
new structure
```

If a tensor has values:

```text
1, 2, 3, 4, 5, 6
```

then `reshape` does not create new values and does not destroy values.
It only asks:

```text
how should these same values be organized into dimensions?
```

That is why a tensor can move between shapes like:

```text
[6]
[2, 3]
[3, 2]
[1, 6]
[6, 1]
```

as long as all of them contain room for the same total number of elements.

The governing law of `reshape` is:

```text
the total number of elements must remain the same
```

This is why `numel()` matters.

If a tensor has `6` elements, then valid reshapes are shapes whose dimensions multiply to `6`.

Examples:

```text
[2, 3] -> 2 * 3 = 6
[3, 2] -> 3 * 2 = 6
[1, 6] -> 1 * 6 = 6
[6, 1] -> 6 * 1 = 6
```

An invalid reshape is one whose dimensions multiply to a different number.

Example:

```text
[2, 2] -> 2 * 2 = 4
```

This cannot hold 6 values, so it fails.

## The Most Important Mental Shift

It is easy to think about `reshape` too much in matrix language.

That is sometimes helpful, but the more general PyTorch way is:

```text
shape[i] tells us the size of dimension i
```

For a 2D tensor:

```text
[rows, cols]
```

is a perfectly fine interpretation.

But the deeper habit is:

```text
[1, 6] means dimension 0 has size 1 and dimension 1 has size 6
[6, 1] means dimension 0 has size 6 and dimension 1 has size 1
```

This matters because tensors in deep learning are often not just matrices.
Later we will see shapes like:

```text
[batch, channels, height, width]
[batch, sequence_length, embedding_dim]
```

So `reshape` should be understood in dimension language first and matrix language second.

## Examples

We started with a 1D tensor of 6 values.

Shape:

```text
[6]
```

This means:

```text
one dimension
size 6
```

Then we reshaped it to:

```text
[2, 3]
```

This means:

```text
2 dimensions
dimension 0 has size 2
dimension 1 has size 3
```

For a 2D tensor, we can also say:

```text
2 rows
3 columns
```

Then we reshaped it to:

```text
[3, 2]
```

Same values, new arrangement.

Later we used:

```text
[1, 6]
```

and:

```text
[6, 1]
```

These were especially useful because they exposed an important confusion:

```text
[6] is 1D
[1, 6] is 2D
[6, 1] is 2D
```

These shapes are not interchangeable.

## Common Confusions

### Confusion 1: Thinking `reshape` changes the values

It does not.

It changes only the structure of the tensor.

### Confusion 2: Thinking `[1, 6]` is basically the same as `[6]`

It is not.

```text
[6] has one dimension
[1, 6] has two dimensions
```

That difference matters in real PyTorch code.

### Confusion 3: Thinking in terms like "each dimension gets values"

Better wording is:

```text
dimension 0 has size ...
dimension 1 has size ...
```

This is more precise and scales better to higher-dimensional tensors.

### Confusion 4: Believing any reshape should work if it looks visually reasonable

The only real test is:

```text
does the new shape preserve the total number of elements?
```

If not, it fails.

### Confusion 5: Thinking `reshape` is the same as transpose

Not at all.

`reshape` reinterprets the same sequence of values in a new shape.
Transpose changes which axes are swapped.
That distinction will matter more later.

## Clarifications

`numel()` tells us how many total elements the tensor has.

That makes it the quickest legality check for `reshape`.

If:

```text
numel = 6
```

then every valid reshape must describe exactly 6 total slots.

Also:

```text
-1 means infer this dimension automatically
```

So:

```text
reshape(2, -1)
```

means:

```text
make dimension 0 equal to 2
let PyTorch solve dimension 1
```

Since the total element count was 6, PyTorch infers `3`.

Important rule:

```text
only one dimension can be -1
```

If more than one dimension is `-1`, the shape is ambiguous.

## Debugging Tips

When `reshape` fails, do not stare at the error message first.

Ask:

```text
How many total elements does the tensor have?
What shape am I asking for?
Do those dimensions multiply to the same total?
```

That will usually explain the error immediately.

When your mental picture feels fuzzy, write both forms explicitly:

```text
old shape: [6]
new shape: [2, 3]
```

Then check:

```text
6 = 2 * 3
```

This simple habit prevents a lot of mistakes.

## Rules Of Thumb

- `reshape` changes shape, not values
- valid reshape means total element count stays the same
- use `numel()` when in doubt
- `[6]`, `[1, 6]`, and `[6, 1]` are all different shapes
- for 2D tensors, row-column language is okay
- for general tensors, think in terms of dimension sizes
- `-1` means PyTorch infers one missing dimension
- only one `-1` is allowed

## Exercises

In this lesson we practiced:

- reshaping a 1D tensor of 6 values into `[2, 3]`
- reshaping the same tensor into `[3, 2]`
- predicting why `[2, 2]` fails
- using `numel()` to justify valid reshapes
- comparing `[6]` with `[1, 6]`
- comparing `[1, 6]` with `[6, 1]`
- using `-1` to let PyTorch infer one dimension
- understanding why `reshape(-1, -1)` fails

## Takeaway Summary

`reshape` is one of the first places where tensor thinking becomes more mature.

The lesson was not really about memorizing a function.
It was about learning to ask:

```text
How many elements are there?
How many dimensions will the new shape have?
What is the size of each dimension?
```

The deepest clarification from this lesson was:

```text
[6] is 1D
[1, 6] is 2D
[6, 1] is 2D
```

Same values do not mean same shape.

## Next Topic

Likely:

```text
squeeze and unsqueeze
```
