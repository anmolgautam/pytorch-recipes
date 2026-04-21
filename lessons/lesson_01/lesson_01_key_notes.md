# Lesson 01: Key Notes Cheat Sheet

## Core Terms

- `tensor`: a box of numbers
- `shape`: tells how many dimensions a tensor has and the size of each dimension
- `ndim`: number of dimensions
- `dtype`: how the values are represented in memory
- `device`: where the tensor lives, such as `cpu` or `mps`
- `scalar tensor`: a 0D tensor

## Shape Basics

```python
torch.tensor([1, 2, 3]).shape
# torch.Size([3])
```

```python
torch.tensor([[1, 2, 3]]).shape
# torch.Size([1, 3])
```

```python
torch.tensor([[1], [2], [3]]).shape
# torch.Size([3, 1])
```

Rule:

```text
same values can have different shapes
```

## How To Read Shape

Example:

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
x.shape
# torch.Size([2, 3])
```

Meaning:

```text
dimension 0 has size 2
dimension 1 has size 3
```

For a 2D tensor:

```text
2 rows
3 columns
```

## Important Distinction

```python
x.ndim
# 2
```

```python
x.shape[0]
# 2
```

These are not the same idea.

```text
x.ndim = number of dimensions
x.shape[0] = size of dimension 0
```

## Indexing And Slicing

```python
x[0]
```

Gets the first row.

```python
x[0, 1]
```

Gets one single element.

Cleaner than:

```python
x[0][1]
```

```python
x[:, 1]
```

Take all rows and column index `1`.

```python
x[:, 1:2]
```

Take all rows and a slice of the second column.

## Biggest Shape Trick

```text
integer indexing removes a dimension
slice indexing preserves a dimension
```

Example:

```python
x[:, 1].shape
# torch.Size([2])
```

```python
x[:, 1:2].shape
# torch.Size([2, 1])
```

Same values, different shape.

## Scalar, 1D, 2D

```python
x[0, 1].shape
# torch.Size([])
```

This is:

```text
0D scalar tensor
```

```python
x[0].shape
# torch.Size([3])
```

This is:

```text
1D tensor
```

```python
x.shape
# torch.Size([2, 3])
```

This is:

```text
2D tensor
```

## `dtype`

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
x.dtype
# torch.int64
```

This means:

```text
integer type
64 bits per value
```

```python
y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y.dtype
# torch.float32
```

This means:

```text
floating-point type
32 bits per value
```

Important:

```text
torch.int64 is not a floating type
```

## `device`

```python
x.device
# cpu
```

```python
z = torch.tensor([1.0, 2.0, 3.0], device="mps")
z.device
# mps
```

Rule:

```text
tensors in the same operation usually need to be on the same device
```

## Broadcasting

Broadcasting means PyTorch can stretch a smaller tensor so an operation works.

Main rule:

```text
PyTorch compares shapes from the right
```

Two dimensions are compatible if:

```text
they are equal
or one of them is 1
```

## Broadcasting Examples

```text
[2, 3]
[   3]
```

Works because:

```text
3 matches 3
```

```text
[2, 3]
[   2]
```

Fails because:

```text
3 does not match 2
neither is 1
```

```text
[2, 3]
[2, 1]
```

Works because:

```text
2 matches 2
1 stretches to 3
```

## Why `reshape(2, 1)` Helped

```python
a = torch.tensor([10.0, 20.0])
a.shape
# torch.Size([2])
```

This could not broadcast with shape `[2, 3]`.

But:

```python
a_col = a.reshape(2, 1)
a_col.shape
# torch.Size([2, 1])
```

Now it works with `[2, 3]` because:

```text
2 matches 2
1 stretches to 3
```

## Mental Checklist

Whenever something feels confusing, ask:

```text
What is the shape?
What is the dtype?
What is the device?
```

## Common Mistakes

- mixing up `ndim` and `shape[0]`
- saying `shape[0]` means number of dimensions
- forgetting that Python indexing starts at `0`
- thinking `[3]` and `[1, 3]` are the same
- forgetting that integer indexing removes a dimension
- trying to operate on tensors on different devices
- ignoring shape mismatch when broadcasting fails

## Best Short Summary

```text
Shape tells us the structure.
Indexing changes shape.
Broadcasting depends on shape.
dtype controls the kind of values.
device controls where the tensor lives.
```

## Next Topic

```text
reshape
```
