# Lesson 01: Tensors Notes

## Core Idea

A tensor is a box of numbers with:

- a shape
- a data type
- a device

For our first lesson, the most important idea was shape.

```text
A tensor's shape tells us how many dimensions it has and the size of each dimension.
```

Example:

```python
import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
x
```

Output:

```python
tensor([[1, 2, 3],
        [4, 5, 6]])
```

This is a 2D tensor. It looks like a matrix:

```text
row 1: 1  2  3
row 2: 4  5  6
```

## Shape

```python
x.shape
```

Output:

```python
torch.Size([2, 3])
```

This means:

```text
dimension 0 has size 2
dimension 1 has size 3
```

For this 2D tensor, we can also say:

```text
2 rows
3 columns
```

## How To Determine Shape

Shape is the structure of the nested lists used to create the tensor.

Start from the outside and move inward.

Example:

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
```

Look at the outer list:

```text
[
  [1, 2, 3],
  [4, 5, 6]
]
```

The outer list has 2 items:

```text
[1, 2, 3]
[4, 5, 6]
```

So the first shape value is `2`.

Each inner list has 3 values:

```text
1, 2, 3
4, 5, 6
```

So the second shape value is `3`.

Therefore:

```python
x.shape
```

is:

```python
torch.Size([2, 3])
```

Read it as:

```text
2 groups
3 values in each group
```

For a 2D tensor, those groups are often called rows, so we say:

```text
2 rows
3 columns
```

### Shape From Brackets

The number of nested bracket levels usually tells you the number of dimensions.

```python
torch.tensor(5)
```

Shape:

```python
torch.Size([])
```

Meaning:

```text
0D scalar tensor
```

Example:

```python
torch.tensor([1, 2, 3])
```

Shape:

```python
torch.Size([3])
```

Meaning:

```text
1D tensor with 3 values
```

Example:

```python
torch.tensor([[1, 2, 3]])
```

Shape:

```python
torch.Size([1, 3])
```

Meaning:

```text
2D tensor
1 row
3 columns
```

Example:

```python
torch.tensor([[1], [2], [3]])
```

Shape:

```python
torch.Size([3, 1])
```

Meaning:

```text
2D tensor
3 rows
1 column
```

The values may be the same, but the structure is different:

```text
[1, 2, 3]        -> shape [3]
[[1, 2, 3]]      -> shape [1, 3]
[[1], [2], [3]]  -> shape [3, 1]
```

### Shape For 3D Tensors

Example:

```python
e = torch.tensor([
    [[1, 2, 3]],
    [[4, 5, 6]]
])
```

Count from outside to inside:

```text
outer list has 2 items
each item has 1 row
each row has 3 values
```

So:

```python
e.shape
```

is:

```python
torch.Size([2, 1, 3])
```

Read it as:

```text
2 blocks
1 row per block
3 values per row
```

In deep learning, dimensions often get names:

```text
[batch, channels, height, width]
[batch, sequence_length, embedding_dim]
```

But the idea is the same:

```text
shape tells us the size of each direction in the tensor
```

### A Useful Shape Habit

Whenever you see a tensor, ask:

```text
How many dimensions does it have?
What is the size of each dimension?
What does each dimension mean in this problem?
```

Important distinction:

```python
x.ndim
```

Output:

```python
2
```

`x.ndim` tells us how many dimensions the tensor has.

`x.shape` tells us the size of each dimension.

So:

```text
x.ndim = number of dimensions
x.shape[0] = size of dimension 0
x.shape[1] = size of dimension 1
```

For our tensor:

```python
x.shape[0]
```

Output:

```python
2
```

This means dimension 0 has size 2, or in this case, there are 2 rows.

```python
x.shape[1]
```

Output:

```python
3
```

This means dimension 1 has size 3, or in this case, there are 3 columns.

## Confusion We Corrected

At first, it was easy to mix these up:

```text
x.ndim = 2
x.shape[0] = 2
```

They are both `2`, but they mean different things.

```text
x.ndim = the tensor has 2 dimensions
x.shape[0] = dimension 0 has size 2
```

This distinction matters a lot.

## Indexing

```python
x[0]
```

Output:

```python
tensor([1, 2, 3])
```

This gives the values at index 0 along dimension 0.

Since dimension 0 is the row direction here, `x[0]` gives the first row.

The shape changes:

```python
x[0].shape
```

Output:

```python
torch.Size([3])
```

So:

```text
x.shape = torch.Size([2, 3])
x[0].shape = torch.Size([3])
```

Indexing selected one row and removed one dimension.

## Scalar Tensor

```python
x[0][1]
```

Output:

```python
tensor(2)
```

This means:

```text
go to row/index 0 -> tensor([1, 2, 3])
then go to index 1 -> 2
```

The cleaner PyTorch style is:

```python
x[0, 1]
```

This gives the same value:

```python
tensor(2)
```

Its shape is:

```python
x[0, 1].shape
```

Output:

```python
torch.Size([])
```

`torch.Size([])` means this is a scalar tensor, or a 0D tensor.

Shape descent:

```text
x.shape        -> torch.Size([2, 3])  -> 2D tensor
x[0].shape     -> torch.Size([3])     -> 1D tensor
x[0, 1].shape  -> torch.Size([])      -> 0D scalar tensor
```

Important pattern:

```text
indexing usually removes dimensions
```

## Slicing

```python
x[:, 1]
```

Output:

```python
tensor([2, 5])
```

Read it as:

```text
: means take all rows
1 means take column/index 1
```

Because Python indexing starts at 0, column index 1 is the second column.

Shape:

```python
x[:, 1].shape
```

Output:

```python
torch.Size([2])
```

This is a 1D tensor with 2 values.

Now compare:

```python
x[:, 1:2]
```

Output:

```python
tensor([[2],
        [5]])
```

Shape:

```python
x[:, 1:2].shape
```

Output:

```python
torch.Size([2, 1])
```

The values are the same, but the shape is different.

```text
x[:, 1]   -> torch.Size([2])
x[:, 1:2] -> torch.Size([2, 1])
```

Rule:

```text
integer indexing removes a dimension
slice indexing keeps a dimension
```

This is one of the first big PyTorch tricks.

## Data Type: `dtype`

```python
x.dtype
```

Output:

```python
torch.int64
```

This means:

```text
integer data type
64-bit integer
whole numbers only
```

Important correction:

```text
torch.int64 is not floating precision.
```

Floating-point types are things like:

```python
torch.float32
torch.float64
```

Example:

```python
y = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y.dtype
```

Output:

```python
torch.float32
```

Because the values contain decimals, PyTorch infers `float32`.

`float32` is the default workhorse dtype for deep learning.

## Adding A Decimal To An Integer Tensor

```python
x + 0.5
```

Output:

```python
tensor([[1.5000, 2.5000, 3.5000],
        [4.5000, 5.5000, 6.5000]])
```

This did two things:

```text
0.5 was broadcast to every element
the result became floating-point
```

Check:

```python
(x + 0.5).dtype
```

Output:

```python
torch.float32
```

Rule:

```text
integer tensor + decimal number -> floating-point result
```

The extra zeros in `1.5000` are just display formatting.

## Device

A tensor lives somewhere:

```text
CPU memory
or GPU memory
```

Check:

```python
x.device
```

Output:

```python
device(type='cpu')
```

On a Mac with Apple Silicon, PyTorch may support the Apple GPU through MPS.

Check:

```python
torch.backends.mps.is_available()
```

Output:

```python
True
```

Create a tensor on MPS:

```python
z = torch.tensor([1.0, 2.0, 3.0], device="mps")
z.device
```

Output:

```python
device(type='mps', index=0)
```

This means `z` lives on the Apple GPU backend.

## Device Mismatch Error

If one tensor is on CPU and another is on MPS:

```python
x + z
```

Error idea:

```text
expected all tensors to be on same device
```

PyTorch cannot directly operate on tensors that live on different devices.

Move `x` to MPS:

```python
x_mps = x.to("mps")
x_mps.device
```

Output:

```python
device(type='mps', index=0)
```

Now `x_mps` and `z` are on the same device.

## Broadcasting

Broadcasting is when PyTorch automatically expands a smaller tensor so an operation can happen.

The key rule:

```text
PyTorch compares shapes from the right side.
```

Two dimensions are compatible if:

```text
they are equal
or one of them is 1
```

If one dimension is `1`, PyTorch can stretch it.

If the dimensions are different and neither is `1`, broadcasting fails.

### Broadcasting With A Scalar

Example:

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
x + 0.5
```

Here `0.5` is a scalar.

PyTorch treats it as if it can be reused for every element:

```text
[[1, 2, 3],      [[0.5, 0.5, 0.5],
 [4, 5, 6]]  +    [0.5, 0.5, 0.5]]
```

Result:

```python
tensor([[1.5000, 2.5000, 3.5000],
        [4.5000, 5.5000, 6.5000]])
```

This is scalar broadcasting.

### Broadcasting A Row-Like Tensor

Example:

```python
x_mps.shape
```

```python
torch.Size([2, 3])
```

```python
z.shape
```

```python
torch.Size([3])
```

Line them up from the right:

```text
[2, 3]
[   3]
```

The last dimensions match:

```text
3 matches 3
```

So PyTorch can reuse `z` for each row.

If:

```python
x_mps = torch.tensor([[1, 2, 3], [4, 5, 6]], device="mps")
z = torch.tensor([1.0, 2.0, 3.0], device="mps")
```

Then:

```python
x_mps + z
```

works like:

```text
[1, 2, 3] + [1, 2, 3] = [2, 4, 6]
[4, 5, 6] + [1, 2, 3] = [5, 7, 9]
```

Output:

```python
tensor([[2., 4., 6.],
        [5., 7., 9.]], device='mps:0')
```

Result shape:

```python
torch.Size([2, 3])
```

Broadcasting usually keeps the larger resulting shape.

### Broadcasting A Column-Like Tensor

Suppose:

```python
a = torch.tensor([10.0, 20.0], device="mps")
```

Its shape is:

```python
torch.Size([2])
```

This does not add cleanly to a `[2, 3]` tensor:

```text
[2, 3]
[   2]
```

The last dimensions are:

```text
3 vs 2
```

They do not match, and neither one is `1`.

So we reshape:

```python
a_col = a.reshape(2, 1)
```

Now:

```text
x_mps.shape = [2, 3]
a_col.shape = [2, 1]
```

Compare dimension by dimension:

```text
2 matches 2
1 stretches to 3
```

So:

```text
[[10],
 [20]]
```

can behave like:

```text
[[10, 10, 10],
 [20, 20, 20]]
```

Then the addition works:

```text
[1, 2, 3] + [10, 10, 10] = [11, 12, 13]
[4, 5, 6] + [20, 20, 20] = [24, 25, 26]
```

### Broadcasting Checklist

To decide whether broadcasting works:

1. Write down both shapes.
2. Align them from the right.
3. Compare each dimension pair.
4. A pair works if the numbers are equal or one number is `1`.
5. If any pair has different numbers and neither is `1`, it fails.

Examples:

```text
[2, 3]
[   3]
```

Works because:

```text
3 matches 3
```

Example:

```text
[2, 3]
[   2]
```

Fails because:

```text
3 does not match 2
neither is 1
```

Example:

```text
[2, 3]
[2, 1]
```

Works because:

```text
2 matches 2
1 stretches to 3
```

### Why Broadcasting Matters

Broadcasting lets us write clean vectorized code.

Instead of manually looping through rows:

```python
for row in x:
    row + z
```

we can write:

```python
x + z
```

This is shorter, clearer, and usually much faster.

Example:

```python
x_mps.shape
```

Output:

```python
torch.Size([2, 3])
```

```python
z.shape
```

Output:

```python
torch.Size([3])
```

Add them:

```python
x_mps + z
```

Output:

```python
tensor([[2., 4., 6.],
        [5., 7., 9.]], device='mps:0')
```

PyTorch lines up shapes from the right:

```text
[2, 3]
[   3]
```

The last dimension matches, so `z` is reused for every row.

Result shape:

```python
(x_mps + z).shape
```

Output:

```python
torch.Size([2, 3])
```

The result keeps the larger shape.

## Broadcasting Error

Create:

```python
a = torch.tensor([10.0, 20.0], device="mps")
```

Try:

```python
x_mps + a
```

Error:

```text
RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1
```

Shapes:

```text
x_mps.shape = [2, 3]
a.shape     = [2]
```

Line them up from the right:

```text
[2, 3]
[   2]
```

Compare the last dimensions:

```text
3 vs 2
```

They do not match, and neither is `1`, so broadcasting fails.

`non-singleton dimension` means:

```text
a dimension whose size is not 1
```

Size `1` is special because it can stretch during broadcasting.

## Fixing Broadcasting With `reshape`

```python
a_col = a.reshape(2, 1)
a_col.shape
```

Output:

```python
torch.Size([2, 1])
```

Now add:

```python
x_mps + a_col
```

Output:

```python
tensor([[11., 12., 13.],
        [24., 25., 26.]], device='mps:0')
```

Why this works:

```text
x_mps.shape = [2, 3]
a_col.shape = [2, 1]
```

Compare:

```text
2 matches 2
1 can stretch to 3
```

So:

```text
[10] -> [10, 10, 10]
[20] -> [20, 20, 20]
```

Then:

```text
[1, 2, 3] + [10, 10, 10] = [11, 12, 13]
[4, 5, 6] + [20, 20, 20] = [24, 25, 26]
```

Student phrasing:

```text
It converted into a column vector so broadcasting worked and dimension 0 matched as 2.
```

Polished phrasing:

```text
a.reshape(2, 1) changed `a` from shape [2] to shape [2, 1].
Then dimension 0 matched, and dimension 1 could stretch from 1 to 3.
```

## Shape Practice

### Column Vector

```python
b = torch.tensor([[1], [2], [3]])
b.shape
```

Output:

```python
torch.Size([3, 1])
```

Meaning:

```text
3 rows
1 column
```

### 1D Tensor

```python
c = torch.tensor([1, 2, 3])
c.shape
```

Output:

```python
torch.Size([3])
```

This is not `[1, 3]`.

Why?

```text
[1, 2, 3] has one level of brackets, so it is 1D.
```

### Row Vector Style 2D Tensor

```python
d = torch.tensor([[1, 2, 3]])
d.shape
```

Output:

```python
torch.Size([1, 3])
```

Why?

```text
[[1, 2, 3]] has two levels of brackets.
```

The three siblings:

```text
c = [1, 2, 3]        -> shape [3]
d = [[1, 2, 3]]      -> shape [1, 3]
b = [[1], [2], [3]]  -> shape [3, 1]
```

Same numbers. Different structure.

## Higher-Dimensional Shape Practice

Predict:

```python
e = torch.tensor([
    [[1, 2, 3]],
    [[4, 5, 6]]
])
e.shape
```

Correct output:

```python
torch.Size([2, 1, 3])
```

Why:

```text
outer list has 2 items
each item has 1 row
each row has 3 values
```

Read:

```text
2 blocks
1 row per block
3 values per row
```

This is the same idea behind common deep learning shapes:

```text
[batch, channels, height, width]
[batch, sequence_length, embedding_dim]
```

## Main Rules From This Lesson

```text
shape tells us how many dimensions a tensor has and the size of each dimension
ndim tells us only the number of dimensions
integer indexing usually removes a dimension
slice indexing usually preserves a dimension
dtype tells us how values are represented
device tells us where the tensor lives
tensors in the same operation must usually be on the same device
broadcasting lines up shapes from the right
a dimension of size 1 can stretch during broadcasting
same values can have different shapes
```

## The Big Takeaway

Deep learning code becomes much easier when we constantly ask:

```text
What is the shape?
What is the dtype?
What is the device?
```

Today, the deepest lesson was shape thinking.

Next topic:

```text
reshape
```
