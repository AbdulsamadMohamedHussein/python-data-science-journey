# Day 2: Broadcasting, Vectorization, Random Numbers and Linear Algebra - Study Notes

> Part of my [Python Data Science Journey](../README.md). Code for these notes is in [`main.py`](main.py).

## 1. Broadcasting

Broadcasting lets NumPy do math between arrays of **different shapes** by stretching the smaller one to match the bigger one. No data is really copied, so it is fast.

```python
a = np.array([[1, 2, 3], [4, 5, 6]])   # (2, 3)
a + 10                                  # scalar goes to every element
a + np.array([10, 20, 30])              # row is added to EACH row
a + np.array([[100], [200]])            # column is added to EACH column
```

### The rules

NumPy compares shapes from the **right-most** dimension. Two dimensions are compatible when:

1. they are **equal**, or
2. one of them is **1**

Missing dimensions on the left count as 1.

| Shapes | Result |
|--------|--------|
| `(2, 3) + (3,)` | OK, shape `(2, 3)` |
| `(2, 3) + (2, 1)` | OK, shape `(2, 3)` |
| `(3, 1) + (1, 4)` | OK, shape `(3, 4)` |
| `(2, 3) + (2,)` | **Error**: 3 vs 2, neither is 1 |

**Fix for the error:** reshape the small array into a column with `bad.reshape(-1, 1)`.

## 2. `np.newaxis`

Adds a new dimension of size 1, so you control how arrays broadcast.

```python
v[:, np.newaxis]   # (3,) -> (3, 1) column
v[np.newaxis, :]   # (3,) -> (1, 3) row
v[:, np.newaxis] - v[np.newaxis, :]   # all pairwise differences
```

## 3. Normalizing Data with Broadcasting

```python
# Min-max scaling (0 to 1), per column
scaled = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

# Z-score (mean 0, std 1), per column
z = (data - data.mean(axis=0)) / data.std(axis=0)

# Row statistics need keepdims=True to broadcast
centered = data - data.mean(axis=1, keepdims=True)
```

`keepdims=True` keeps the reduced axis as size 1, so the shape is `(5, 1)` instead of `(5,)`.

## 4. Vectorization vs Loops

**Vectorization** means replacing Python `for` loops with NumPy operations. In my speed test on 2 million numbers, NumPy was roughly **30x faster** than a loop.

| Task | Loop-free tool |
|------|----------------|
| if / else on every element | `np.where(cond, a, b)` |
| Limit values to a range | `np.clip(arr, low, high)` |
| Sum, mean, max | `arr.sum()`, `arr.mean()`, `arr.max()` |

**Rule of thumb:** if I write a loop over array elements, I should look for a NumPy function first.

### Distance matrix (broadcasting + vectorization)

```python
diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]   # (n, n, 2)
dist = np.sqrt((diff ** 2).sum(axis=2))                      # (n, n)
```

This is the idea behind K-Nearest Neighbors and clustering.

## 5. Random Numbers

Use the modern generator and always set a **seed** so results are repeatable.

```python
rng = np.random.default_rng(seed=7)
rng.random(3)                         # floats in [0, 1)
rng.integers(1, 7, size=5)            # dice rolls (7 is excluded)
rng.normal(70, 10, size=4)            # mean 70, std 10
rng.choice(items, size=3, replace=False)
rng.permutation(items)                # shuffled copy
```

**Law of large numbers:** the mean of 100,000 dice rolls came out close to 3.5. More samples means a result closer to the true value.

## 6. Dot Product and Matrix Multiplication

| Operation | Symbol | Meaning |
|-----------|--------|---------|
| Element-wise | `A * B` | Multiply matching positions |
| Matrix multiplication | `A @ B` | Row of A times column of B |
| Dot product | `u @ w` or `np.dot(u, w)` | `1*4 + 2*5 + 3*6 = 32` |

**Shape rule:** `(m, n) @ (n, k) -> (m, k)`. The **inner** numbers must match. The **outer** numbers give the result shape.

- `(3, 4) @ (4, 2)` works and gives `(3, 2)`
- `(3, 4) @ (3, 4)` fails because 4 does not equal 3
- **Order matters:** `A @ B` is usually not equal to `B @ A`

## 7. Linear Algebra Tools (`np.linalg`)

| Code | Meaning |
|------|---------|
| `A.T` | Transpose |
| `np.trace(A)` | Sum of the diagonal |
| `np.linalg.det(A)` | Determinant |
| `np.linalg.matrix_rank(A)` | Rank |
| `np.linalg.inv(A)` | Inverse (`A @ inv(A)` is the identity) |
| `np.linalg.solve(A, b)` | Solve `A x = b` |
| `np.linalg.norm(v)` | Length of a vector |
| `np.allclose(a, b)` | Compare floats safely |

### Solving equations

```python
# 2x + y = 5
#  x + 3y = 10
A = np.array([[2.0, 1.0], [1.0, 3.0]])
b = np.array([5.0, 10.0])
np.linalg.solve(A, b)    # [1. 3.]  ->  x = 1, y = 3
```

**Never compare decimals with `==`.** Use `np.allclose`.

## 8. Eigenvalues (first look)

An eigenvector keeps its **direction** when multiplied by the matrix. It only gets stretched by the eigenvalue: `A @ v = lambda * v`. This is the foundation of **PCA**, an important AI technique.

```python
values, vectors = np.linalg.eig(S)
```

## 9. AI Mini Examples

**Cosine similarity** (1 means same direction):

```python
(p @ q) / (np.linalg.norm(p) * np.linalg.norm(q))
```

**One neural network layer:**

```python
out = X @ W + b              # (4,3) @ (3,2) -> (4,2), bias broadcasts
activated = np.maximum(0, out)   # ReLU: negatives become 0
```

**Linear regression with matrices** (data followed `y = 2x + 1`):

```python
design = np.column_stack([np.ones_like(xs), xs])
weights, *_ = np.linalg.lstsq(design, ys, rcond=None)   # [1, 2]
```

## 10. Common Mistakes

1. Using `*` when I want matrix multiplication (use `@`)
2. Forgetting that `(3,)` and `(3, 1)` are different shapes
3. Comparing floats with `==`
4. Forgetting `keepdims=True` when subtracting row statistics
5. Looping over array elements when a NumPy function exists
6. Not setting a random seed
7. Thinking `A @ B` equals `B @ A`

## 11. Quick Reference Card

```python
import numpy as np

a + row;  a + col;  v[:, np.newaxis]
(x - x.mean(axis=0)) / x.std(axis=0)
np.where(cond, a, b);  np.clip(x, lo, hi)
rng = np.random.default_rng(42);  rng.integers(1, 7, 5);  rng.normal(0, 1, 3)
A @ B;  u @ w;  A.T
np.linalg.inv(A);  np.linalg.solve(A, b);  np.linalg.norm(v);  np.linalg.det(A)
np.allclose(x, y)
```

## 12. My Own Words

*(Fill this in yourself. This is the part employers actually read.)*

- **The most useful thing I learned today:**
- **What confused me at first (broadcasting? matrix shapes?):**
- **How I figured it out:**
- **One thing I want to practice more:**

## Resources Used

- NumPy official docs: "Broadcasting" and "Linear algebra (numpy.linalg)"
- Add the tutorials or videos you actually used here

---

**Next:** Day 3 is the NumPy mini project: a **Student Grade Analyzer**.