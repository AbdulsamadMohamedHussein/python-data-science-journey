# Day 1: NumPy Basics - Study Notes

> Part of my [Python Data Science Journey](../README.md). Code for these notes is in [`main.py`](main.py).

## 1. What is NumPy?

NumPy (Numerical Python) is a library for fast math on arrays. Its main object is the **`ndarray`**, a grid of values that all share the same data type.

**Why use it instead of Python lists?**

| Python list | NumPy array |
|-------------|-------------|
| Can hold mixed types | All elements have one type |
| Slow loops for math | Fast, runs in C |
| `list * 2` repeats the list | `array * 2` multiplies every element |
| Needs `for` loops | Works on the whole array at once (vectorization) |

NumPy is the foundation of Pandas, Matplotlib, scikit-learn, and PyTorch, so it matters a lot for AI.

```python
import numpy as np
```

## 2. Array Dimensions

| Dimension | Name | Example use |
|-----------|------|-------------|
| 1D | Vector | A list of scores |
| 2D | Matrix | A spreadsheet, a grayscale image |
| 3D | Stack of matrices | A color image (height x width x channels) |

## 3. Array Attributes

| Attribute | Meaning | Example for `[[1,2,3],[4,5,6]]` |
|-----------|---------|----------------------------------|
| `.ndim` | Number of dimensions | `2` |
| `.shape` | Size of each dimension | `(2, 3)` = 2 rows, 3 columns |
| `.size` | Total number of elements | `6` |
| `.dtype` | Data type | `int64` |
| `.itemsize` | Bytes per element | `8` |

## 4. Data Types

- Common types: `int32`, `int64`, `float32`, `float64`, `bool`
- Mixing ints and floats makes everything a **float**
- Set the type when creating: `np.array([1, 2, 3], dtype=np.float32)`
- Convert with `astype`: `arr.astype(int)` returns a new array
- Converting float to int **cuts off decimals**, it does not round (`2.9` becomes `2`)

## 5. Creating Arrays

| Function | What it does |
|----------|--------------|
| `np.array([1, 2, 3])` | From a Python list |
| `np.zeros((2, 3))` | All zeros |
| `np.ones((2, 3))` | All ones |
| `np.full((2, 3), 7)` | All the same value |
| `np.eye(3)` | Identity matrix |
| `np.arange(0, 10, 2)` | Like `range`, stop **not** included |
| `np.linspace(0, 1, 5)` | 5 evenly spaced values, stop **included** |
| `np.random.rand(2, 3)` | Random floats between 0 and 1 |
| `np.random.randint(0, 10, 5)` | Random integers |
| `np.random.seed(42)` | Makes random results repeatable |

**`arange` vs `linspace`:** with `arange` I choose the **step**. With `linspace` I choose the **count**.

## 6. Indexing

- Counting starts at **0**. Negative index counts from the end (`-1` is the last).
- 2D uses `array[row, column]`.

```python
arr[0]       # first element
arr[-1]      # last element
m[1, 2]      # row 1, column 2
m[0, 0] = 100   # change a value
```

## 7. Slicing

Syntax: `array[start:stop:step]` (stop is **not** included).

```python
x[:3]        # first 3
x[-3:]       # last 3
x[::2]       # every second element
x[::-1]      # reversed
m[0]         # first row
m[:, 1]      # second column
m[:2, :2]    # top-left 2x2 block
```

### Important: views vs copies

A slice is a **view**, not a copy. Changing the slice changes the original array.

```python
view = original[1:4]        # changing view changes original
safe = original[1:4].copy() # independent copy
```

## 8. Reshaping

`reshape` changes the layout but not the data. The total number of elements must stay the same.

```python
r = np.arange(12)
r.reshape(3, 4)     # 3 rows, 4 columns
r.reshape(4, -1)    # -1 means "NumPy, work this one out"
grid.flatten()      # back to 1D (copy)
grid.T              # transpose: rows become columns
v.reshape(-1, 1)    # column vector, shape (3, 1)
v.reshape(1, -1)    # row vector, shape (1, 3)
```

## 9. Math Operations (Vectorization)

Operations apply to **every element**, no loops needed.

```python
p + q        # element-wise add
p * q        # element-wise multiply (NOT matrix multiplication)
p ** 2       # square each element
p * 10       # a single number applies to every element
np.sqrt(p)   np.exp(p)   np.log(p)
p > 2        # comparison returns True/False for each element
```

## 10. Statistics and the `axis` Parameter

| Function | Meaning |
|----------|---------|
| `sum()`, `mean()`, `median()`, `std()` | Basic statistics |
| `min()`, `max()` | Smallest and largest |
| `argmin()`, `argmax()` | **Position** of smallest and largest |
| `np.cumsum(a)` | Running total |

**The `axis` rule (the one that confused me most):**

- `axis=0` goes **down** the rows, giving **one result per column**
- `axis=1` goes **across** the columns, giving **one result per row**
- No `axis` uses every element

```python
scores.sum(axis=0)   # total of each column
scores.sum(axis=1)   # total of each row
```

## 11. Boolean Masking

A mask is an array of `True`/`False` used to filter data.

```python
mask = marks >= 50
marks[mask]                          # keep values that pass
marks[~mask]                         # ~ means NOT
marks[(marks >= 60) & (marks <= 80)] # & means AND
marks[(marks == 45) | (marks == 90)] # | means OR
np.where(marks >= 50, "pass", "fail")
```

**Rules to remember:**
- Use `&` and `|`, **not** `and` / `or`
- Put each condition in **parentheses**
- `(marks >= 50).sum()` counts the `True` values, because `True` counts as 1

## 12. Other Useful Tools

| Function | What it does |
|----------|--------------|
| `np.sort(a)` | Sorted copy |
| `np.argsort(a)` | Positions that would sort the array |
| `np.unique(a)` | Distinct values |
| `np.concatenate([a, b])` | Join arrays |
| `np.vstack([a, b])` | Stack as rows |
| `np.hstack([a, b])` | Place side by side |
| `.any()` / `.all()` | At least one / every element is True |

## 13. Common Mistakes

1. Thinking a slice is a copy (it is a view)
2. Mixing up `axis=0` and `axis=1`
3. Using `and` / `or` instead of `&` / `|` in masks
4. Forgetting parentheses around each condition
5. Expecting `astype(int)` to round
6. Expecting `arange` to include the stop value
7. Using `*` and expecting matrix multiplication (that is `@` or `np.dot`, Day 2)

## 14. Quick Reference Card

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
a.shape, a.ndim, a.size, a.dtype
np.zeros((2, 3)); np.ones((2, 3)); np.arange(0, 10, 2); np.linspace(0, 1, 5)
a[1, 2]; a[:, 0]; a[0, :]; a[::-1]
a.reshape(3, 2); a.T; a.flatten()
a.sum(axis=0); a.mean(axis=1); a.argmax()
a[a > 2]; np.where(a > 2, 1, 0)
```

## 15. My Own Words

*(Fill this in yourself. This is the part employers actually read.)*

- **The most useful thing I learned today:**
- **What confused me at first:**
- **How I figured it out:**
- **One thing I want to practice more:**

## Resources Used

- NumPy official docs: "NumPy: the absolute basics for beginners"
- Add the tutorials or videos you actually used here

---

**Next:** Day 2 covers broadcasting, linear algebra, and matrix multiplication.