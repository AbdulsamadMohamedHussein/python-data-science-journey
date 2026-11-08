"""
DAY 1: NUMPY BASICS - STUDY NOTES + RUNNABLE CODE
Run:  .\\venv\\Scripts\\python.exe day01-numpy-basics\\main.py

Read each section, run it, then CHANGE the numbers and see what happens.
That is how you really learn. Section 12 at the bottom is for YOU to practice.
"""

import time
import numpy as np


def section(title):
    """Prints a nice header so the output is easy to read."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# =====================================================================
# 1. WHAT IS NUMPY?
# =====================================================================
# NumPy = "Numerical Python". Its main object is the ndarray
# (n-dimensional array): a grid of values that ALL have the same type.
#
# Why not just use Python lists?
#   - NumPy is MUCH faster (it is written in C and stores data compactly)
#   - You can do math on the whole array at once (no for loops)
#   - Pandas, Matplotlib, scikit-learn and PyTorch are all built on it
# So NumPy is the foundation of everything we do in AI.

section("1. LIST vs NUMPY ARRAY")

py_list = [1, 2, 3, 4]
np_array = np.array([1, 2, 3, 4])

print("list * 2   :", py_list * 2)    # repeats the list: [1,2,3,4,1,2,3,4]
print("array * 2  :", np_array * 2)   # math on every element: [2 4 6 8]
# KEY IDEA: on a list, * means "repeat". On an array, * means "multiply".

# Speed test: sum of squares for 1 million numbers
n = 1_000_000
big_list = list(range(n))
big_array = np.arange(n)

start = time.time()
result_list = [x * x for x in big_list]
list_time = time.time() - start

start = time.time()
result_array = big_array * big_array
array_time = time.time() - start

print(f"Python list time : {list_time:.4f} s")
print(f"NumPy array time : {array_time:.4f} s")
print(f"NumPy is about {list_time / max(array_time, 1e-9):.0f}x faster")


# =====================================================================
# 2. ARRAY DIMENSIONS: 1D, 2D, 3D
# =====================================================================
# 1D = a row of numbers (vector)
# 2D = rows and columns (matrix, like a spreadsheet or a grayscale image)
# 3D = a stack of matrices (like a color image: height x width x channels)

section("2. 1D, 2D AND 3D ARRAYS")

a1 = np.array([1, 2, 3])                        # 1D
a2 = np.array([[1, 2, 3], [4, 5, 6]])           # 2D (2 rows, 3 columns)
a3 = np.array([[[1, 2], [3, 4]],
               [[5, 6], [7, 8]]])               # 3D (2 blocks of 2x2)

print("1D:\n", a1)
print("2D:\n", a2)
print("3D:\n", a3)


# =====================================================================
# 3. ARRAY ATTRIBUTES (information about an array)
# =====================================================================
#   .ndim  -> number of dimensions (axes)
#   .shape -> size of each dimension, as a tuple (rows, columns)
#   .size  -> total number of elements
#   .dtype -> data type of the elements
#   .itemsize -> bytes used by ONE element

section("3. ARRAY ATTRIBUTES")

print("a2 =\n", a2)
print("ndim     :", a2.ndim)        # 2
print("shape    :", a2.shape)       # (2, 3)
print("size     :", a2.size)        # 6
print("dtype    :", a2.dtype)       # int64 (int32 on some Windows setups)
print("itemsize :", a2.itemsize)   # bytes per element
# TIP: shape (2, 3) means 2 rows and 3 columns. size = 2 * 3 = 6.


# =====================================================================
# 4. DATA TYPES (dtype)
# =====================================================================
# Every element has the same dtype. Common ones:
#   int32, int64      whole numbers
#   float32, float64  decimal numbers
#   bool              True / False
#   str_ (U)          text
# If you mix ints and floats, NumPy upgrades everything to float.

section("4. DATA TYPES")

ints = np.array([1, 2, 3])
mixed = np.array([1, 2.5, 3])          # becomes float automatically
bools = np.array([True, False, True])
forced = np.array([1, 2, 3], dtype=np.float32)   # you choose the type

print("ints   :", ints, ints.dtype)
print("mixed  :", mixed, mixed.dtype)
print("bools  :", bools, bools.dtype)
print("forced :", forced, forced.dtype)

# astype() converts and returns a NEW array
floats = np.array([1.9, 2.7, 3.1])
print("float -> int (cuts the decimals, no rounding!):", floats.astype(int))


# =====================================================================
# 5. CREATING ARRAYS
# =====================================================================
section("5. CREATING ARRAYS")

print("zeros((2,3)):\n", np.zeros((2, 3)))          # all 0.0
print("ones((2,2)):\n", np.ones((2, 2)))            # all 1.0
print("full((2,3), 7):\n", np.full((2, 3), 7))      # all 7
print("eye(3) identity matrix:\n", np.eye(3))       # 1s on the diagonal
print("empty((2,2)) is random garbage, do not trust it:\n", np.empty((2, 2)))

# arange(start, stop, step): like range(), stop is NOT included
print("arange(0, 10, 2):", np.arange(0, 10, 2))     # [0 2 4 6 8]
print("arange(5)       :", np.arange(5))            # [0 1 2 3 4]

# linspace(start, stop, count): stop IS included, evenly spaced
print("linspace(0, 1, 5):", np.linspace(0, 1, 5))   # [0. 0.25 0.5 0.75 1.]

# arange = you control the STEP. linspace = you control the COUNT.

# Random numbers (very common in AI)
np.random.seed(42)    # seed makes results repeatable
print("random floats 0-1 (2x3):\n", np.random.rand(2, 3))
print("random ints 0-9 (5 numbers):", np.random.randint(0, 10, 5))
print("normal distribution (3 numbers):", np.random.randn(3))


# =====================================================================
# 6. INDEXING (getting one element)
# =====================================================================
# Counting starts at 0. Negative index counts from the END (-1 = last).
# 2D syntax: array[row, column]

section("6. INDEXING")

arr = np.array([10, 20, 30, 40, 50])
print("arr[0]  :", arr[0])     # 10 (first)
print("arr[2]  :", arr[2])     # 30
print("arr[-1] :", arr[-1])    # 50 (last)

m = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print("m[0, 0] :", m[0, 0])    # 1  (row 0, col 0)
print("m[1, 2] :", m[1, 2])    # 6  (row 1, col 2)
print("m[-1, -1]:", m[-1, -1]) # 9  (last row, last col)

m[0, 0] = 100                  # assigning a new value
print("after m[0,0] = 100:\n", m)
m[0, 0] = 1                    # put it back


# =====================================================================
# 7. SLICING (getting a part)
# =====================================================================
# Syntax: array[start:stop:step]   (stop is NOT included)
# For 2D:  array[row_slice, column_slice]
# A lone ":" means "everything along this axis".

section("7. SLICING")

x = np.arange(1, 11)           # [1 2 3 4 5 6 7 8 9 10]
print("x             :", x)
print("x[:3]  first 3:", x[:3])
print("x[-3:] last 3 :", x[-3:])
print("x[2:6]        :", x[2:6])
print("x[::2] every 2nd:", x[::2])
print("x[::-1] reversed:", x[::-1])

print("m =\n", m)
print("m[0]      first row     :", m[0])
print("m[:, 1]   second column :", m[:, 1])
print("m[:2, :2] top-left 2x2  :\n", m[:2, :2])
print("m[1:, 1:] bottom-right  :\n", m[1:, 1:])

# IMPORTANT: a slice is a VIEW, not a copy. Changing it changes the original!
original = np.array([1, 2, 3, 4, 5])
view = original[1:4]
view[0] = 999
print("original changed by the view:", original)

# Use .copy() when you want an independent array
original = np.array([1, 2, 3, 4, 5])
safe = original[1:4].copy()
safe[0] = 999
print("original untouched with copy:", original)


# =====================================================================
# 8. RESHAPING
# =====================================================================
# reshape changes the layout, NOT the data. The total number of elements
# must stay the same. Use -1 to let NumPy work out one dimension.

section("8. RESHAPING")

r = np.arange(12)
print("original (12,):", r)
print("reshape(3, 4):\n", r.reshape(3, 4))
print("reshape(2, 6):\n", r.reshape(2, 6))
print("reshape(4, -1) (NumPy finds 3):\n", r.reshape(4, -1))

grid = r.reshape(3, 4)
print("flatten  :", grid.flatten())            # back to 1D (a copy)
print("ravel    :", grid.ravel())              # back to 1D (a view when possible)
print("transpose shape:", grid.T.shape)        # (4, 3): rows <-> columns

# Turning a 1D array into a column or row vector (needed in ML a lot)
v = np.array([1, 2, 3])
print("column vector shape:", v.reshape(-1, 1).shape)   # (3, 1)
print("row vector shape   :", v.reshape(1, -1).shape)   # (1, 3)


# =====================================================================
# 9. MATH OPERATIONS (vectorization)
# =====================================================================
# Operations apply element by element, with NO loops. This is called
# VECTORIZATION and it is the main reason NumPy is fast.

section("9. MATH OPERATIONS")

p = np.array([1, 2, 3, 4])
q = np.array([10, 20, 30, 40])

print("p + q :", p + q)
print("q - p :", q - p)
print("p * q :", p * q)       # element-wise, NOT matrix multiplication
print("q / p :", q / p)
print("p ** 2:", p ** 2)
print("p * 10:", p * 10)      # a single number is applied to every element
print("sqrt  :", np.sqrt(p))
print("exp   :", np.exp(p))
print("log   :", np.log(p))

# Comparison also works element-wise and returns booleans
print("p > 2 :", p > 2)       # [False False  True  True]


# =====================================================================
# 10. STATISTICS AND THE axis PARAMETER
# =====================================================================
# axis=0 -> go DOWN the rows    (result: one value per COLUMN)
# axis=1 -> go ACROSS the columns (result: one value per ROW)
# No axis -> use every element

section("10. STATISTICS AND AXIS")

scores = np.array([[80, 90, 70],
                   [60, 85, 95],
                   [75, 65, 88]])
print("scores =\n", scores)

print("sum all     :", scores.sum())
print("mean all    :", scores.mean())
print("min / max   :", scores.min(), "/", scores.max())
print("std         :", round(scores.std(), 2))
print("median      :", np.median(scores))

print("sum axis=0 (per column):", scores.sum(axis=0))    # [215 240 253]
print("sum axis=1 (per row)   :", scores.sum(axis=1))    # [240 240 228]
print("mean axis=1            :", scores.mean(axis=1))

print("argmax (position of the biggest, flattened):", scores.argmax())
print("argmin (position of the smallest)           :", scores.argmin())
print("best score in each row (argmax axis=1)      :", scores.argmax(axis=1))
print("cumulative sum:", np.cumsum([1, 2, 3, 4]))        # [1 3 6 10]


# =====================================================================
# 11. BOOLEAN MASKING AND USEFUL TOOLS
# =====================================================================
# A mask is an array of True/False. Put it inside [] to FILTER the data.
# This idea becomes filtering rows in Pandas on Day 4.

section("11. BOOLEAN MASKING AND TOOLS")

marks = np.array([45, 82, 67, 90, 38, 71])

mask = marks >= 50
print("mask          :", mask)
print("passed        :", marks[mask])
print("failed        :", marks[~mask])                    # ~ means NOT
print("between 60-80 :", marks[(marks >= 60) & (marks <= 80)])   # & = and
print("45 or 90      :", marks[(marks == 45) | (marks == 90)])    # | = or
# Use & and |, NOT the words and / or, and keep the parentheses!

# np.where(condition, value_if_true, value_if_false)
labels = np.where(marks >= 50, "pass", "fail")
print("labels        :", labels)

# Changing values with a mask
curved = marks.copy()
curved[curved < 50] = 50
print("minimum 50    :", curved)

# Sorting, unique values, joining, splitting
print("sort          :", np.sort(marks))
print("argsort (positions that would sort it):", np.argsort(marks))
print("unique        :", np.unique([3, 1, 3, 2, 1]))
print("concatenate   :", np.concatenate([[1, 2], [3, 4]]))
print("vstack:\n", np.vstack([[1, 2], [3, 4]]))           # stack as rows
print("hstack:", np.hstack([[1, 2], [3, 4]]))             # side by side
print("any > 80:", (marks > 80).any(), "| all > 30:", (marks > 30).all())
print("count of passes:", (marks >= 50).sum())            # True counts as 1


# =====================================================================
# 12. YOUR PRACTICE (write your own code below)
# =====================================================================
# Delete the "pass" and write the code. Run often. Google is allowed!

section("12. YOUR PRACTICE")

# Exercise A: create an array of the numbers 1 to 20 and print only the
#             odd numbers using a boolean mask.
pass

# Exercise B: create a 5x5 multiplication table (cell = row * column,
#             starting from 1). Hint: np.arange(1, 6) and reshape.
pass

# Exercise C: make a 4x4 array of random integers from 1 to 100. Print
#             the max of each column and the mean of each row.
pass

# Exercise D: take [5, 12, 7, 20, 3, 15]. Replace every number bigger
#             than 10 with 10 (hint: np.where or a mask). Print the result.
pass

# Exercise E (challenge): normalize the array [10, 20, 30, 40, 50] so it
#             goes from 0 to 1 using (x - min) / (max - min).
pass

print("\nDone! Now write your README notes and push to GitHub.")