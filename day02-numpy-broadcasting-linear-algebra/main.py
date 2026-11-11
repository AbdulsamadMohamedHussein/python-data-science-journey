"""
DAY 2: NUMPY BROADCASTING, VECTORIZATION, RANDOM AND LINEAR ALGEBRA
Run:  .\\venv\\Scripts\\python.exe day02-numpy-broadcasting-linear-algebra\\main.py

Read each section, run it, then CHANGE the numbers and see what happens.
Section 13 at the bottom is for YOU to practice.
"""

import time
import numpy as np


def section(title):
    """Prints a nice header so the output is easy to read."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# =====================================================================
# 1. BROADCASTING: THE BIG IDEA
# =====================================================================
# Broadcasting lets NumPy do math between arrays of DIFFERENT shapes
# by "stretching" the smaller one to match the bigger one.
# Nothing is really copied in memory, so it is fast and efficient.
#
# You already used the simplest case on Day 1: array * 10
# (the single number 10 is stretched to every element).

section("1. BROADCASTING BASICS")

a = np.array([[1, 2, 3],
              [4, 5, 6]])            # shape (2, 3)

print("a + 10 (scalar to every element):\n", a + 10)

row = np.array([10, 20, 30])         # shape (3,)
print("a + row (row added to EACH row of a):\n", a + row)

col = np.array([[100],
                [200]])              # shape (2, 1)
print("a + col (column added to EACH column of a):\n", a + col)


# =====================================================================
# 2. THE BROADCASTING RULES
# =====================================================================
# NumPy compares the shapes starting from the RIGHT-most dimension.
# Two dimensions are compatible when:
#   (1) they are equal, OR
#   (2) one of them is 1
# Missing dimensions on the left are treated as 1.
#
#   (2, 3) + (3,)    -> OK   (3 matches 3, the missing dim becomes 1)
#   (2, 3) + (2, 1)  -> OK   (1 stretches to 3)
#   (2, 3) + (2,)    -> ERROR (3 vs 2: not equal, neither is 1)
#   (3, 1) + (1, 4)  -> OK   result shape (3, 4)

section("2. BROADCASTING RULES (AND AN ERROR)")

print("(2,3) + (3,) works:", (a + row).shape)

bad = np.array([1, 2])               # shape (2,)
try:
    print(a + bad)
except ValueError as error:
    print("(2,3) + (2,) FAILS:", error)

# Fix: turn the (2,) array into a column (2, 1) so it broadcasts down
print("fixed with reshape(-1, 1):\n", a + bad.reshape(-1, 1))

# Column + row = a full grid (like a multiplication table!)
c = np.array([[1], [2], [3]])        # (3, 1)
r = np.array([1, 2, 3, 4])           # (4,)  -> acts like (1, 4)
print("(3,1) * (4,) = (3,4) table:\n", c * r)


# =====================================================================
# 3. np.newaxis: ADD A DIMENSION ON PURPOSE
# =====================================================================
# np.newaxis (same as None) inserts a new axis of size 1.
# It is a shortcut for reshape when you want to control broadcasting.

section("3. np.newaxis")

v = np.array([1, 2, 3])
print("v shape          :", v.shape)               # (3,)
print("v[:, np.newaxis] :", v[:, np.newaxis].shape)  # (3, 1) column
print("v[np.newaxis, :] :", v[np.newaxis, :].shape)  # (1, 3) row

# Subtract every pair: differences between all elements of v
print("all pairwise differences:\n", v[:, np.newaxis] - v[np.newaxis, :])


# =====================================================================
# 4. BROADCASTING IN REAL LIFE: NORMALIZING DATA
# =====================================================================
# In machine learning we often rescale each COLUMN (feature).
# Broadcasting does this without a single loop.

section("4. NORMALIZATION WITH BROADCASTING")

# 5 students, 3 features: hours studied, past score, sleep hours
data = np.array([[2.0, 50.0, 6.0],
                 [4.0, 60.0, 7.0],
                 [6.0, 70.0, 8.0],
                 [8.0, 80.0, 5.0],
                 [10.0, 90.0, 9.0]])
print("data shape:", data.shape)

# Min-max scaling: squeeze every column into the range 0 to 1
col_min = data.min(axis=0)           # shape (3,)
col_max = data.max(axis=0)           # shape (3,)
scaled = (data - col_min) / (col_max - col_min)
print("min-max scaled:\n", scaled.round(2))

# Standardization (z-score): mean 0, standard deviation 1
mean = data.mean(axis=0)
std = data.std(axis=0)
z = (data - mean) / std
print("z-score:\n", z.round(2))
print("column means after z-score ~ 0:", z.mean(axis=0).round(6))
print("column stds  after z-score = 1:", z.std(axis=0).round(6))

# Centering: subtract the mean of each ROW (needs keepdims to broadcast)
row_means = data.mean(axis=1, keepdims=True)   # shape (5, 1), not (5,)
print("row mean shape with keepdims:", row_means.shape)
print("centered rows:\n", (data - row_means).round(2))
# TIP: keepdims=True keeps the reduced axis as size 1, so broadcasting works.


# =====================================================================
# 5. VECTORIZATION vs LOOPS (SPEED)
# =====================================================================
# Vectorization = replace Python for-loops with NumPy operations.
# Rule of thumb: if you write a for loop over array elements, look for a
# NumPy function instead.

section("5. VECTORIZATION vs LOOPS")

rng = np.random.default_rng(42)      # modern way to make random numbers
x = rng.random(2_000_000)

start = time.time()
total = 0.0
for value in x:
    total += value * 2 + 1
loop_time = time.time() - start

start = time.time()
total_np = (x * 2 + 1).sum()
numpy_time = time.time() - start

print(f"loop result  : {total:.2f}  time: {loop_time:.3f} s")
print(f"numpy result : {total_np:.2f}  time: {numpy_time:.5f} s")
print(f"NumPy is about {loop_time / max(numpy_time, 1e-9):.0f}x faster")

# np.where instead of if/else in a loop
temps = np.array([12, 30, 25, 8, 35, 19])
feels = np.where(temps > 25, "hot", np.where(temps < 15, "cold", "nice"))
print("temps :", temps)
print("labels:", feels)

# np.clip limits values to a range
print("clip to [10, 30]:", np.clip(temps, 10, 30))


# =====================================================================
# 6. DISTANCE MATRIX (BROADCASTING + VECTORIZATION TOGETHER)
# =====================================================================
# Goal: distance between every pair of 2D points, with NO loops.
# This idea is used in K-Nearest Neighbors and clustering.

section("6. DISTANCE BETWEEN ALL POINT PAIRS")

points = np.array([[0, 0],
                   [3, 4],
                   [6, 8],
                   [1, 1]], dtype=float)        # shape (4, 2)

# (4,1,2) - (1,4,2) -> (4,4,2): the difference for every pair
diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
dist = np.sqrt((diff ** 2).sum(axis=2))         # (4, 4)
print("distance matrix:\n", dist.round(2))
print("distance between point 0 and point 1:", dist[0, 1])   # 5.0 (3-4-5)


# =====================================================================
# 7. RANDOM NUMBERS (np.random)
# =====================================================================
# Always set a seed so your results are repeatable.
# default_rng is the recommended modern API.

section("7. RANDOM NUMBERS")

rng = np.random.default_rng(seed=7)

print("random floats [0,1) :", rng.random(3).round(3))
print("random ints 1-6 (dice):", rng.integers(1, 7, size=5))
print("normal (mean 0, std 1):", rng.normal(0, 1, size=3).round(3))
print("normal (mean 70, std 10), 4 values:", rng.normal(70, 10, 4).round(1))
print("random 2x3 ints:\n", rng.integers(0, 100, size=(2, 3)))

items = np.array(["a", "b", "c", "d", "e"])
print("choice of 3, no repeats:", rng.choice(items, size=3, replace=False))
print("shuffled copy:", rng.permutation(items))

# Law of large numbers: the mean of many dice rolls gets close to 3.5
rolls = rng.integers(1, 7, size=100_000)
print("mean of 100,000 dice rolls (should be ~3.5):", rolls.mean().round(3))

# Quick fake dataset: heights with a normal distribution
heights = rng.normal(170, 8, size=1000)
print(f"fake heights: mean={heights.mean():.1f}, std={heights.std():.1f}")


# =====================================================================
# 8. DOT PRODUCT AND MATRIX MULTIPLICATION
# =====================================================================
# Element-wise  A * B  : multiply matching positions (shapes must broadcast)
# Matrix mult.  A @ B  : rows of A times columns of B (row-by-column)
#
# Matrix rule: (m, n) @ (n, k) -> (m, k)
# The INNER numbers must match. The OUTER numbers give the result shape.

section("8. DOT PRODUCT AND MATRIX MULTIPLICATION")

u = np.array([1, 2, 3])
w = np.array([4, 5, 6])

print("element-wise u * w:", u * w)                 # [4 10 18]
print("dot product u @ w :", u @ w)                 # 1*4 + 2*5 + 3*6 = 32
print("np.dot(u, w)      :", np.dot(u, w))          # same thing

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print("A * B (element-wise):\n", A * B)
print("A @ B (matrix mult.):\n", A @ B)
# First cell of A @ B: row 0 of A . column 0 of B = 1*5 + 2*7 = 19

# Order matters for matrices!
print("A @ B equals B @ A?", np.array_equal(A @ B, B @ A))   # False

# Shapes
X = np.ones((3, 4))
Y = np.ones((4, 2))
print("(3,4) @ (4,2) ->", (X @ Y).shape)            # (3, 2)
try:
    X @ X
except ValueError:
    print("(3,4) @ (3,4) fails: inner numbers 4 and 3 do not match")

# Matrix times vector
M = np.array([[1, 0, 2],
              [0, 1, 3]])
vec = np.array([1, 2, 3])
print("M @ vec:", M @ vec)                          # [7 11]


# =====================================================================
# 9. LINEAR ALGEBRA TOOLS (np.linalg)
# =====================================================================
section("9. LINEAR ALGEBRA TOOLS")

A = np.array([[2.0, 1.0],
              [1.0, 3.0]])

print("transpose A.T:\n", A.T)
print("trace (sum of diagonal):", np.trace(A))
print("determinant:", round(np.linalg.det(A), 2))        # 2*3 - 1*1 = 5
print("rank:", np.linalg.matrix_rank(A))

A_inv = np.linalg.inv(A)
print("inverse:\n", A_inv.round(3))
print("A @ A_inv is the identity:\n", (A @ A_inv).round(3))
# np.allclose compares floats safely (never use == for decimals)
print("allclose to identity?", np.allclose(A @ A_inv, np.eye(2)))

# Solving a system of equations:
#   2x +  y = 5
#    x + 3y = 10
b = np.array([5.0, 10.0])
solution = np.linalg.solve(A, b)
print("solution [x, y]:", solution)                      # [1. 3.]
print("check A @ solution:", A @ solution)

# Vector length (norm) and distance
vec = np.array([3.0, 4.0])
print("norm of [3,4]:", np.linalg.norm(vec))             # 5.0
print("distance between points:", np.linalg.norm(np.array([1, 1]) - np.array([4, 5])))


# =====================================================================
# 10. EIGENVALUES (A FIRST LOOK)
# =====================================================================
# An eigenvector of A keeps its direction when multiplied by A. It only
# gets stretched by the eigenvalue: A @ v = lambda * v
# Used in PCA, a very important AI technique. Just see it for now.

section("10. EIGENVALUES AND EIGENVECTORS")

S = np.array([[4.0, 1.0],
              [2.0, 3.0]])
values, vectors = np.linalg.eig(S)
values, vectors = values.real, vectors.real   # drop the harmless +0j parts
print("eigenvalues:", values.round(3))
first = vectors[:, 0]
print("S @ v      :", (S @ first).round(3))
print("lambda * v :", (values[0] * first).round(3))      # same numbers


# =====================================================================
# 11. AI MINI EXAMPLES USING MATRIX MATH
# =====================================================================
section("11. AI MINI EXAMPLES")

# (a) Cosine similarity: how similar are two vectors? (1 = same direction)
p = np.array([1, 2, 3])
q = np.array([4, 5, 6])
cosine = (p @ q) / (np.linalg.norm(p) * np.linalg.norm(q))
print("cosine similarity:", round(cosine, 4))

# (b) One layer of a neural network: output = ReLU(X @ W + b)
rng = np.random.default_rng(1)
X = rng.random((4, 3))               # 4 samples, 3 features
W = rng.random((3, 2))               # 3 inputs -> 2 neurons
b = np.array([0.1, -0.5])            # one bias per neuron (broadcasts!)
out = X @ W + b                      # (4,3)@(3,2) -> (4,2), then + (2,)
activated = np.maximum(0, out)       # ReLU: negatives become 0
print("layer output shape:", out.shape)
print("after ReLU:\n", activated.round(3))

# (c) Linear regression with matrices. Data follows y = 2x + 1
xs = np.array([1, 2, 3, 4, 5], dtype=float)
ys = np.array([3, 5, 7, 9, 11], dtype=float)
design = np.column_stack([np.ones_like(xs), xs])        # [1, x] columns
weights, *_ = np.linalg.lstsq(design, ys, rcond=None)
print("intercept and slope:", weights.round(3))          # [1. 2.]
print("prediction for x = 10:", round(weights[0] + weights[1] * 10, 2))


# =====================================================================
# 12. COMMON MISTAKES (READ THESE)
# =====================================================================
# 1. Using * when you want matrix multiplication (use @)
# 2. Forgetting that (3,) and (3,1) are DIFFERENT shapes
# 3. Comparing floats with == (use np.allclose)
# 4. Forgetting keepdims=True when subtracting row statistics
# 5. Looping over array elements when a NumPy function exists
# 6. Not setting a random seed, so results change every run
# 7. Thinking A @ B equals B @ A (order matters)


# =====================================================================
# 13. YOUR PRACTICE (write your own code below)
# =====================================================================
section("13. YOUR PRACTICE")

# Exercise A: Create a (4, 3) array of numbers 1 to 12. Add the row
#             [10, 20, 30] to every row WITHOUT a loop.
pass

# Exercise B: Using a (5,1) column and a (1,5) row, build a 5x5 table
#             where each cell = row_number + column_number.
pass

# Exercise C: Make a (6, 3) array of random ints 0-100 (use a seed).
#             Standardize each column (z-score). Check the means ~ 0.
pass

# Exercise D: Speed test. Compute the sum of squares of 1,000,000
#             numbers with a for loop and with NumPy. Print both times.
pass

# Exercise E: Multiply A = [[1,2,3],[4,5,6]] by B = [[1,2],[3,4],[5,6]].
#             Predict the result shape BEFORE running it.
pass

# Exercise F: Solve this system with np.linalg.solve:
#                 3x + 2y = 12
#                  x -  y = 1
#             Check your answer with A @ solution.
pass

# Exercise G (challenge): Simulate rolling two dice 100,000 times.
#             What is the probability that the sum is 7? (about 1/6)
pass

print("\nDone! Now fill your README, update notes.md and push to GitHub.")