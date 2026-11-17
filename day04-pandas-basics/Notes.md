# Day 4: Pandas Basics - Study Notes

> Part of my [Python Data Science Journey](../README.md). Code is in [`main.py`](main.py), project summary in [`README.md`](README.md).

## 1. What is Pandas?

Pandas is the main Python library for working with **tables** (rows and columns), like Excel or a database, but programmable.

- It is built **on top of NumPy**. Every column is a NumPy array with labels.
- NumPy is great for numbers only. Pandas handles **mixed data** (text, numbers, True/False, dates) in one table.
- It reads and writes CSV, Excel, JSON, and SQL.

```python
import pandas as pd
import numpy as np
```

| | NumPy | Pandas |
|--|-------|--------|
| Main object | `ndarray` | `Series` and `DataFrame` |
| Data types | One type per array | One type per column |
| Labels | Positions only (0, 1, 2) | Row labels and column names |
| Best for | Math, matrices | Real-world tabular data |

## 2. The two core objects

| Object | What it is | Analogy |
|--------|------------|---------|
| **Series** | One column with an index | One column of a spreadsheet |
| **DataFrame** | A table of Series sharing one index | A whole spreadsheet |

A column picked from a DataFrame is a Series: `type(df["math"])` gives `Series`.

## 3. The Series

```python
marks = pd.Series([88, 92, 79], index=["Ali", "Bao", "Chen"], name="math")
```

| Code | Meaning |
|------|---------|
| `marks.values` | The raw NumPy array inside |
| `marks.index` | The labels |
| `marks.loc["Bao"]` | Get by **label** |
| `marks.iloc[1]` | Get by **position** |
| `marks.idxmax()` | The **label** of the biggest value (NumPy's `argmax` gives a position) |
| `marks[marks > 80]` | Boolean filtering |
| `marks.sort_values()` | Sort |
| `marks.describe()` | Instant statistics |
| `series.value_counts()` | Count how often each value appears |

### Alignment (important and surprising)

Math between two Series matches values by **label**, not position. A label that exists in only one Series becomes **NaN**.

```python
s1 = pd.Series([1, 2, 3], index=["a", "b", "c"])
s2 = pd.Series([10, 20, 30], index=["b", "c", "d"])
s1 + s2      # a: NaN, b: 12, c: 23, d: NaN
```

**NaN** means "Not a Number", which is how Pandas shows missing data.

## 4. Creating a DataFrame

```python
# From a dictionary: each key is a COLUMN
pd.DataFrame({"name": ["Ali", "Bao"], "math": [88, 92]})

# From a list of dictionaries: each dictionary is a ROW
pd.DataFrame([{"name": "Dana", "math": 65}, {"name": "Eli", "math": 95}])

# From a NumPy array plus column names
pd.DataFrame(np.arange(6).reshape(3, 2), columns=["x", "y"])
```

### Attributes

| Attribute | Meaning |
|-----------|---------|
| `df.shape` | `(rows, columns)` |
| `df.columns` | Column names |
| `df.index` | Row labels (default `RangeIndex` 0, 1, 2, ...) |
| `df.dtypes` | The type of each column |

### Adding and removing columns

```python
df["double"] = df["math"] * 2          # new column from another column
df["bonus"] = 5                        # one value copied to every row
df = df.drop(columns=["double"])       # drop returns a NEW DataFrame
```

## 5. Reading a CSV and inspecting it

```python
df = pd.read_csv("data/students.csv")
```

**The first things I do with any new dataset:**

| Command | What it tells me |
|---------|------------------|
| `df.shape` | How many rows and columns |
| `df.head()` / `df.tail()` | First / last rows |
| `df.sample(3)` | Random rows (avoids a misleading sorted top) |
| `df.info()` | Column names, **non-null counts**, types |
| `df.describe()` | Statistics of the numeric columns |
| `df.isna().sum()` | Missing values per column |
| `df.duplicated().sum()` | Number of duplicate rows |
| `df["col"].nunique()` / `.unique()` | How many / which distinct values |
| `df["col"].value_counts()` | Counts (add `normalize=True` for fractions) |

**Reading `info()`:** if a column has fewer non-null values than the number of rows, it has missing data. In my dataset `attendance` had 57 of 60, so 3 values were missing.

**Reading `describe()`:** compare the mean and the median (50%). A big gap means extreme values. Check the `min` and `max` for impossible values (a score of 150, an age of -3).

### Useful `read_csv` options

```python
pd.read_csv(path, usecols=["name", "major"], nrows=4)   # only some columns and rows
pd.read_csv(path, index_col="student_id")               # use a column as the index
```

### Saving

```python
df.to_csv("results/out.csv", index=False)   # index=False avoids an extra column
```

## 6. Selecting data

### Columns

```python
df["name"]                    # one column -> Series
df[["name", "math"]]          # several columns -> DataFrame (DOUBLE brackets)
```

### Rows and columns: `loc` vs `iloc`

| | `loc` | `iloc` |
|--|-------|--------|
| Uses | **Labels** (names) | **Positions** (numbers) |
| Slice end | **Included** | **Excluded** |
| Example | `df.loc[0:2, ["name", "math"]]` gives **3 rows** | `df.iloc[0:2, 1:4]` gives **2 rows** |

**This difference is the most common source of bugs.** With `loc`, `0:2` means rows 0, 1 **and 2**. With `iloc`, `0:2` means rows 0 and 1 only.

```python
df.iloc[-1]            # last row (a Series)
df.iloc[::20, [0, 1]]  # every 20th row, two columns
df.at[0, "name"]       # one value by label (fast)
df.iat[0, 1]           # one value by position (fast)
```

### Using a column as the index

```python
by_name = df.set_index("name")
by_name.loc["Amina Ahmed", ["major", "math"]]
by_name.reset_index()          # turn the index back into a normal column
```

### Changing values: use `loc`

```python
df.loc[0, "math"] = 100
df.loc[df["math"] > 99, "math"] = 99     # condition on rows, column name
```

**Never** write `df[condition]["col"] = value`. It may silently change nothing (a chained assignment problem).

## 7. Filtering rows

A condition returns a Series of True/False (a **mask**, like NumPy Day 1). Putting it in `df[...]` keeps the True rows.

```python
df[df["math"] > 90]
```

### Combining conditions

| Operator | Meaning |
|----------|---------|
| `&` | AND |
| `\|` | OR |
| `~` | NOT |

```python
df[(df["major"] == "AI") & (df["programming"] >= 80) & df["scholarship"]]
df[(df["year"] == 1) | (df["year"] == 4)]
df[~df["scholarship"]]
```

**Always put each condition in parentheses.** Use `&` and `|`, not `and` and `or`.

### Shortcuts

| Code | Meaning |
|------|---------|
| `df["city"].isin(["Guilin", "Nanning"])` | Value is in a list |
| `df["physics"].between(40, 50)` | Inside a range (both ends included) |
| `df["name"].str.startswith("A")` | Text starts with |
| `df["name"].str.contains("ei", case=False)` | Text contains |
| `df["x"].isna()` / `.notna()` | Missing / not missing |
| `df.query("ai > @limit and major == 'AI'")` | Condition written as text (`@` uses a Python variable) |

### Counting with masks

`True` counts as 1 and `False` as 0, so:

```python
(df["math"] < 50).sum()                 # how many failed
df["scholarship"].mean() * 100          # percent with a scholarship
```

### `.copy()`

A filtered table is a new table. If I want to change it, I use `.copy()` first to avoid warnings and accidental changes:

```python
high = df[df["math"] > 80].copy()
high["note"] = "strong in math"
```

## 8. Creating columns, sorting, and ranking

```python
df["average"] = df[SUBJECTS].mean(axis=1)        # across the 5 subject columns
df["best_subject"] = df[SUBJECTS].idxmax(axis=1) # COLUMN NAME of each row's max
df["passed_all"] = (df[SUBJECTS] >= 50).all(axis=1)
df["grade"] = pd.cut(df["average"], bins=[0, 50, 60, 70, 80, 101],
                     labels=["F", "D", "C", "B", "A"], right=False)
df["rank"] = df["average"].rank(ascending=False, method="min").astype(int)
```

- `axis=1` works **across columns** (one result per row), exactly like NumPy
- `pd.cut` puts numbers into labeled ranges (the Pandas version of `np.digitize`)
- `right=False` means the left edge is included, so 80 is an "A"
- `rank(method="min")` gives tied students the same rank

### Sorting

```python
df.sort_values("average", ascending=False)
df.sort_values(["major", "average"], ascending=[True, False])   # two columns
df.nlargest(3, "average")                                       # shortcuts
df.nsmallest(3, "average")
```

### Renaming and `apply`

```python
df.rename(columns={"study_hours": "hours_per_week"})
df["name"].apply(len)       # run my own function on each value
```

`apply` is slower than built-in vectorized methods, so I only use it when no built-in exists.

## 9. Missing values

- A missing number appears as **NaN**
- `df.isna().sum()` counts missing values per column
- Pandas **skips NaN** in `mean()`, `sum()`, `count()` by default. NumPy would return `nan`.
- `df["attendance"].count()` gives the number of **non-missing** values (57 of 60 here)
- Day 5 covers how to fill or remove them

## 10. A first look at `groupby` (Day 6 preview)

```python
df.groupby("major")["average"].mean()
```

It means: **split** the table into groups, **calculate** something for each group, and **combine** the results.

## 11. Pandas versions: what changed

I ran this on **Pandas 3**, where text columns show the dtype `str`. In Pandas 2 the same columns show `object`. My code works in both because I avoid `describe(include="object")` and always use `.loc` for assignments.

## 12. Common mistakes

1. Forgetting that `loc` includes the slice end and `iloc` excludes it
2. Using `and` / `or` instead of `&` / `|`
3. Forgetting the parentheses around each condition
4. Using single brackets for several columns (`df["a", "b"]` is wrong, use `df[["a", "b"]]`)
5. Chained assignment: `df[mask]["col"] = x`
6. Forgetting `.copy()` after filtering when I plan to modify the result
7. Forgetting `index=False` in `to_csv` and getting an extra column
8. Assuming `drop` or `sort_values` change the original (they return a new DataFrame)
9. Not checking `info()` and `isna()` before analyzing
10. Treating NaN as zero. They are not the same.

## 13. Quick reference card

```python
df = pd.read_csv("file.csv")
df.shape; df.head(); df.info(); df.describe(); df.isna().sum()
df["a"]; df[["a", "b"]]
df.loc[rows, cols]; df.iloc[rows, cols]
df[df["a"] > 5]; df[(c1) & (c2)]; df[df["a"].isin([...])]
df.query("a > 5 and b == 'x'")
df["new"] = df["a"] * 2
df.sort_values("a", ascending=False); df.nlargest(3, "a")
df["a"].value_counts(); df.groupby("k")["a"].mean()
df.to_csv("out.csv", index=False)
```

## 14. Pandas vs the NumPy project from Day 3

The same questions needed far less code in Pandas:

| Task | NumPy (Day 3) | Pandas (Day 4) |
|------|---------------|----------------|
| Student average | `scores.mean(axis=1)` | `df[SUBJECTS].mean(axis=1)` (and the result keeps the student labels) |
| Ranking | `argsort` plus an inverse trick | `rank(ascending=False)` |
| Letter grades | `np.digitize` plus an index | `pd.cut(..., labels=...)` |
| Names | A separate file | Same table as the scores |
| Missing values | Manual handling | Built-in (`isna`, skipped automatically) |

## 15. My own words

*(Fill this in yourself. This is the part employers actually read.)*

- **The difference between a Series and a DataFrame, in my words:**
- **How `loc` and `iloc` differ:**
- **The most useful command I learned today:**
- **What confused me:**
- **One question about the dataset I answered myself:**

## Resources used

- Pandas official docs: "10 minutes to pandas" and "Intro to data structures"
- Kaggle's free "Pandas" micro-course
- Add the videos or tutorials you actually used here

---

**Next:** Day 5 is **data cleaning**: missing values (`fillna`, `dropna`), duplicates, wrong types, and text cleanup on a messy dataset.