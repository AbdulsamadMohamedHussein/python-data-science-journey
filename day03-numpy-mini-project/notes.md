# Day 3: NumPy Mini Project - Student Grade Analyzer (Study Notes)

> Part of my [Python Data Science Journey](../README.md). The code is in [`main.py`](main.py) and the project summary is in [`README.md`](README.md).

## 1. Why a mini project?

Days 1 and 2 taught individual tools. A project forces me to **combine** them to answer real questions. The goal today was to think like a data analyst:

1. Load the data
2. **Look at it first** (shape, types, missing values, range)
3. Ask questions and answer each one with NumPy
4. Save the results

## 2. The data

- A 2D array of shape **(30, 5)**
- **Rows = students**, **columns = subjects**
- `axis=0` goes down the rows, giving **one value per subject**
- `axis=1` goes across the columns, giving **one value per student**

```python
scores.mean(axis=0)   # average of each SUBJECT (5 values)
scores.mean(axis=1)   # average of each STUDENT (30 values)
```

**Memory trick:** the axis you name is the one that **disappears**. `(30, 5)` with `axis=0` leaves `(5,)`. With `axis=1` it leaves `(30,)`.

## 3. How the program is organized

| Part | Purpose |
|------|---------|
| Constants (`SUBJECTS`, `CREDITS`, `PASS_MARK`) | Settings in CAPITALS, easy to change in one place |
| `generate_dataset()` | Makes fake grades with a seed |
| `ensure_data()` / `load_data()` | Saves and loads the CSV |
| Small functions (`subject_stats`, `rank_students`, `letter_grades`) | Each does one job |
| `main()` | Runs the analysis in order |
| `if __name__ == "__main__":` | Runs `main()` only when the file is run directly |

Splitting code into small functions makes it easier to read, test, and reuse.

## 4. Files with `pathlib`

```python
BASE = Path(__file__).parent       # the folder containing main.py
DATA_DIR = BASE / "data"           # the / operator joins paths
DATA_DIR.mkdir(exist_ok=True)      # create the folder if missing
```

Using `Path(__file__).parent` means the project works no matter which folder I run it from.

## 5. Saving and loading CSV files

```python
np.savetxt("grades.csv", scores, delimiter=",",
           header="Math,Physics,...", comments="", fmt="%.1f")

scores = np.loadtxt("grades.csv", delimiter=",", skiprows=1)
```

- `delimiter=","` separates the columns
- `header` writes a first line with column names
- `comments=""` stops NumPy adding a `#` before the header
- `skiprows=1` skips that header line when loading
- `fmt="%.1f"` writes one decimal place

`loadtxt` only reads numbers, so the student names are in a separate `students.txt`. **Pandas (Day 4) handles mixed text and numbers in one file much better.**

## 6. Generating realistic data with broadcasting

```python
ability = rng.normal(0, 8, size=(30, 1))                  # one value per student
scores = rng.normal(means, stds, size=(30, 5)) + ability  # (30,5) + (30,1)
scores = np.clip(scores, 0, 100).round(1)
```

- `means` and `stds` have shape `(5,)`, so they broadcast across all 30 rows. Each subject gets its own average and spread.
- `ability` has shape `(30, 1)` and is added to every subject of the same student, so good students tend to be good everywhere.
- `np.clip` keeps scores between 0 and 100. Because of clipping, several students got exactly 100.

## 7. Statistics and finding "who"

| Question | Code |
|----------|------|
| Which subject is hardest? | `SUBJECTS[stats["mean"].argmin()]` |
| Who scored highest in each subject? | `scores.argmax(axis=0)` gives a row index per column |
| Highest weighted student? | `names[weighted.argmax()]` |

`argmax` and `argmin` return **positions**. I use the position to look up the name.

**Mean vs median:** if they are very different, the data has extreme values. Here they were close, so the data is fairly balanced.

**Standard deviation** shows how spread out scores are. A high std (Math) means very different results between students.

## 8. Weighted average with matrix multiplication

Subjects with more credit hours should count more.

```python
weighted = scores @ CREDITS / CREDITS.sum()
```

Shape check: `(30, 5) @ (5,)` gives `(30,)`. Each student's weighted average is the sum of `score x credits`, divided by the total credits. This is exactly the matrix-vector multiplication from Day 2.

## 9. Ranking (the trickiest part)

```python
order = np.argsort(-avg, kind="stable")
ranks = np.empty_like(order)
ranks[order] = np.arange(1, len(avg) + 1)
```

- `argsort` returns the **positions that would sort the array**, smallest first.
- The **minus sign** flips the values, so the highest average comes first.
- `order[0]` is the position of the best student, `order[1]` is second, and so on.
- `ranks[order] = 1, 2, 3...` writes each student's rank into their own position. So `ranks[i]` answers "what rank is student i?".
- `kind="stable"` keeps the original order when there are ties.

**Two directions to remember:**
- `order` answers "who is in position 1?"
- `ranks` answers "what position is this student in?"

## 10. Letter grades with `np.digitize`

```python
edges = [50, 60, 70, 80]
letters = np.array(["F", "D", "C", "B", "A"])
grades = letters[np.digitize(avg, edges)]
```

`digitize` returns which bucket each value falls into: below 50 gives 0, 50 to 59.99 gives 1, and so on up to 4. Using that number as an **index** into `letters` converts it to a grade.

## 11. Boolean masks for pass/fail analysis

```python
passed = scores >= PASS_MARK        # (30, 5) grid of True/False
(~passed).sum(axis=0)               # failures per subject
(~passed).sum(axis=1)               # failures per student
passed.all(axis=1)                  # passed EVERY subject?
passed[:, j].mean() * 100           # pass rate in %
```

Key ideas:

- `True` counts as 1 and `False` as 0, so `.sum()` counts and `.mean()` gives a **fraction**
- `~` means NOT, `&` means AND, `|` means OR
- Each condition needs **parentheses**: `(weighted >= 75) & all_pass`
- `.all(axis=1)` is True only if every subject in that row is True

```python
honors = (weighted >= 75) & all_pass
at_risk = (weighted < 55) | (fails_per_student >= 2)
names[honors]      # use a mask to pick names
```

## 12. Outliers with z-scores

```python
z = (scores - scores.mean(axis=0)) / scores.std(axis=0)
rows, cols = np.where(np.abs(z) > 2)
```

- The z-score says **how many standard deviations** a value is from the subject mean
- `(30, 5) - (5,)` is **broadcasting**: the 5 column means are subtracted from every row
- `np.where(condition)` on a 2D array returns **two arrays**: the row numbers and column numbers of every `True`
- A value beyond 2 standard deviations is unusual

## 13. Correlation between subjects

```python
corr = np.corrcoef(scores, rowvar=False)       # (5, 5)
upper = np.triu(corr, k=1)
a, b = np.unravel_index(np.argmax(np.abs(upper)), upper.shape)
```

- **Correlation (r)** is between -1 and +1. Near +1 means students good at one subject are good at the other. Near 0 means no relationship. Negative means the opposite.
- `rowvar=False` tells NumPy that each **column** is a variable (otherwise it treats rows as variables).
- The matrix is symmetric and its diagonal is always 1, so `np.triu(k=1)` keeps only the upper triangle without the diagonal. That avoids picking the same pair twice or a subject with itself.
- `argmax` on a 2D array gives a **flat** position. `np.unravel_index` converts it back into `(row, column)`.
- **Correlation is not causation.** A strong r does not mean one subject causes the other.

## 14. What-if analysis

```python
curved = scores.copy()                      # never edit the original
curved[:, col] = np.clip(curved[:, col] + 5, 0, 100)
```

`.copy()` is essential. Without it, `curved` would be a reference to the same data and the original scores would change too. Always copy before simulating changes.

## 15. Histogram and text bars

```python
counts, _ = np.histogram(weighted, bins=[0, 50, 60, 70, 80, 90, 100])
"#" * count
```

`np.histogram` counts how many values fall in each range. Real charts come on **Day 8 with Matplotlib**.

## 16. Saving results

```python
summary = np.column_stack([ranks, total, avg, weighted, fails_per_student])
np.savetxt("summary.csv", summary, delimiter=",", fmt="%.2f", header="...", comments="")
```

`np.column_stack` joins 1D arrays as columns of a 2D array. I also saved the whole printed report by collecting every line in a list and writing it to `report.txt`.

## 17. Formatting output with f-strings

```python
f"{name:<9}"     # left-align, width 9
f"{value:>7.1f}" # right-align, width 7, 1 decimal
f"{z:+.2f}"      # always show the + or - sign
```

## 18. Common mistakes I should avoid

1. Mixing up `axis=0` and `axis=1`
2. Forgetting parentheses around each condition when using `&` and `|`
3. Forgetting `.copy()` before changing data
4. Mixing up `argsort` positions with the ranks themselves
5. Forgetting the minus sign in `argsort(-avg)` and getting the ranking backwards
6. Using `argmax` and treating the result as the value instead of the position
7. Not checking for missing values and the shape before analyzing

## 19. How I would explain this project in an interview

> "I built a student grade analyzer in pure NumPy. It loads a 30 by 5 dataset, calculates statistics per student and per subject, ranks students by a credit-weighted average using matrix multiplication, detects outliers with z-scores, finds the correlation between subjects, and runs a what-if simulation for curving a subject. It saves a report and a summary CSV. I used vectorized operations and boolean masks instead of loops."

Be ready to answer: *Why a weighted average? What does `axis=1` do? Why did you copy the array before curving? What does a correlation of 0.44 mean?*

## 20. Ideas to extend it

- Use my own real class grades
- Add a 6th subject
- Let the user type a student name and print only that student's report
- Find each student's strongest and weakest subject (`argmax` and `argmin` with `axis=1`)
- Rebuild the same project with Pandas (Day 7) and compare the code length
- Draw charts with Matplotlib (Day 8)

## 21. My own words

*(Fill this in yourself. This is the part employers actually read.)*

- **What the project does, in my own words:**
- **The part I understood best:**
- **The part that confused me (ranking? correlation? broadcasting?):**
- **How I solved it:**
- **One thing I would improve:**

## Resources used

- NumPy official docs: `argsort`, `digitize`, `corrcoef`, `loadtxt` and `savetxt`
- Add the videos or tutorials you actually used here

---

**Next:** Day 4 starts **Pandas**: Series, DataFrames, reading CSV files, and filtering. The same grade data will be much easier to handle there.