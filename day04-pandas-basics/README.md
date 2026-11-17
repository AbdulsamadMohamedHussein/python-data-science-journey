# Day 4: Pandas Basics - Exploring a Student Dataset

I explored a table of 60 students (text, numbers, and True/False columns) using **Pandas**: reading a CSV, inspecting it, selecting data, filtering rows, creating new columns, sorting, ranking, and answering questions about the data.

> Part of my [Python Data Science Journey](../README.md). Detailed study notes are in [`notes.md`](notes.md).

## What I studied

- The Pandas **Series** (labeled 1D data) and **DataFrame** (a table)
- Reading a CSV with `read_csv` and inspecting it with `head`, `tail`, `sample`, `info`, `describe`, `isna`, `value_counts`
- Selecting columns and rows with `df[...]`, `loc`, `iloc`, `at`, `iat`, and `set_index`
- Filtering with conditions: `&`, `|`, `~`, `isin`, `between`, `str.contains`, `query`
- Creating columns: averages, best subject, letter grades (`pd.cut`), ranks
- Sorting with `sort_values`, `nlargest`, `nsmallest`
- A first look at missing values and `groupby`
- Saving results with `to_csv`

## The dataset

`data/students.csv` is created automatically on the first run. It is randomly generated practice data (no real students).

| Column | Type | Meaning |
|--------|------|---------|
| `student_id`, `name` | text | Identifiers |
| `gender`, `major`, `city` | text | Categories |
| `year` | integer | Study year (1 to 4) |
| `math`, `physics`, `programming`, `english`, `ai` | decimal | Scores out of 100 |
| `attendance` | decimal | Percent attended (3 values are missing on purpose) |
| `study_hours` | integer | Hours per week |
| `scholarship` | True/False | Has a scholarship |

## How to run

```powershell
cd python-data-science-journey
.\venv\Scripts\python.exe day04-pandas-basics\main.py
```

Output is also saved to `results/day4_report.txt`, and the top 10 students to `results/top10_students.csv`.

## Sample output

**Inspecting the data with `info()`**

```
RangeIndex: 60 entries, 0 to 59
Data columns (total 14 columns):
 #   Column       Non-Null Count  Dtype
 0   student_id   60 non-null     str
 ...
 11  attendance   57 non-null     float64
 12  study_hours  60 non-null     int64
 13  scholarship  60 non-null     bool
```

`attendance` has only 57 non-null values, so 3 are missing. I will clean this on Day 5.

**Filtering: AI students with programming >= 80 and a scholarship**

```python
df[(df["major"] == "AI") & (df["programming"] >= 80) & df["scholarship"]]
```

```
           name  programming  scholarship
0   Amina Ahmed         94.4         True
8     Ivy Ahmed         95.9         True
16  Qiang Ahmed         86.7         True
58    Cleo Chen         80.2         True
```

**New columns**

```
          name  average best_subject weakest_subject grade  passed_all  rank
0  Amina Ahmed     85.9  programming         english     A        True     5
1   Bilal Chen     64.7  programming              ai     C       False    41
```

**Some of the questions I answered**

```
Which subject has the highest mean?   programming
Top 3 students by average:            ['Ivy Ahmed', 'Jamal Chen', 'Noor Wang']
Students with a scholarship:          21
Average score with / without a scholarship:  81.1 / 64.5
Correlation: study_hours vs average:  0.0
City with the highest average score:  Guilin
Students who passed every subject:    39 of 60
```

The data is random, so these findings only show how the tools work and are not real conclusions. For example, the correlation between study hours and average is about 0 because the fake data was not built to link them.

## Pandas vs NumPy (Day 3)

The same kinds of analysis needed less code and kept the student names with their scores in a single table. Pandas handles mixed data types and missing values for me.

## What I learned

*(Write this part yourself in 3 to 5 sentences: what was new, what was confusing, what you want to practice.)*

## Next steps

- Day 5: clean a messy dataset (missing values, duplicates, wrong types, messy text)
- Day 6: `groupby`, `merge`, `concat`, `pivot_table`
- Day 8: turn these results into charts with Matplotlib