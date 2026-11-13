# Student Grade Analyzer (NumPy Mini Project)

A command-line data analysis project built **only with NumPy**. It loads the grades of 30 students in 5 subjects, analyzes them, and writes a full text report.

> Day 3 of my [Python Data Science Journey](../README.md). Detailed study notes are in [`notes.md`](notes.md).

## What it does

- Generates a realistic dataset (30 students x 5 subjects) and saves it to `data/grades.csv`
- Loads the data back with `np.loadtxt` and checks it (shape, types, missing values)
- Calculates statistics **per subject** (mean, median, std, min, max) and **per student**
- Finds the hardest and easiest subjects and the top scorer in each one
- Ranks students by **weighted average** (subjects have different credit hours)
- Converts averages to letter grades (A to F)
- Analyzes pass/fail rates per subject and finds students who failed 2 or more subjects
- Builds **honors** and **at-risk** lists using boolean masks
- Detects **outliers** with z-scores
- Calculates the **correlation** between subjects
- Runs a **what-if** simulation: what happens if the hardest subject is curved by +5?
- Saves `results/report.txt` and `results/summary.csv`

## NumPy concepts used

| Concept | Where |
|---------|-------|
| `axis=0` and `axis=1` statistics | Subject stats and student averages |
| Broadcasting | Random data generation, z-scores |
| Matrix multiplication `@` | Weighted averages |
| `argsort`, `argmax`, `argmin` | Ranking and top scorers |
| Boolean masks, `&`, `\|`, `~` | Pass/fail, honors, at-risk |
| `np.digitize`, `np.histogram` | Letter grades and distribution |
| `np.where`, `np.clip` | Outliers and curving |
| `np.corrcoef`, `np.triu`, `np.unravel_index` | Subject correlation |
| `np.savetxt`, `np.loadtxt`, `np.column_stack` | Reading and saving files |

## How to run

```powershell
cd python-data-science-journey
.\venv\Scripts\python.exe day03-numpy-mini-project\main.py
```

The first run creates the `data/` and `results/` folders automatically.

## Project structure

```
day03-numpy-mini-project/
|-- main.py          # the whole program
|-- notes.md         # my study notes
|-- README.md        # this file
|-- data/
|   |-- grades.csv   # 30 x 5 scores (created on first run)
|   `-- students.txt # student names
`-- results/
    |-- report.txt   # full text report
    `-- summary.csv  # one numeric row per student
```

## Sample output

**Subject statistics**

```
Subject         Mean  Median    Std    Min    Max
Math            70.0    68.5   18.8   35.7  100.0
Physics         60.7    56.4   18.4   24.4  100.0
Programming     75.8    76.6   11.9   49.9  100.0
English         72.1    71.7   11.2   53.5  100.0
AI              73.0    72.9   16.4   43.3  100.0

Hardest subject (lowest mean)  : Physics
Easiest subject (highest mean) : Programming
```

**Top of the ranking**

```
Rank  Name         Avg  Weighted  Grade   Scores
1     Dana        93.1      91.4      A   [ 89.2 100.   76.4 100.  100. ]
2     Bao         83.6      83.8      A   [65.9 84.6 94.7 74.  98.9]
3     Aisha       81.5      82.3      A   [ 71.  100.   80.6  56.9  98.9]
```

**Pass / fail analysis**

```
Failures per subject:
  Math           5 failed   pass rate  83.3%
  Physics        8 failed   pass rate  73.3%
  Programming    1 failed   pass rate  96.7%
  English        0 failed   pass rate 100.0%
  AI             3 failed   pass rate  90.0%
```

**Grade distribution**

```
  A:  8  ########
  B:  8  ########
  C:  9  #########
  D:  5  #####
  F:  0
```

**What-if: curve the hardest subject by +5**

```
Physics mean before: 60.7   after: 65.4
Physics students passing: 22 -> 24
Overall weighted class average: 70.55 -> 71.43
```

## What I learned

*(Write this part yourself in 3 to 5 sentences: what was new, what was hard, what you would improve.)*

## Next steps

- Replace the fake data with real class grades
- Draw charts of these results with **Matplotlib** (Day 8)
- Rebuild the same analysis with **Pandas** and compare how much shorter the code becomes (Day 7)

## Note

The data is randomly generated for practice and does not belong to real students.