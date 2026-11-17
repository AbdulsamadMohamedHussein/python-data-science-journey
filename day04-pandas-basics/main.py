"""
DAY 4 - PANDAS BASICS: SERIES, DATAFRAMES, READING CSV, loc / iloc, FILTERING
==============================================================================
Run:  .\\venv\\Scripts\\python.exe day04-pandas-basics\\main.py

HOW TO STUDY THIS FILE
    1. Read the comments of one section.
    2. Run the file and find that section's output.
    3. Change a number or a column name, run again, and see what changes.
    4. Do the practice exercises in the last section.

What you will learn today
    - Series    : one labeled column of data
    - DataFrame : a whole table (many Series sharing one index)
    - Reading a CSV file and inspecting it (head, info, describe ...)
    - Selecting columns and rows (df["x"], loc, iloc)
    - Filtering rows with conditions (& | ~ isin between query)
    - Creating columns, sorting, ranking
    - Saving results back to CSV

Files created for you when you run it:
    data/students.csv   60 students with text, numbers and True/False columns
    results/day4_report.txt, results/top10_students.csv
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------
BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results"
SUBJECTS = ["math", "physics", "programming", "english", "ai"]

# These options control how pandas PRINTS tables (they do not change data)
pd.set_option("display.width", 140)        # wider lines before wrapping
pd.set_option("display.max_columns", 25)   # show up to 25 columns
pd.set_option("display.max_rows", 40)      # show up to 40 rows

REPORT_LINES = []                          # everything we print is saved here


def say(*parts):
    """Print anything (text, Series, DataFrame) and remember it for the report."""
    text = " ".join(str(p) for p in parts)
    print(text)
    REPORT_LINES.append(text)


def title(text):
    """Print a big section header."""
    say("")
    say("=" * 70)
    say(text)
    say("=" * 70)


# =====================================================================
# PART 0: CREATE THE DATASET (a CSV file to practice on)
# =====================================================================
FIRST_NAMES = [
    "Amina", "Bilal", "Chen", "Dana", "Eli", "Fatima", "Gao", "Hassan",
    "Ivy", "Jamal", "Kenji", "Layla", "Mei", "Noor", "Omar", "Priya",
    "Qiang", "Rania", "Sara", "Tariq", "Uma", "Victor", "Wei", "Xin",
    "Yusuf", "Zainab", "Aisha", "Bao", "Cleo", "Dawit",
]
LAST_NAMES = ["Ahmed", "Zhang", "Hussein", "Wang", "Khan", "Li", "Ali", "Chen"]
MAJORS = ["AI", "Software", "DataSci", "Electronics"]
CITIES = ["Guilin", "Nanning", "Shenzhen", "Guangzhou", "Chengdu", "Wuhan"]


def make_dataset(n=60, seed=4):
    """Build a fake but realistic student table as a DataFrame."""
    rng = np.random.default_rng(seed)           # seed = repeatable results

    # Text columns are given as a dictionary: {column name: list of values}
    df = pd.DataFrame({
        "student_id": [f"S{i:03d}" for i in range(1, n + 1)],        # S001, S002 ...
        "name": [f"{FIRST_NAMES[i % 30]} {LAST_NAMES[(i * 7 + i // 30) % 8]}"
                 for i in range(n)],
        "gender": rng.choice(["F", "M"], size=n),
        "major": rng.choice(MAJORS, size=n),
        "year": rng.integers(1, 5, size=n),                          # 1 to 4
        "city": rng.choice(CITIES, size=n),
    })

    # Scores: same idea as Day 3 (each subject has its own mean and spread,
    # plus a hidden "ability" added to every subject of the same student)
    means = np.array([68, 62, 75, 72, 70])
    stds = np.array([14, 15, 12, 10, 13])
    ability = rng.normal(0, 8, size=(n, 1))
    scores = rng.normal(means, stds, size=(n, 5)) + ability
    scores = np.clip(scores, 0, 100).round(1)
    df[SUBJECTS] = scores            # assign a 2D array to 5 columns at once

    # Attendance: we deliberately make 3 values MISSING (NaN) so you can
    # see how pandas shows missing data. Day 5 teaches how to clean it.
    attendance = np.clip(rng.normal(88, 8, size=n), 50, 100).round(1)
    attendance[rng.choice(n, size=3, replace=False)] = np.nan
    df["attendance"] = attendance

    df["study_hours"] = rng.integers(5, 36, size=n)                  # hours per week
    df["scholarship"] = (scores.mean(axis=1) + rng.normal(0, 5, size=n)) > 76  # True/False
    return df


def load_data():
    """Create the CSV on the first run, then ALWAYS read it from the file."""
    DATA_DIR.mkdir(exist_ok=True)
    csv_path = DATA_DIR / "students.csv"
    if not csv_path.exists():
        make_dataset().to_csv(csv_path, index=False)   # index=False: no extra index column
    return csv_path, pd.read_csv(csv_path)


# =====================================================================
# PART 1: THE SERIES (one labeled column)
# =====================================================================
def part1_series():
    title("1. SERIES: A 1D ARRAY WITH LABELS")

    # A Series = values + an index (the labels). Think of NumPy array + names.
    marks = pd.Series([88, 92, 79, 65, 95],
                      index=["Ali", "Bao", "Chen", "Dana", "Eli"],
                      name="math")
    say(marks)
    say("")
    say("type          :", type(marks).__name__)
    say("values        :", marks.values)         # the raw NumPy array inside
    say("index         :", list(marks.index))    # the labels
    say("dtype         :", marks.dtype)
    say("shape / size  :", marks.shape, "/", marks.size)

    # Two ways to get one item:
    #   .loc[label]      -> by LABEL    (the name)
    #   .iloc[position]  -> by POSITION (0, 1, 2 ... like NumPy)
    say("")
    say("marks.loc['Bao']  :", marks.loc["Bao"])
    say("marks.iloc[1]     :", marks.iloc[1])
    say("several labels:")
    say(marks.loc[["Ali", "Eli"]])
    say("label slice 'Bao':'Dana' (END IS INCLUDED for labels!):")
    say(marks.loc["Bao":"Dana"])

    # Math works on the whole Series (vectorized, just like NumPy)
    say("")
    say("marks + 5 (curve):")
    say(marks + 5)
    say("mean / max / idxmax:", marks.mean(), "/", marks.max(), "/", marks.idxmax())
    # idxmax gives the LABEL of the max ("Eli"); NumPy's argmax gives a position.

    # Boolean filtering works too (like masks on Day 1)
    say("")
    say("students with marks above 80:")
    say(marks[marks > 80])
    say("sorted from best to worst:")
    say(marks.sort_values(ascending=False))
    say("describe() gives instant statistics:")
    say(marks.describe())

    # IMPORTANT: Series ALIGN on their labels when you do math between them.
    # A label that exists in only one Series becomes NaN (Not a Number = missing).
    s1 = pd.Series([1, 2, 3], index=["a", "b", "c"])
    s2 = pd.Series([10, 20, 30], index=["b", "c", "d"])
    say("")
    say("s1 + s2 (aligned by label, 'a' and 'd' have no partner -> NaN):")
    say(s1 + s2)

    # value_counts counts how often each value appears (VERY useful)
    letters = pd.Series(["A", "B", "A", "C", "A", "B"])
    say("")
    say("value_counts:")
    say(letters.value_counts())


# =====================================================================
# PART 2: THE DATAFRAME (a whole table)
# =====================================================================
def part2_dataframe():
    title("2. DATAFRAME: A TABLE OF SERIES")

    # Way 1: from a DICTIONARY. Each key is a column, each value a list.
    small = pd.DataFrame({
        "name": ["Ali", "Bao", "Chen"],
        "math": [88, 92, 79],
        "passed": [True, True, True],
    })
    say("From a dictionary of columns:")
    say(small)

    # Way 2: from a LIST of dictionaries. Each dictionary is one ROW.
    rows = pd.DataFrame([
        {"name": "Dana", "math": 65},
        {"name": "Eli", "math": 95},
    ])
    say("")
    say("From a list of row-dictionaries:")
    say(rows)

    # Way 3: from a NumPy array + column names (your Day 1-3 skills connect here)
    grid = pd.DataFrame(np.arange(6).reshape(3, 2), columns=["x", "y"])
    say("")
    say("From a NumPy array:")
    say(grid)

    # The most important attributes
    say("")
    say("shape   :", small.shape)             # (rows, columns)
    say("columns :", list(small.columns))
    say("index   :", small.index)             # RangeIndex = 0, 1, 2 ...
    say("dtypes  :")
    say(small.dtypes)

    # Adding, changing and removing columns
    small["double"] = small["math"] * 2       # new column from another column
    small["bonus"] = 5                        # one value copied to every row
    say("")
    say("after adding 'double' and 'bonus':")
    say(small)
    small = small.drop(columns=["double", "bonus"])   # drop returns a NEW DataFrame
    say("after drop(columns=[...]):", list(small.columns))
    say("")
    say("A column of a DataFrame is a Series:", type(small["math"]).__name__)


# =====================================================================
# PART 3: READING AND INSPECTING THE CSV
# =====================================================================
def part3_inspect(df, csv_path):
    title("3. READ AND INSPECT A CSV FILE")
    say("File:", csv_path.name)

    # The first things you do with ANY new dataset:
    say("")
    say("shape (rows, columns):", df.shape)
    say("")
    say("head(5) = first 5 rows:")
    say(df.head(5))
    say("")
    say("tail(3) = last 3 rows:")
    say(df.tail(3))
    say("")
    say("sample(3, random_state=1) = 3 random rows:")
    say(df.sample(3, random_state=1))

    # info() shows column names, non-null counts and types.
    # It prints directly, so we capture it in a buffer to save it in the report.
    say("")
    say("info():")
    buffer = io.StringIO()
    df.info(buf=buffer)
    say(buffer.getvalue())
    # Look at "Non-Null Count": 'attendance' has fewer than 60 -> missing values!

    say("describe() = statistics of the NUMBER columns:")
    say(df.describe().round(2))

    say("")
    say("describe() on TEXT columns (top = most common value, freq = how often):")
    say(df[["gender", "major", "city"]].describe())

    say("")
    say("Missing values per column:")
    say(df.isna().sum())

    say("")
    say("Duplicate rows       :", df.duplicated().sum())
    say("Unique cities (count):", df["city"].nunique())
    say("Unique cities (list) :", sorted(df["city"].unique()))
    say("")
    say("Students per major:")
    say(df["major"].value_counts())
    say("")
    say("Same thing as percentages:")
    say((df["major"].value_counts(normalize=True) * 100).round(1))

    # You can read just part of a file (useful for huge files)
    say("")
    say("read_csv with usecols + nrows (only 3 columns, only 4 rows):")
    say(pd.read_csv(csv_path, usecols=["name", "major", "math"], nrows=4))


# =====================================================================
# PART 4: SELECTING COLUMNS AND ROWS
# =====================================================================
def part4_selecting(df):
    title("4. SELECTING DATA: COLUMNS, loc AND iloc")

    # --- COLUMNS ---
    say("One column (a Series):")
    say(df["name"].head(3))
    say("")
    say("Several columns (a DataFrame, note the DOUBLE brackets):")
    say(df[["name", "major", "math"]].head(3))

    # --- ROWS and COLUMNS TOGETHER ---
    # loc  = LABELS   -> df.loc[row_labels, column_labels]   END IS INCLUDED
    # iloc = POSITIONS-> df.iloc[row_pos, column_pos]        END IS EXCLUDED
    say("")
    say("df.loc[0:2, ['name', 'math']]  (rows 0,1,2 -> 3 rows, end INCLUDED):")
    say(df.loc[0:2, ["name", "math"]])
    say("")
    say("df.iloc[0:2, 1:4]  (rows 0,1 -> 2 rows, end EXCLUDED, columns 1,2,3):")
    say(df.iloc[0:2, 1:4])
    say("")
    say("last row with iloc[-1]:")
    say(df.iloc[-1])                              # a row becomes a Series
    say("")
    say("every 20th row, 3 columns:")
    say(df.iloc[::20, [0, 1, 6]])

    # Single values (fast access)
    say("")
    say("df.at[0, 'name']  (by label)   :", df.at[0, "name"])
    say("df.iat[0, 1]      (by position):", df.iat[0, 1])

    # --- USING A COLUMN AS THE INDEX ---
    # With the name as index we can look up students by name.
    by_name = df.set_index("name")
    first_student = df["name"].iloc[0]
    say("")
    say(f"After set_index('name'), look up '{first_student}':")
    say(by_name.loc[first_student, ["major", "math", "ai"]])
    say("")
    say("reset_index() turns the index back into a normal column:")
    say(by_name.reset_index().head(2).iloc[:, :4])

    # --- CHANGING VALUES: always use loc ---
    # (Never write df[condition]["col"] = value, it may silently do nothing.)
    demo = df.copy()                              # work on a copy, keep df safe
    demo.loc[0, "math"] = 100
    demo.loc[demo["math"] > 99, "math"] = 99      # condition on rows, column name
    say("")
    say("demo.loc[0, 'math'] after the two changes:", demo.loc[0, "math"])


# =====================================================================
# PART 5: FILTERING ROWS
# =====================================================================
def part5_filtering(df):
    title("5. FILTERING ROWS WITH CONDITIONS")

    # A condition gives a Series of True/False (a "mask", like NumPy Day 1).
    # Putting the mask inside df[...] keeps only the True rows.
    mask = df["math"] > 90
    say("mask (first 5 values):", list(mask.head(5)))
    say("")
    say("Students with math > 90:")
    say(df[mask][["name", "major", "math"]])

    # Several conditions: & (and), | (or), ~ (not). ALWAYS use brackets.
    say("")
    say("AI students with programming >= 80 AND a scholarship:")
    both = df[(df["major"] == "AI") & (df["programming"] >= 80) & df["scholarship"]]
    say(both[["name", "programming", "scholarship"]])

    say("")
    say("Year 1 OR year 4 students (count):",
        len(df[(df["year"] == 1) | (df["year"] == 4)]))
    say("Students WITHOUT a scholarship (count):", len(df[~df["scholarship"]]))

    # Handy shortcuts
    say("")
    say("isin: students from Guilin or Nanning (count):",
        len(df[df["city"].isin(["Guilin", "Nanning"])]))
    say("between: physics from 40 to 50 (count):",
        len(df[df["physics"].between(40, 50)]))

    # Text filtering with the .str accessor
    say("")
    say("Names starting with 'A':")
    say(df[df["name"].str.startswith("A")]["name"].tolist())
    say("Names containing 'ei' (case ignored):")
    say(df[df["name"].str.contains("ei", case=False)]["name"].tolist())

    # query(): write the condition as text (clean for long conditions)
    limit = 85
    say("")
    say("query: ai > @limit and major == 'AI':")
    say(df.query("ai > @limit and major == 'AI'")[["name", "major", "ai"]])

    # Missing data: isna() / notna()
    say("")
    say("Students with missing attendance:")
    say(df[df["attendance"].isna()][["name", "attendance"]])

    # Counting with masks: True counts as 1
    say("")
    say("How many students failed math (< 50)?     :", (df["math"] < 50).sum())
    say("What percent have a scholarship?          :",
        round(df["scholarship"].mean() * 100, 1), "%")

    # A filtered table is a NEW table. Use .copy() if you want to modify it.
    high = df[df["math"] > 80].copy()
    high["note"] = "strong in math"              # no warning because of .copy()
    say("")
    say("A filtered copy with a new column:")
    say(high[["name", "math", "note"]].head(3))


# =====================================================================
# PART 6: NEW COLUMNS, SORTING AND RANKING
# =====================================================================
def part6_columns_sorting(df):
    title("6. NEW COLUMNS, SORTING AND RANKING")

    # df[SUBJECTS] selects the 5 subject columns. axis=1 works ACROSS columns
    # (one result per student), exactly like NumPy Day 3.
    df["average"] = df[SUBJECTS].mean(axis=1).round(1)
    df["total"] = df[SUBJECTS].sum(axis=1).round(1)

    # idxmax(axis=1) returns the COLUMN NAME of each row's biggest value
    df["best_subject"] = df[SUBJECTS].idxmax(axis=1)
    df["weakest_subject"] = df[SUBJECTS].idxmin(axis=1)

    # pd.cut sorts numbers into labeled ranges (replaces np.digitize)
    df["grade"] = pd.cut(df["average"],
                         bins=[0, 50, 60, 70, 80, 101],
                         labels=["F", "D", "C", "B", "A"],
                         right=False)            # right=False: 80 belongs to "A"

    # (df[SUBJECTS] >= 50) is a True/False table; .all(axis=1) asks
    # "is every value in this row True?"
    df["passed_all"] = (df[SUBJECTS] >= 50).all(axis=1)

    # rank(): 1 = best. method="min" gives tied students the same rank.
    df["rank"] = df["average"].rank(ascending=False, method="min").astype(int)

    say("New columns:")
    say(df[["name", "average", "best_subject", "weakest_subject", "grade",
            "passed_all", "rank"]].head(5))

    # Sorting
    say("")
    say("sort_values('average', ascending=False).head(5):")
    say(df.sort_values("average", ascending=False)[["name", "average"]].head(5))

    say("")
    say("Sort by TWO columns (major A-Z, then average high-to-low):")
    two = df.sort_values(["major", "average"], ascending=[True, False])
    say(two[["major", "name", "average"]].head(6))

    say("")
    say("nlargest(3) and nsmallest(3) are shortcuts:")
    say(df.nlargest(3, "average")[["name", "average"]])
    say(df.nsmallest(3, "average")[["name", "average"]])

    # Renaming and reordering columns
    renamed = df.rename(columns={"study_hours": "hours_per_week"})
    say("")
    say("renamed column:", "hours_per_week" in renamed.columns)

    # apply: run your own function on each value (use only when no
    # built-in exists, because it is slower than vectorized code)
    df["name_length"] = df["name"].apply(len)
    say("name_length example:", df.loc[0, "name"], "->", int(df.loc[0, "name_length"]))

    say("")
    say("Grade counts (sorted by grade, not by count):")
    say(df["grade"].value_counts().sort_index())
    return df


# =====================================================================
# PART 7: A FIRST LOOK AT SUMMARIES (more on Day 6)
# =====================================================================
def part7_summaries(df):
    title("7. QUICK SUMMARIES")

    say("Mean of each subject (like NumPy axis=0):")
    say(df[SUBJECTS].mean().round(1))

    say("")
    say("Correlation between subjects:")
    say(df[SUBJECTS].corr().round(2))

    # Missing values are SKIPPED by default in pandas (NumPy would give nan)
    say("")
    say("attendance mean (NaN skipped automatically):", round(df["attendance"].mean(), 2))
    say("values actually used:", df["attendance"].count(), "of", len(df))

    # groupby preview: "split into groups, then calculate"
    say("")
    say("PREVIEW (Day 6): average score per major:")
    say(df.groupby("major")["average"].mean().round(1).sort_values(ascending=False))


# =====================================================================
# PART 8: TEN QUESTIONS ABOUT THE DATASET
# =====================================================================
def part8_questions(df):
    title("8. TEN QUESTIONS ANSWERED WITH PANDAS")

    say("Q1. How many students are there?          ", len(df))
    say("Q2. Students per major:")
    say(df["major"].value_counts().to_string())
    say("Q3. Which subject has the highest mean?   ",
        df[SUBJECTS].mean().idxmax())
    say("Q4. Top 3 students by average:            ",
        df.nlargest(3, "average")["name"].tolist())
    say("Q5. How many students have a scholarship? ", int(df["scholarship"].sum()))
    say("Q6. Average score with / without scholarship:")
    say(df.groupby("scholarship")["average"].mean().round(1).to_string())
    say("Q7. Students with attendance below 75:    ",
        df[df["attendance"] < 75]["name"].tolist())
    say("Q8. Correlation: study_hours vs average:  ",
        round(df["study_hours"].corr(df["average"]), 2))
    say("Q9. City with the highest average score:  ",
        df.groupby("city")["average"].mean().idxmax())
    say("Q10. Students who passed every subject:   ",
        f"{int(df['passed_all'].sum())} of {len(df)}")


# =====================================================================
# PART 9: SAVE RESULTS
# =====================================================================
def part9_save(df):
    title("9. SAVING RESULTS")
    RESULTS_DIR.mkdir(exist_ok=True)

    top10 = df.nlargest(10, "average")[["rank", "name", "major", "average", "grade"]]
    top10.to_csv(RESULTS_DIR / "top10_students.csv", index=False)
    say("Saved results/top10_students.csv")

    (RESULTS_DIR / "day4_report.txt").write_text("\n".join(REPORT_LINES), encoding="utf-8")
    print("Saved results/day4_report.txt")


# =====================================================================
# MAIN
# =====================================================================
def main():
    csv_path, df = load_data()
    part1_series()
    part2_dataframe()
    part3_inspect(df, csv_path)
    part4_selecting(df)
    part5_filtering(df)
    df = part6_columns_sorting(df)
    part7_summaries(df)
    part8_questions(df)
    part9_save(df)


if __name__ == "__main__":
    main()


# =====================================================================
# YOUR PRACTICE (write the code BELOW this line, inside main() or in a
# new function, then run the file)
# =====================================================================
# 1.  Print the 'name' and 'city' of students in year 3.
# 2.  How many students study more than 25 hours per week?
# 3.  Find the student with the highest 'physics' score (use idxmax, then loc).
# 4.  Show all Software students with english >= 75, sorted by english.
# 5.  Create a column 'hours_level': "high" if study_hours >= 20 else "low"
#     (hint: np.where).
# 6.  Which 5 students have the lowest attendance? (NaN is skipped by nsmallest)
# 7.  What percent of AI students have a scholarship?
# 8.  Use query() to find students from Wuhan with average > 70.
# 9.  Which subject has the lowest mean among year-4 students only?
# 10. Challenge: save all students with passed_all == False to
#     results/needs_support.csv, sorted by average (lowest first).