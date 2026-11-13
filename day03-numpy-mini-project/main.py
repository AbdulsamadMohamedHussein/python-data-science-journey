"""
DAY 3 - NUMPY MINI PROJECT: STUDENT GRADE ANALYZER
===================================================
Uses ONLY NumPy (plus Python's pathlib for file paths) to analyze the
grades of 30 students in 5 subjects.

What this project practices (everything from Days 1 and 2):
    - creating arrays, random data, saving and loading CSV files
    - axis=0 / axis=1 statistics
    - sorting and ranking (argsort)
    - boolean masks and np.where / np.select / np.digitize
    - broadcasting (z-scores, curving)
    - matrix math (weighted averages with @)
    - correlation (np.corrcoef)

Run:  .\\venv\\Scripts\\python.exe day03-numpy-mini-project\\main.py

Output files created for you:
    data/grades.csv     the raw scores (30 rows x 5 columns)
    data/students.txt   one student name per line
    results/report.txt  the full text report
    results/summary.csv one numeric row per student
"""

from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------
# SETTINGS (constants are written in CAPITAL LETTERS by convention)
# ---------------------------------------------------------------------
BASE = Path(__file__).parent            # the folder this file lives in
DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results"

SUBJECTS = ["Math", "Physics", "Programming", "English", "AI"]
CREDITS = np.array([4, 3, 4, 2, 3])     # credit hours per subject
PASS_MARK = 50                          # minimum score to pass a subject

NAMES = [
    "Amina", "Bilal", "Chen", "Dana", "Eli", "Fatima", "Gao", "Hassan",
    "Ivy", "Jamal", "Kenji", "Layla", "Mei", "Noor", "Omar", "Priya",
    "Qiang", "Rania", "Sara", "Tariq", "Uma", "Victor", "Wei", "Xin",
    "Yusuf", "Zainab", "Aisha", "Bao", "Cleo", "Dawit",
]

# We collect every printed line here so we can also save the report
REPORT_LINES = []


def say(text=""):
    """Print a line AND remember it for the report file."""
    print(text)
    REPORT_LINES.append(text)


def title(text):
    """Print a section header."""
    say("")
    say("=" * 64)
    say(text)
    say("=" * 64)


# =====================================================================
# PART 1: CREATE, SAVE AND LOAD THE DATA
# =====================================================================
def generate_dataset(n_students=30, seed=2026):
    """Make realistic fake grades.

    Each subject has its own average and spread. A hidden 'ability'
    value per student is added to ALL their subjects, so good students
    tend to be good everywhere (this creates correlation, see Part 9).
    """
    rng = np.random.default_rng(seed)
    means = np.array([68, 62, 75, 72, 70])      # typical score per subject
    stds = np.array([14, 15, 12, 10, 13])       # how spread out each subject is

    ability = rng.normal(0, 8, size=(n_students, 1))        # (30, 1)
    # means and stds have shape (5,), so they BROADCAST across 30 rows
    scores = rng.normal(means, stds, size=(n_students, len(SUBJECTS)))
    scores = scores + ability                                # (30,5) + (30,1)
    return np.clip(scores, 0, 100).round(1)                  # keep 0 to 100


def ensure_data():
    """Create the data files the first time the program runs."""
    DATA_DIR.mkdir(exist_ok=True)
    grades_file = DATA_DIR / "grades.csv"
    names_file = DATA_DIR / "students.txt"
    if not grades_file.exists():
        scores = generate_dataset(len(NAMES))
        np.savetxt(grades_file, scores, delimiter=",",
                   header=",".join(SUBJECTS), comments="", fmt="%.1f")
        names_file.write_text("\n".join(NAMES), encoding="utf-8")
    return grades_file, names_file


def load_data():
    """Read the CSV back into NumPy. skiprows=1 skips the header line."""
    grades_file, names_file = ensure_data()
    scores = np.loadtxt(grades_file, delimiter=",", skiprows=1)
    names = np.array(names_file.read_text(encoding="utf-8").splitlines())
    return names, scores


# =====================================================================
# PART 2: ANALYSIS FUNCTIONS (each one does ONE job)
# =====================================================================
def subject_stats(scores):
    """Statistics for each SUBJECT (one value per column -> axis=0)."""
    return {
        "mean": scores.mean(axis=0),
        "median": np.median(scores, axis=0),
        "std": scores.std(axis=0),
        "min": scores.min(axis=0),
        "max": scores.max(axis=0),
    }


def rank_students(avg):
    """Return the order (best first) and each student's rank number.

    np.argsort gives the POSITIONS that would sort the array.
    We sort the negative averages so the biggest average comes first.
    """
    order = np.argsort(-avg, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(avg) + 1)   # put rank 1..n in the right spots
    return order, ranks


def letter_grades(avg):
    """Convert averages to letters: A 80+, B 70-79, C 60-69, D 50-59, F <50."""
    edges = [50, 60, 70, 80]                    # boundaries between grades
    letters = np.array(["F", "D", "C", "B", "A"])
    # digitize tells us which "bucket" each average falls in (0 to 4)
    return letters[np.digitize(avg, edges)]


def text_bar(count, scale=1):
    """A tiny text chart bar made of # characters."""
    return "#" * int(count * scale)


# =====================================================================
# PART 3: THE MAIN PROGRAM
# =====================================================================
def main():
    names, scores = load_data()
    n_students, n_subjects = scores.shape

    # -----------------------------------------------------------------
    # 1. LOOK AT THE DATA FIRST
    # -----------------------------------------------------------------
    title("1. DATASET OVERVIEW")
    say(f"Students : {n_students}")
    say(f"Subjects : {', '.join(SUBJECTS)}")
    say(f"Shape    : {scores.shape}  (rows = students, columns = subjects)")
    say(f"dtype    : {scores.dtype}")
    say(f"Missing  : {np.isnan(scores).sum()} missing values")
    say(f"Range    : {scores.min()} to {scores.max()}")
    say("First 3 students:")
    for i in range(3):
        say(f"  {names[i]:<8} {scores[i]}")

    # -----------------------------------------------------------------
    # 2. STATISTICS PER SUBJECT (axis=0: go DOWN the rows)
    # -----------------------------------------------------------------
    title("2. SUBJECT STATISTICS")
    stats = subject_stats(scores)
    say(f"{'Subject':<13}{'Mean':>7}{'Median':>8}{'Std':>7}{'Min':>7}{'Max':>7}")
    for j, subject in enumerate(SUBJECTS):
        say(f"{subject:<13}{stats['mean'][j]:>7.1f}{stats['median'][j]:>8.1f}"
            f"{stats['std'][j]:>7.1f}{stats['min'][j]:>7.1f}{stats['max'][j]:>7.1f}")

    hardest = SUBJECTS[stats["mean"].argmin()]
    easiest = SUBJECTS[stats["mean"].argmax()]
    most_spread = SUBJECTS[stats["std"].argmax()]
    say("")
    say(f"Hardest subject (lowest mean)  : {hardest}")
    say(f"Easiest subject (highest mean) : {easiest}")
    say(f"Most spread out (highest std)  : {most_spread}")

    # Who got the top score in each subject?
    say("")
    say("Top scorer in each subject:")
    best_rows = scores.argmax(axis=0)           # row index of the max per column
    for j, subject in enumerate(SUBJECTS):
        r = best_rows[j]
        say(f"  {subject:<13}{names[r]:<8}{scores[r, j]}")

    # -----------------------------------------------------------------
    # 3. STATISTICS PER STUDENT (axis=1: go ACROSS the columns)
    # -----------------------------------------------------------------
    total = scores.sum(axis=1)
    avg = scores.mean(axis=1)

    # Weighted average: subjects with more credits count more.
    # (30,5) @ (5,) -> (30,)   then divide by the total credits
    weighted = scores @ CREDITS / CREDITS.sum()

    # -----------------------------------------------------------------
    # 4. RANKING
    # -----------------------------------------------------------------
    title("3. RANKING (by weighted average)")
    order, ranks = rank_students(weighted)
    grades = letter_grades(weighted)

    header = f"{'Rank':<6}{'Name':<9}{'Avg':>7}{'Weighted':>10}{'Grade':>7}   Scores"
    say(header)
    say("-" * len(header) + "-" * 22)
    for position in order:
        say(f"{ranks[position]:<6}{names[position]:<9}{avg[position]:>7.1f}"
            f"{weighted[position]:>10.1f}{grades[position]:>7}   {scores[position]}")

    say("")
    say("Top 5   : " + ", ".join(names[order[:5]]))
    say("Bottom 5: " + ", ".join(names[order[-5:]]))
    # Does the weighting change the ranking compared to the plain average?
    order_plain = np.argsort(-avg, kind="stable")
    changed = (order != order_plain).sum()
    say(f"Positions that change between plain and weighted ranking: {changed}")

    # -----------------------------------------------------------------
    # 5. PASS / FAIL ANALYSIS (boolean masks)
    # -----------------------------------------------------------------
    title("4. PASS / FAIL ANALYSIS (pass mark = %d)" % PASS_MARK)
    passed = scores >= PASS_MARK                 # (30, 5) True/False grid
    fails_per_student = (~passed).sum(axis=1)    # how many subjects failed
    fails_per_subject = (~passed).sum(axis=0)    # how many students failed

    say("Failures per subject:")
    for j, subject in enumerate(SUBJECTS):
        pass_rate = passed[:, j].mean() * 100    # mean of booleans = fraction
        say(f"  {subject:<13}{fails_per_subject[j]:>3} failed   "
            f"pass rate {pass_rate:5.1f}%")

    all_pass = passed.all(axis=1)
    say("")
    say(f"Students who passed ALL subjects : {all_pass.sum()}")
    say(f"Students who failed at least one : {(~all_pass).sum()}")
    say(f"Students who failed 2 or more    : {(fails_per_student >= 2).sum()}")
    if (~all_pass).any():
        say("Students with a failed subject:")
        for r in np.where(~all_pass)[0]:
            failed_names = [s for s, ok in zip(SUBJECTS, passed[r]) if not ok]
            say(f"  {names[r]:<8}-> {', '.join(failed_names)}")

    # -----------------------------------------------------------------
    # 6. GRADE DISTRIBUTION
    # -----------------------------------------------------------------
    title("5. LETTER GRADE DISTRIBUTION")
    for letter in ["A", "B", "C", "D", "F"]:
        count = (grades == letter).sum()
        say(f"  {letter}: {count:>2}  {text_bar(count)}")

    # np.histogram counts how many averages fall in each range
    bins = [0, 50, 60, 70, 80, 90, 100]
    counts, _ = np.histogram(weighted, bins=bins)
    say("")
    say("Histogram of weighted averages:")
    for k, count in enumerate(counts):
        say(f"  {bins[k]:>3}-{bins[k + 1]:<3}: {count:>2}  {text_bar(count)}")

    # -----------------------------------------------------------------
    # 7. HONORS AND AT-RISK LISTS
    # -----------------------------------------------------------------
    title("6. HONORS AND AT-RISK STUDENTS")
    honors = (weighted >= 75) & all_pass         # & means AND (use brackets!)
    at_risk = (weighted < 55) | (fails_per_student >= 2)   # | means OR

    say("Honors (weighted >= 75 and no failed subject):")
    say("  " + (", ".join(names[honors]) if honors.any() else "nobody"))
    say("At risk (weighted < 55 or 2+ failed subjects):")
    say("  " + (", ".join(names[at_risk]) if at_risk.any() else "nobody"))

    # -----------------------------------------------------------------
    # 8. OUTLIERS WITH Z-SCORES (broadcasting)
    # -----------------------------------------------------------------
    title("7. OUTLIERS (|z-score| > 2)")
    # (30,5) - (5,) broadcasts the column means down every row
    z = (scores - scores.mean(axis=0)) / scores.std(axis=0)
    rows, cols = np.where(np.abs(z) > 2)
    if len(rows) == 0:
        say("No outliers found.")
    for r, c in zip(rows, cols):
        say(f"  {names[r]:<8}{SUBJECTS[c]:<13}score {scores[r, c]:>5}   z = {z[r, c]:+.2f}")

    # -----------------------------------------------------------------
    # 9. CORRELATION BETWEEN SUBJECTS
    # -----------------------------------------------------------------
    title("8. CORRELATION BETWEEN SUBJECTS")
    # rowvar=False means: each COLUMN is a variable (a subject)
    corr = np.corrcoef(scores, rowvar=False)     # (5, 5)
    say(f"{'':<13}" + "".join(f"{s[:7]:>9}" for s in SUBJECTS))
    for j, subject in enumerate(SUBJECTS):
        say(f"{subject:<13}" + "".join(f"{value:>9.2f}" for value in corr[j]))

    # Find the strongest pair. We keep only the upper triangle so we do not
    # count the diagonal (always 1) or the same pair twice.
    upper = np.triu(corr, k=1)
    a, b = np.unravel_index(np.argmax(np.abs(upper)), upper.shape)
    say("")
    say(f"Strongest relationship: {SUBJECTS[a]} and {SUBJECTS[b]} "
        f"(r = {corr[a, b]:.2f})")
    say("(r near +1 means students good at one are good at the other.)")

    # -----------------------------------------------------------------
    # 10. WHAT-IF: CURVING A SUBJECT
    # -----------------------------------------------------------------
    title("9. WHAT-IF: CURVE THE HARDEST SUBJECT BY +5")
    col = SUBJECTS.index(hardest)
    curved = scores.copy()                       # never edit the original!
    curved[:, col] = np.clip(curved[:, col] + 5, 0, 100)
    before = (scores[:, col] >= PASS_MARK).sum()
    after = (curved[:, col] >= PASS_MARK).sum()
    say(f"{hardest} mean before: {scores[:, col].mean():.1f}   "
        f"after: {curved[:, col].mean():.1f}")
    say(f"{hardest} students passing: {before} -> {after}")
    new_weighted = curved @ CREDITS / CREDITS.sum()
    say(f"Overall weighted class average: {weighted.mean():.2f} -> "
        f"{new_weighted.mean():.2f}")

    # -----------------------------------------------------------------
    # 11. CLASS SUMMARY
    # -----------------------------------------------------------------
    title("10. CLASS SUMMARY")
    say(f"Class weighted average : {weighted.mean():.2f}")
    say(f"Highest weighted avg   : {weighted.max():.2f} ({names[weighted.argmax()]})")
    say(f"Lowest weighted avg    : {weighted.min():.2f} ({names[weighted.argmin()]})")
    say(f"Median weighted avg    : {np.median(weighted):.2f}")
    say(f"Overall pass rate      : {passed.mean() * 100:.1f}% of all subject results")
    say(f"Students on honors     : {honors.sum()} of {n_students}")
    say(f"Students at risk       : {at_risk.sum()} of {n_students}")

    # -----------------------------------------------------------------
    # 12. SAVE RESULTS
    # -----------------------------------------------------------------
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "report.txt").write_text("\n".join(REPORT_LINES), encoding="utf-8")

    summary = np.column_stack([ranks, total, avg, weighted, fails_per_student])
    np.savetxt(RESULTS_DIR / "summary.csv", summary, delimiter=",", fmt="%.2f",
               header="rank,total,average,weighted_average,failed_subjects",
               comments="")
    print("\nSaved: results/report.txt and results/summary.csv")


if __name__ == "__main__":
    main()


# =====================================================================
# EXTENSION IDEAS (do these to make the project YOURS)
# =====================================================================
# 1. Add a 6th subject (change SUBJECTS, CREDITS, means, stds) and re-run.
#    Delete data/grades.csv first so new data is generated.
# 2. Use your OWN class grades (as a CSV) instead of the fake data.
# 3. Add a function that finds the most improved subject per student.
# 4. Let the user type a student name and print only that student's report
#    (use input() and np.where(names == typed_name)).
# 5. Change the grade boundaries and see how the distribution changes.
# 6. On Day 8 you will draw real charts from this same data with Matplotlib.