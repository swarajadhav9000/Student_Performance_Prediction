import numpy as np
import pandas as pd

np.random.seed(42)

N = 600

student_id = [f"STU{1000 + i}" for i in range(N)]
attendance = np.clip(np.random.normal(75, 12, N), 40, 100).round(1)
study_hours = np.clip(np.random.normal(4, 2, N), 0, 12).round(1)
previous_score = np.clip(np.random.normal(65, 15, N), 20, 100).round(1)
participation = np.random.randint(1, 11, N)
internal_marks = np.clip(np.random.normal(28, 8, N), 0, 50).round(1)
subject = np.random.choice(
    ["Math", "Science", "English", "Computer Science", "Social Studies"], N
)
gender = np.random.choice(["Male", "Female"], N)

score = (
    0.25 * attendance
    + 4.5 * study_hours
    + 0.35 * previous_score
    + 1.8 * participation
    + 0.9 * internal_marks
    + np.random.normal(0, 6, N)
)
final_score = np.clip(score / 1.6, 0, 100).round(1)


def bucket(s):
    if s >= 75:
        return "High"
    elif s >= 50:
        return "Average"
    return "Low"


performance_level = [bucket(s) for s in final_score]

df = pd.DataFrame(
    {
        "student_id": student_id,
        "gender": gender,
        "subject": subject,
        "attendance": attendance,
        "study_hours": study_hours,
        "previous_score": previous_score,
        "participation": participation,
        "internal_marks": internal_marks,
        "final_score": final_score,
        "performance_level": performance_level,
    }
)

missing_idx = np.random.choice(df.index, size=15, replace=False)
df.loc[missing_idx, "attendance"] = np.nan

dup_rows = df.sample(5, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

df.to_csv("data/student_performance_raw.csv", index=False)
print(f"Generated data/student_performance_raw.csv with {len(df)} rows")
