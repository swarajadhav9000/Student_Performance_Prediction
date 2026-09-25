import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

df = pd.read_csv("data/student_performance_raw.csv")

df = df.drop_duplicates(subset="student_id")

for col in ["attendance", "study_hours", "previous_score", "internal_marks"]:
    df[col] = df[col].fillna(df[col].mean())

encoders = {}
for col in ["gender", "subject"]:
    le = LabelEncoder()
    df[col + "_encoded"] = le.fit_transform(df[col])
    encoders[col] = le

feature_cols = [
    "attendance",
    "study_hours",
    "previous_score",
    "participation",
    "internal_marks",
]
scaler = MinMaxScaler()
df[[c + "_scaled" for c in feature_cols]] = scaler.fit_transform(df[feature_cols])

df.to_csv("data/student_performance_clean.csv", index=False)
print(f"Saved cleaned dataset with {len(df)} rows to data/student_performance_clean.csv")
print(df.isnull().sum())
