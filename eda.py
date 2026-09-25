import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

os.makedirs("charts", exist_ok=True)
df = pd.read_csv("data/student_performance_clean.csv")

sns.set_style("whitegrid")

plt.figure(figsize=(6, 4))
sns.countplot(x="performance_level", data=df, order=["Low", "Average", "High"])
plt.title("Distribution of Performance Levels")
plt.tight_layout()
plt.savefig("charts/performance_distribution.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.scatterplot(x="study_hours", y="final_score", hue="performance_level", data=df)
plt.title("Study Hours vs Final Score")
plt.tight_layout()
plt.savefig("charts/study_hours_vs_score.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.scatterplot(x="attendance", y="final_score", hue="performance_level", data=df)
plt.title("Attendance vs Final Score")
plt.tight_layout()
plt.savefig("charts/attendance_vs_score.png")
plt.close()

plt.figure(figsize=(7, 5))
corr_cols = ["attendance", "study_hours", "previous_score", "participation",
             "internal_marks", "final_score"]
sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("charts/correlation_heatmap.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.boxplot(x="subject", y="final_score", data=df)
plt.title("Score Spread by Subject")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("charts/score_by_subject.png")
plt.close()

print("EDA charts saved in charts/")
print(df[corr_cols].corr()["final_score"].sort_values(ascending=False))
