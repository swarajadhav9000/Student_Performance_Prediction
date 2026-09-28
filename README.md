# 🎓 Student Performance Prediction System


A machine learning project that analyzes student academic data — attendance,
study hours, prior scores, participation, and internal marks — to predict
performance outcomes (High / Average / Low) and generate actionable study
recommendations through an interactive dashboard.

## Overview

Every student's academic outcome is shaped by a handful of measurable habits:
how often they show up, how much they study, and how they've performed
before. This project turns those signals into a working prediction tool —
one that a school or ed-tech platform could plug into a student portal to
flag at-risk learners early and suggest concrete next steps, rather than
just reporting grades after the fact.

The system covers the full ML workflow end to end: data preparation,
exploratory analysis, model training and comparison, and a live dashboard
for predictions and recommendations.

## Project Structure

```
student-performance-prediction/
├── data/
│   ├── student_performance_raw.csv      (generated)
│   └── student_performance_clean.csv    (generated)
├── charts/                              (generated EDA charts)
├── screenshots/                         (add your own dashboard screenshots here)
├── generate_dataset.py                  Dataset creation
├── preprocess.py                        Module 1: cleaning & preprocessing
├── eda.py                               Module 2: exploratory data analysis
├── train_model.py                       Module 3: model training & evaluation
├── app.py                               Module 4 & 5: dashboard + recommendations
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

Since no ready-made dataset was provided within the project timeline, a
synthetic dataset (`generate_dataset.py`) was generated instead — 600
student records with realistic ranges for attendance, study hours, previous
scores, participation, and internal marks, plus intentional missing values
and duplicate rows to make the cleaning step meaningful. The feature-to-score
relationship was built with controlled noise so the resulting patterns are
learnable but not trivial, closer to real academic data than a purely random
dataset would be.

If a real dataset (e.g. the Kaggle Student Performance Dataset) is
substituted later, only the column names in `preprocess.py` need to be
adjusted to match.


## Results

| Model | Task | Score |
|---|---|---|
| Decision Tree | Classification (performance level) | 79.2% accuracy |
| **Random Forest** (selected) | Classification (performance level) | **90.8% accuracy** |
| Linear Regression | Regression (final score) | R² = 0.863 |

Random Forest was selected automatically in `train_model.py` based on test
accuracy. Full metrics are saved in `model_metadata.json` after training.

Feature correlation with final score (see `charts/correlation_heatmap.png`):
study hours and internal marks are the strongest predictors, attendance the
weakest — a genuinely useful insight for where intervention effort should go.

## Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly
- **Dashboard:** Streamlit
- **Deployment:** Streamlit Community Cloud

## Deployment Link: https://smart-student-predictor.streamlit.app/