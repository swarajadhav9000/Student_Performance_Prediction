# Student Performance Prediction System

A machine learning capstone project that analyzes student academic data, predicts
performance outcomes (High / Average / Low), and generates an interactive
dashboard with study recommendations.

## Project Structure

```
student-performance-prediction/
├── data/
│   ├── student_performance_raw.csv      (generated)
│   └── student_performance_clean.csv    (generated)
├── charts/                              (generated EDA charts)
├── generate_dataset.py                  Module: creates the dataset
├── preprocess.py                        Module 1: cleaning & preprocessing
├── eda.py                               Module 2: exploratory data analysis
├── train_model.py                       Module 3: model training & evaluation
├── app.py                               Module 4 & 5: dashboard + recommendations
├── requirements.txt
└── README.md
```

## How to Run on Your PC (Windows / Mac / Linux)

### 1. Install Python
Make sure Python 3.9+ is installed. Check with:
```
python --version
```

### 2. Open a terminal in the project folder
Unzip the project folder and `cd` into it:
```
cd student-performance-prediction
```

### 3. Create a virtual environment (recommended)
```
python -m venv venv
```
Activate it:
- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

### 4. Install dependencies
```
pip install -r requirements.txt
```

### 5. Run the pipeline, in order

```
python generate_dataset.py
python preprocess.py
python eda.py
python train_model.py
```

This will:
- Create `data/student_performance_raw.csv` and `data/student_performance_clean.csv`
- Save 5 EDA charts into `charts/`
- Train a Decision Tree and Random Forest classifier + a Linear Regression
  score predictor, print accuracy/R² scores, and save `model_classifier.pkl`,
  `model_regressor.pkl`, and `model_metadata.json`

### 6. Launch the dashboard
```
streamlit run app.py
```
This opens the dashboard in your browser (usually `http://localhost:8501`).

- **📊 Dashboard tab**: KPI cards, performance-level donut chart, average
  score by subject, study-hours-vs-score scatter plot, subject-wise table
- **🔮 Predict a Student tab**: enter attendance/study hours/etc. and get a
  color-coded predicted performance level, predicted score, a confidence
  chart, and tailored recommendations
- **🗂️ Data Explorer tab**: browse and download the full dataset

The dashboard uses custom CSS (gradient header, styled metric cards,
color-coded result cards) and Plotly charts, so it looks like a finished
product rather than a default Streamlit app.

## Notes

- The dataset here is **synthetically generated** (`generate_dataset.py`) since
  no real dataset was provided. If you'd rather use a real one (e.g. the
  Kaggle Student Performance Dataset mentioned in the brief), drop it into
  `data/` and adjust the column names in `preprocess.py` to match.
- Model choice: Random Forest was selected automatically as the best
  classifier based on test accuracy (see `train_model.py` output and
  `model_metadata.json`).
- Scope intentionally excludes deep learning, real-time monitoring, and
  large-scale infrastructure, per the project's stated scope limitations.

## Deliverables Checklist (from the brief)

- [x] Source Code — this folder
- [x] Project Report — write up your workflow, findings from `charts/`, and
      model accuracy from `model_metadata.json` (I can generate a Word/PDF
      report from this project if you want one)
- [ ] PPT Presentation (I can generate this too if you want)
- [ ] GitHub Repository — push this folder to a new repo
- [x] Dashboard Screenshots — take screenshots of the running Streamlit app
- [ ] Deployment Link — deploy via Streamlit Community Cloud (free) or Render
- [ ] Demo Video — screen-record yourself walking through the dashboard
