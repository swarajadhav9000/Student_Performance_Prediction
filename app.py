import pickle

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = ["attendance", "study_hours", "previous_score", "participation", "internal_marks"]

CUSTOM_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 1.75rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 700;
    }
    .hero p {
        margin: 0.35rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    .result-card {
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        color: white;
    }
    .result-high { background: linear-gradient(135deg, #059669, #10b981); }
    .result-average { background: linear-gradient(135deg, #d97706, #f59e0b); }
    .result-low { background: linear-gradient(135deg, #dc2626, #ef4444); }

    .result-card h2 { margin: 0 0 0.2rem 0; font-size: 1.6rem; }
    .result-card p { margin: 0; opacity: 0.92; }

    .tip-box {
        background: #f8fafc;
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.92rem;
    }

    section[data-testid="stSidebar"] {
        background: #f8fafc;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_models():
    with open("model_classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("model_regressor.pkl", "rb") as f:
        reg = pickle.load(f)
    return clf, reg


@st.cache_data
def load_data():
    return pd.read_csv("data/student_performance_clean.csv")


def get_recommendations(attendance, study_hours, participation, internal_marks):
    tips = []
    if attendance < 75:
        tips.append("📌 Improve attendance — aim for at least 75%.")
    if study_hours < 3:
        tips.append("📚 Increase daily study hours to at least 3–4 hours.")
    if participation < 5:
        tips.append("🙋 Attend practice sessions and participate more in class.")
    if internal_marks < 25:
        tips.append("🎯 Focus on weak subjects to raise internal assessment scores.")
    if not tips:
        tips.append("✅ Performance indicators look solid — keep up the consistency.")
    return tips


LEVEL_STYLE = {
    "High": ("result-high", "🌟", "Excellent trajectory — keep reinforcing these habits."),
    "Average": ("result-average", "⚡", "Solid footing — a few tweaks could push this higher."),
    "Low": ("result-low", "⚠️", "At risk — early intervention is recommended."),
}

clf, reg = load_models()
df = load_data()

with st.sidebar:
    st.markdown("### 🎓 Navigation")
    st.markdown("Use the tabs to explore overall trends, run a prediction for "
                "one student, or browse the raw dataset.")
    st.divider()
    st.markdown("### About")
    st.caption(
        "Student Performance Prediction System — a machine learning capstone "
        "project that analyzes academic data and predicts outcomes using a "
        "Random Forest classifier."
    )
    st.divider()
    st.markdown("### Model Info")
    st.caption(f"Classifier: **{type(clf).__name__}**")
    st.caption(f"Regressor: **{type(reg).__name__}**")

st.markdown(
    """
    <div class="hero">
        <h1>🎓 Student Performance Prediction System</h1>
        <p>ML-powered dashboard for predicting academic outcomes and generating
        actionable insights.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["📊  Dashboard", "🔮  Predict a Student", "🗂️  Data Explorer"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", len(df))
    col2.metric("Average Score", f"{df['final_score'].mean():.1f}")
    col3.metric("High Performers", int((df["performance_level"] == "High").sum()))
    col4.metric("Avg. Attendance", f"{df['attendance'].mean():.1f}%")

    st.write("")
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("##### Performance Level Distribution")
        counts = df["performance_level"].value_counts().reindex(["High", "Average", "Low"])
        fig = px.pie(
            values=counts.values,
            names=counts.index,
            color=counts.index,
            color_discrete_map={"High": "#10b981", "Average": "#f59e0b", "Low": "#ef4444"},
            hole=0.45,
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320,
                           legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.markdown("##### Average Score by Subject")
        subj = df.groupby("subject")["final_score"].mean().sort_values()
        fig2 = px.bar(x=subj.values, y=subj.index, orientation="h",
                       color=subj.values, color_continuous_scale="Purples")
        fig2.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320,
                            xaxis_title="Average Score", yaxis_title="",
                            coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##### Study Hours vs Final Score")
    fig3 = px.scatter(
        df, x="study_hours", y="final_score", color="performance_level",
        color_discrete_map={"High": "#10b981", "Average": "#f59e0b", "Low": "#ef4444"},
        opacity=0.75, labels={"study_hours": "Study Hours / Day", "final_score": "Final Score"},
    )
    fig3.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("##### Subject-wise Analysis")
    st.dataframe(
        df.groupby("subject")[["final_score", "attendance", "study_hours"]]
        .mean().round(1).sort_values("final_score", ascending=False),
        use_container_width=True,
    )

with tab2:
    st.markdown("##### Enter student details")
    c1, c2 = st.columns(2)
    with c1:
        attendance = st.slider("Attendance (%)", 0, 100, 75)
        study_hours = st.slider("Daily Study Hours", 0.0, 12.0, 4.0, 0.5)
        previous_score = st.slider("Previous Score", 0, 100, 65)
    with c2:
        participation = st.slider("Participation (1–10)", 1, 10, 5)
        internal_marks = st.slider("Internal Marks (out of 50)", 0, 50, 28)

    predict_clicked = st.button("🔮 Predict Performance", type="primary", use_container_width=True)

    if predict_clicked:
        input_df = pd.DataFrame(
            [[attendance, study_hours, previous_score, participation, internal_marks]],
            columns=FEATURES,
        )
        pred_level = clf.predict(input_df)[0]
        pred_proba = clf.predict_proba(input_df)[0]
        pred_score = reg.predict(input_df)[0]

        css_class, emoji, note = LEVEL_STYLE[pred_level]

        st.markdown(
            f"""
            <div class="result-card {css_class}">
                <h2>{emoji} Predicted Level: {pred_level}</h2>
                <p>Predicted final score: <b>{pred_score:.1f} / 100</b> — {note}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("###### Prediction Confidence")
            proba_df = pd.DataFrame(
                {"Level": clf.classes_, "Confidence": pred_proba}
            ).sort_values("Confidence", ascending=True)
            fig4 = px.bar(proba_df, x="Confidence", y="Level", orientation="h",
                           color="Confidence", color_continuous_scale="Viridis", range_x=[0, 1])
            fig4.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=220,
                                coloraxis_showscale=False)
            st.plotly_chart(fig4, use_container_width=True)

        with col_b:
            st.markdown("###### Recommendations")
            for tip in get_recommendations(attendance, study_hours, participation, internal_marks):
                st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)

with tab3:
    st.markdown("##### Full Dataset")
    st.dataframe(df, use_container_width=True, height=420)
    st.download_button(
        "⬇️ Download dataset as CSV",
        df.to_csv(index=False).encode("utf-8"),
        "student_performance_clean.csv",
        "text/csv",
    )
