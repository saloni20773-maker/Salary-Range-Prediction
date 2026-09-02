import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Salary Range Prediction",
    page_icon="💼",
    layout="wide"
)
DATA_PATH = "Jobs_NYC_Postings_small.csv"

CATEGORICAL_FEATURES = [
    "Posting Type",
    "Full-Time/Part-Time indicator",
    "Salary Frequency",
    "Career Level",
    "Title Classification",
    "Level",
    "Agency",
    "Residency Requirement"
]

NUMERICAL_FEATURES = [
    "# Of Positions",
    "Posting Year",
    "Posting Month",
    "Posting Day of Week",
    "Posting Duration Days",
    "Post Until Missing",
    "Business Title Character Count",
    "Business Title Word Count",
    "Job Description Character Count",
    "Job Description Word Count",
    "Preferred Skills Character Count",
    "Preferred Skills Word Count",
    "Minimum Qual Requirements Character Count",
    "Minimum Qual Requirements Word Count",
    "Additional Information Character Count",
    "Additional Information Word Count"
]

TEXT_FEATURES = [
    "Business Title",
    "Job Description",
    "Preferred Skills",
    "Minimum Qual Requirements",
    "Additional Information"
]


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Match the notebook's cleaning and annualization logic.
    df = df.drop_duplicates().reset_index(drop=True)

    empty_columns = df.columns[df.isna().all()].tolist()
    if empty_columns:
        df = df.drop(columns=empty_columns)

    # Dates
    df["Post Until"] = pd.to_datetime(df["Post Until"], errors="coerce")
    df["Posting Date"] = pd.to_datetime(df["Posting Date"], errors="coerce")
    df["Post Until Missing"] = df["Post Until"].isna().astype(int)

    # Salary annualization
    df["Salary Range From Annual"] = df["Salary Range From"]
    df["Salary Range To Annual"] = df["Salary Range To"]

    frequency = df["Salary Frequency"].fillna("Not Provided").astype(str)
    hourly_mask = frequency.str.lower().eq("hourly")
    daily_mask = frequency.str.lower().eq("daily")

    df.loc[hourly_mask, "Salary Range From Annual"] *= 2080
    df.loc[hourly_mask, "Salary Range To Annual"] *= 2080
    df.loc[daily_mask, "Salary Range From Annual"] *= 260
    df.loc[daily_mask, "Salary Range To Annual"] *= 260

    # Keep records used by the notebook for modelling.
    df = df[
        (df["Salary Range From Annual"] > 0)
        & (df["Salary Range To Annual"] > 0)
    ].copy()

    # Date features
    df["Posting Year"] = df["Posting Date"].dt.year
    df["Posting Month"] = df["Posting Date"].dt.month
    df["Posting Day of Week"] = df["Posting Date"].dt.dayofweek
    df["Posting Duration Days"] = (
        df["Post Until"] - df["Posting Date"]
    ).dt.days

    # Text length features
    for column in TEXT_FEATURES:
        text = df[column].fillna("Not Provided").astype(str)
        df[f"{column} Character Count"] = text.str.len()
        df[f"{column} Word Count"] = text.str.split().str.len()

    # Match the notebook's categorical missing-value treatment.
    for column in CATEGORICAL_FEATURES:
        if column in df.columns:
            df[column] = df[column].fillna("Not Provided").astype(str)

    return df


@st.cache_resource
def train_models(df):
    X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES].copy()
    y_min = df["Salary Range From Annual"].copy()
    y_max = df["Salary Range To Annual"].copy()

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    # The notebook selected Random Forest as the best model for both targets.
    # Its tuned configuration was n_estimators=200, max_depth=None,
    # min_samples_split=2, min_samples_leaf=1.
    min_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    max_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    min_model.fit(X, y_min)
    max_model.fit(X, y_max)

    return min_model, max_model


st.title("💼 Salary Range Prediction")
st.markdown(
    "Predict the estimated **minimum and maximum annual salary** "
    "for a job posting using a Random Forest regression model."
)

with st.spinner("Loading data and preparing the prediction models..."):
    df = load_data()
    min_model, max_model = train_models(df)

st.success("Prediction model is ready.")

st.sidebar.header("Job Details")

def select_from_data(label, column):
    values = (
        df[column]
        .dropna()
        .astype(str)
        .value_counts()
        .index
        .tolist()
    )
    return st.sidebar.selectbox(label, values)

posting_type = select_from_data("Posting Type", "Posting Type")
employment_type = select_from_data(
    "Full-Time / Part-Time", "Full-Time/Part-Time indicator"
)
salary_frequency = select_from_data("Salary Frequency", "Salary Frequency")
career_level = select_from_data("Career Level", "Career Level")
title_classification = select_from_data(
    "Title Classification", "Title Classification"
)
level = select_from_data("Level", "Level")
agency = select_from_data("Agency", "Agency")
residency = select_from_data(
    "Residency Requirement", "Residency Requirement"
)

positions = st.sidebar.number_input(
    "Number of Positions",
    min_value=1,
    value=1,
    step=1
)

posting_date = st.sidebar.date_input(
    "Posting Date",
    value=pd.Timestamp("2026-01-01").date()
)

has_post_until = st.sidebar.checkbox(
    "Closing date is available",
    value=True
)

post_until = None
if has_post_until:
    post_until = st.sidebar.date_input(
        "Post Until",
        value=(pd.Timestamp(posting_date) + pd.Timedelta(days=30)).date()
    )

st.subheader("Job Information")

business_title = st.text_input(
    "Business Title",
    placeholder="e.g. Data Analyst"
)

job_description = st.text_area(
    "Job Description",
    placeholder="Enter the job description..."
)

preferred_skills = st.text_area(
    "Preferred Skills",
    placeholder="Enter preferred skills..."
)

minimum_qualifications = st.text_area(
    "Minimum Qualifications",
    placeholder="Enter minimum qualifications..."
)

additional_information = st.text_area(
    "Additional Information",
    placeholder="Enter any additional information..."
)

if st.button("🔮 Predict Salary Range", type="primary"):
    posting_timestamp = pd.Timestamp(posting_date)

    if has_post_until:
        post_until_timestamp = pd.Timestamp(post_until)
        duration = (post_until_timestamp - posting_timestamp).days
        post_until_missing = 0
    else:
        duration = np.nan
        post_until_missing = 1

    def text_counts(value):
        value = value if value.strip() else "Not Provided"
        return len(value), len(value.split())

    bt_chars, bt_words = text_counts(business_title)
    jd_chars, jd_words = text_counts(job_description)
    ps_chars, ps_words = text_counts(preferred_skills)
    mq_chars, mq_words = text_counts(minimum_qualifications)
    ai_chars, ai_words = text_counts(additional_information)

    input_data = pd.DataFrame([{
        "Posting Type": posting_type,
        "Full-Time/Part-Time indicator": employment_type,
        "Salary Frequency": salary_frequency,
        "Career Level": career_level,
        "Title Classification": title_classification,
        "Level": level,
        "Agency": agency,
        "Residency Requirement": residency,
        "# Of Positions": positions,
        "Posting Year": posting_timestamp.year,
        "Posting Month": posting_timestamp.month,
        "Posting Day of Week": posting_timestamp.dayofweek,
        "Posting Duration Days": duration,
        "Post Until Missing": post_until_missing,
        "Business Title Character Count": bt_chars,
        "Business Title Word Count": bt_words,
        "Job Description Character Count": jd_chars,
        "Job Description Word Count": jd_words,
        "Preferred Skills Character Count": ps_chars,
        "Preferred Skills Word Count": ps_words,
        "Minimum Qual Requirements Character Count": mq_chars,
        "Minimum Qual Requirements Word Count": mq_words,
        "Additional Information Character Count": ai_chars,
        "Additional Information Word Count": ai_words
    }])

    min_prediction = float(min_model.predict(input_data)[0])
    max_prediction = float(max_model.predict(input_data)[0])

    # The notebook validates the final range so that the lower bound
    # does not exceed the upper bound.
    predicted_min = min(min_prediction, max_prediction)
    predicted_max = max(min_prediction, max_prediction)

    st.subheader("🎯 Predicted Salary Range")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Estimated Minimum Annual Salary",
            f"${predicted_min:,.0f}"
        )
    with col2:
        st.metric(
            "Estimated Maximum Annual Salary",
            f"${predicted_max:,.0f}"
        )

    st.info(
        "This prediction is a decision-support estimate based on historical "
        "job-posting data. It should not replace current market information "
        "or human compensation decisions."
    )

st.caption(
    "Model: Random Forest Regressor | "
    "Targets: Minimum and Maximum Annual Salary"
)
