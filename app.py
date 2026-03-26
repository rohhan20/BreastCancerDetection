from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Breast Cancer ML vs Simple Heuristics",
    page_icon="🩺",
    layout="wide",
)


DOCTOR_RULE_TEXT = (
    "Rule-based baseline: predict **malignant** when at least 2 of these 3 signs are present: "
    "`worst concave points > 0.14`, `worst perimeter > 110`, `mean radius > 15`. "
    "Otherwise predict **benign**."
)


@st.cache_data
def load_demo_objects():
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.copy()
    y = dataset.target.copy()  # 0 = malignant, 1 = benign

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    coef = pd.Series(model.named_steps["clf"].coef_[0], index=X.columns)
    top_features = coef.abs().sort_values(ascending=False).head(10).index.tolist()

    feature_medians = X_train.median()
    feature_ranges = pd.DataFrame(
        {
            "min": X.min(),
            "max": X.max(),
            "median": X.median(),
        }
    )

    benign_example = X_test.loc[y_test[y_test == 1].index[0]].copy()
    malignant_example = X_test.loc[y_test[y_test == 0].index[0]].copy()

    doctor_pred_test = doctor_rule_batch(X_test)

    metrics = pd.DataFrame(
        {
            "Model": ["Doctor-style rule", "Logistic Regression ML"],
            "Accuracy": [
                accuracy_score(y_test, doctor_pred_test),
                accuracy_score(y_test, y_pred),
            ],
            "Precision (Benign)": [
                precision_score(y_test, doctor_pred_test),
                precision_score(y_test, y_pred),
            ],
            "Recall (Benign)": [
                recall_score(y_test, doctor_pred_test),
                recall_score(y_test, y_pred),
            ],
        }
    )

    cm_doctor = confusion_matrix(y_test, doctor_pred_test, labels=[0, 1])
    cm_ml = confusion_matrix(y_test, y_pred, labels=[0, 1])

    return {
        "dataset": dataset,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "coef": coef,
        "top_features": top_features,
        "feature_medians": feature_medians,
        "feature_ranges": feature_ranges,
        "benign_example": benign_example,
        "malignant_example": malignant_example,
        "metrics": metrics,
        "cm_doctor": cm_doctor,
        "cm_ml": cm_ml,
    }


def doctor_rule(row: pd.Series) -> int:
    score = 0
    score += int(row["worst concave points"] > 0.14)
    score += int(row["worst perimeter"] > 110)
    score += int(row["mean radius"] > 15)
    return 0 if score >= 2 else 1



def doctor_rule_batch(frame: pd.DataFrame) -> np.ndarray:
    score = (
        (frame["worst concave points"] > 0.14).astype(int)
        + (frame["worst perimeter"] > 110).astype(int)
        + (frame["mean radius"] > 15).astype(int)
    )
    return np.where(score >= 2, 0, 1)



def label_from_target(target_value: int) -> str:
    return "Benign" if int(target_value) == 1 else "Malignant"



def risk_from_prob(prob_benign: float) -> str:
    prob_malignant = 1 - prob_benign
    if prob_malignant >= 0.80:
        return "High"
    if prob_malignant >= 0.50:
        return "Moderate"
    return "Low"



def format_confusion(cm: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        cm,
        index=["Actual Malignant", "Actual Benign"],
        columns=["Pred Malignant", "Pred Benign"],
    )



def set_profile(profile: pd.Series, editable_features: list[str]) -> None:
    for feature in editable_features:
        st.session_state[feature] = float(profile[feature])



def initialize_inputs(editable_features: list[str], medians: pd.Series) -> None:
    for feature in editable_features:
        if feature not in st.session_state:
            st.session_state[feature] = float(medians[feature])



def build_patient_row(
    editable_features: list[str], medians: pd.Series
) -> pd.Series:
    patient = medians.copy()
    for feature in editable_features:
        patient[feature] = float(st.session_state[feature])
    return patient



def feature_contributions(
    row: pd.Series,
    X_train: pd.DataFrame,
    coef: pd.Series,
    top_n: int = 8,
) -> pd.DataFrame:
    std = X_train.std().replace(0, 1)
    z = (row - X_train.mean()) / std
    contrib = z * coef
    out = pd.DataFrame(
        {
            "feature": contrib.index,
            "contribution": contrib.values,
        }
    )
    out["direction"] = np.where(
        out["contribution"] >= 0,
        "pushes toward benign",
        "pushes toward malignant",
    )
    out["impact"] = out["contribution"].abs()
    return out.sort_values("impact", ascending=False).head(top_n)



def main() -> None:
    objs = load_demo_objects()
    dataset = objs["dataset"]
    model = objs["model"]
    X_train = objs["X_train"]
    coef = objs["coef"]
    editable_features = objs["top_features"]
    medians = objs["feature_medians"]
    feature_ranges = objs["feature_ranges"]

    initialize_inputs(editable_features, medians)

    st.title("🩺 Breast Cancer Detection Demo: ML vs Simple Heuristics")
    st.caption(
        "Demo only — not for medical use. This app compares a simple rule-based baseline with a logistic regression model."
    )

    with st.sidebar:
        st.header("Patient input")
        st.write("Choose a preset or edit the top 10 model features.")

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Median"):
                set_profile(medians, editable_features)
        with c2:
            if st.button("Benign ex"):
                set_profile(objs["benign_example"], editable_features)
        with c3:
            if st.button("Malignant ex"):
                set_profile(objs["malignant_example"], editable_features)

        st.markdown("---")
        st.markdown("**Editable features**")
        for feature in editable_features:
            bounds = feature_ranges.loc[feature]
            span = float(bounds["max"] - bounds["min"])
            step = max(span / 500.0, 0.001)
            st.number_input(
                feature,
                min_value=float(bounds["min"]),
                max_value=float(bounds["max"]),
                value=float(st.session_state[feature]),
                step=float(step),
                key=feature,
                format="%.4f",
            )

        st.info(
            "The remaining 20 features stay fixed at training-set median values to keep the UI simple."
        )

    patient = build_patient_row(editable_features, medians)

    prob_benign = float(model.predict_proba(pd.DataFrame([patient]))[0, 1])
    ml_pred = int(model.predict(pd.DataFrame([patient]))[0])
    doc_pred = int(doctor_rule(patient))

    left, right = st.columns(2)
    with left:
        st.subheader("ML prediction")
        st.metric("Prediction", label_from_target(ml_pred))
        st.metric("P(benign)", f"{prob_benign:.1%}")
        st.metric("Estimated risk", risk_from_prob(prob_benign))

    with right:
        st.subheader("Simple heuristic style baseline")
        st.metric("Prediction", label_from_target(doc_pred))
        st.write(DOCTOR_RULE_TEXT)
        checks = pd.DataFrame(
            {
                "Condition": [
                    "worst concave points > 0.14",
                    "worst perimeter > 110",
                    "mean radius > 15",
                ],
                "Triggered": [
                    bool(patient["worst concave points"] > 0.14),
                    bool(patient["worst perimeter"] > 110),
                    bool(patient["mean radius"] > 15),
                ],
            }
        )
        st.dataframe(checks, hide_index=True, use_container_width=True)

    if ml_pred == doc_pred:
        st.success("Both approaches agree on this patient.")
    else:
        st.warning("The ML model and the doctor-style rule disagree for this patient.")

    st.markdown("---")
    st.subheader("What is driving the ML prediction?")
    contrib = feature_contributions(patient, X_train, coef, top_n=8)
    chart_df = contrib.set_index("feature")[["impact"]]
    st.bar_chart(chart_df)
    st.dataframe(
        contrib[["feature", "direction", "contribution"]],
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Held-out test performance")
    metrics_display = objs["metrics"].copy()
    for col in ["Accuracy", "Precision (Benign)", "Recall (Benign)"]:
        metrics_display[col] = metrics_display[col].map(lambda x: f"{x:.1%}")
    st.dataframe(metrics_display, hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Doctor-style rule confusion matrix**")
        st.dataframe(format_confusion(objs["cm_doctor"]), use_container_width=True)
    with c2:
        st.markdown("**ML confusion matrix**")
        st.dataframe(format_confusion(objs["cm_ml"]), use_container_width=True)

    st.markdown("---")
    with st.expander("About the dataset"):
        st.write(
            f"Samples: {len(objs['X'])} | Features: {objs['X'].shape[1]} | "
            f"Classes: malignant and benign"
        )
        target_counts = pd.Series(objs["y"]).value_counts().sort_index()
        target_table = pd.DataFrame(
            {
                "Class": ["Malignant", "Benign"],
                "Count": [int(target_counts[0]), int(target_counts[1])],
            }
        )
        st.dataframe(target_table, hide_index=True, use_container_width=True)
        st.write(
            "This app uses `sklearn.datasets.load_breast_cancer()` and splits the data into train/test sets with a fixed random seed for reproducibility."
        )

    with st.expander("Presentation talking points"):
        st.markdown(
            """
            - The rule-based baseline mimics a simple clinician-style heuristic.
            - The ML model learns from all 30 measurements at once.
            - Even with a simple logistic regression model, performance on the held-out test set is much stronger.
            """
        )


if __name__ == "__main__":
    main()
