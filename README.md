# Clinical Decision Support Demo

A polished Streamlit demo for the Breast Cancer Wisconsin dataset.

## What it includes

- Professional Streamlit UI for presentations
- Logistic regression model trained on the Breast Cancer Wisconsin dataset
- Simple doctor-style rule baseline for side-by-side comparison
- Live patient input controls for the 10 most influential features
- Held-out test metrics and confusion matrices
- Presentation notes and demo talking points built into the app

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Recommended presentation framing

Use this as a **decision-support** demo, not a "doctor replacement" demo.
A strong line is:

> The model helps combine many measurements consistently and quickly, while the clinician still owns the final judgment.

## Notes

- This is for class/demo use only.
- It is **not** for clinical use.
- To keep the interface manageable, 20 of the 30 features stay fixed at median values.
