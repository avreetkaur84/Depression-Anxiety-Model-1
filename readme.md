# 🧠 Depression Detection from DAIC-WOZ Conversations

This project aims to detect depression in participants using the DAIC-WOZ dataset by analyzing emotional and semantic features extracted from interview transcripts. The project is divided into two main phases:

- **Phase 1**: Detect whether a participant is depressed (classification).
- **Phase 2**: Predict the severity of depression only for depressed participants (regression).

## 📂 Project Structure

```

anxiety-model/
├── data/                  # Raw and preprocessed transcript data
├── EDA/                   # Cleaned text, extracted features, visualizations
├── model/                 # Notebooks and scripts for classification & regression
├── docs/                  # Paper, charts, tables, reference materials
├── venv/                  # Python virtual environment
├── requirements.txt       # Project dependencies
└── README.md              # You're here!

````

## ✅ Features Used

- **Linguistic & Emotional Features**: 15 features (e.g., sadness, fear, joy, nervousness, etc.)
- **Semantic Sentence Embeddings**: 384-dimensional embeddings (from a transformer model)
- **Target Variables**:
  - `phq8_binary`: 0 = Not Depressed, 1 = Depressed (used for classification)
  - `phq8_score`: Continuous PHQ-8 score (used for regression)

## 🔍 Project Workflow

### Phase 1: Depression Classification
- Input: Emotional features + sentence embeddings
- Output: Binary label (depressed or not)
- Models: Logistic Regression, Random Forest, XGBoost, Neural Networks
- Metrics: Accuracy, F1-score, ROC-AUC

### Phase 2: Depression Severity Estimation
- Input: Features from **only depressed** participants
- Output: PHQ-8 score (0–24)
- Models: Ridge, Lasso, Gradient Boosting, Neural Networks
- Metrics: MAE, RMSE, R²

## 📌 Novelty
- Applies a **two-stage pipeline**:
  1. **Classify** depression status
  2. **Predict** severity only for depressed participants
- Focuses exclusively on **text modality** (no audio/video) for lightweight deployment
- Future integration with **CBT-based therapy modules**

## 📊 Tools & Technologies

- Python (Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn)
- Jupyter Notebooks
- Pretrained transformer-based sentence embeddings
- DAIC-WOZ dataset (transcripts only)

## 🧪 Setup Instructions

```bash
git clone https://github.com/<your-username>/anxiety-model.git
cd anxiety-model
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate (on Windows)
pip install -r requirements.txt
````

## 📈 Future Work

* Extend model for anxiety detection post depression classification
* Use temporal patterns across sessions
* Deploy a web-based mental health screening tool

---

> ⚠️ **Ethics Note**: This project is for academic/research use only. It is not a substitute for clinical diagnosis or treatment.

---