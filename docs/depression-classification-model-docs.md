## ✅ Phase 1 Goal: Depression Detection (Classification)

> **Objective:** Predict `phq8_binary` (0 = Not Depressed, 1 = Depressed)

---

## 🧠 Plan Breakdown

### 🔹 Step 1: Dataset Preparation

#### ❌ Columns to Drop:

* **Drop `phq8_score`**: It's the regression target for later, and using it now would leak information.
* **Do NOT drop `phq8_binary`** — it **is the current target**.
  You **only drop `phq8_binary`** from the **feature matrix (`X`)**, not from the **dataset or the label vector (`y`)**.

#### ✅ Inputs (X):

* 15 **emotional/lexical features**
* 384 **sentence embeddings** (`embed_0` to `embed_383`)

#### ✅ Output (y):

* `phq8_binary` (classification label)

---

### 🔹 Step 2: Model Choice

We’ll begin with **baseline traditional ML models** because:

* **Dataset is small (20 samples)** → Deep learning is **overkill** for now.
* Tree-based and linear models handle small datasets well.
* This helps us **quickly test feasibility and feature importance**.

#### 🔧 Models to Try:

| Model                   | Why Use It                                                         |
| ----------------------- | ------------------------------------------------------------------ |
| **Logistic Regression** | Simple linear baseline; easy to interpret                          |
| **Random Forest**       | Captures nonlinear patterns; works well with mixed features        |
| **XGBoost**             | More powerful tree ensemble; good with small tabular datasets      |
| **SVM (optional)**      | Good with small data & high-dimensional features (like embeddings) |

Later (once you have more transcripts), we can try:

* **MLP** (Neural Network)
* **Fine-tuned BERT** or transformer-based classifiers

---

### 🔹 Step 3: Evaluation Metrics

Since class imbalance exists (14 vs. 6), **accuracy isn't enough**.

Use:

* **F1 Score** (balances precision & recall)
* **ROC-AUC**
* **Confusion Matrix**
* **Precision/Recall**

---

### 🔹 Step 4: Pipeline Flow Summary

```text
1. Load final feature file
2. Drop `phq8_score`
3. Separate X (features) and y (`phq8_binary`)
4. Standardize embedding columns (recommended)
5. Train-test split (80-20, stratified)
6. Fit models (Logistic Regression, Random Forest, XGBoost)
7. Evaluate using F1, ROC-AUC
8. Save results & confusion matrix
```

---