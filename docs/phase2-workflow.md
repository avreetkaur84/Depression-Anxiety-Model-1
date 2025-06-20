## 📌 **PHASE 2 WORKFLOW: Depression → Anxiety Detection**

### 🔄 Overall Logic Flow:

1. **Detect Depression** using full dataset → `phq8_binary = 1` means depressed.
2. **If Depressed**, check for **Anxiety** using a model trained on anxiety signals from transcripts.

---

## ✅ Step-by-Step Phase 2 Pipeline

### 🔹 **Step 1: Depression Detection (Phase 1 — Already Done)**

* Input: All participants
* Output: `phq8_binary` = 0 or 1
* Store or filter for: Only participants with `phq8_binary == 1`

```python
depressed_df = full_df[full_df['phq8_binary'] == 1].copy()
```

---

### 🔹 **Step 2: Create Anxiety Labels (Pseudo-Labeling)**

* Use emotion features like:

  ```python
  anxiety_features = ['nervousness', 'fear', 'confusion', 'sadness', 'negative_emotion', 'anger']
  ```
* Normalize → Average → Create `anxiety_score`
* Convert to `anxiety_level` (low, moderate, high) or binary (e.g., high = 1, else = 0)

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
depressed_df[anxiety_features] = scaler.fit_transform(depressed_df[anxiety_features])
depressed_df['anxiety_score'] = depressed_df[anxiety_features].mean(axis=1)

# Optional: convert to binary anxiety label
depressed_df['anxiety_binary'] = depressed_df['anxiety_score'].apply(lambda x: 1 if x >= 0.5 else 0)
```

---

### 🔹 **Step 3: Save a Clean `label.csv` for Phase 2**

This new label file will only include depressed individuals and their **anxiety status**:

```python
label_df = depressed_df[['participant_id', 'anxiety_score', 'anxiety_binary']]
label_df.to_csv("labels_phase2.csv", index=False)
```

---

### 🔹 **Step 4: Model Training for Anxiety Detection**

* Inputs:

  * `embed_0` to `embed_383`
  * Or Empath + custom features
* Target: `anxiety_binary` or `anxiety_level`
* Model: Logistic Regression, Random Forest, XGBoost, etc.

---

### 🔹 **Step 5: Test the Model on New Depressed Participants**

When you get new participants:

1. Run PHASE 1 → Check if `phq8_binary = 1`
2. If yes → Pass transcript → Extract features → Use Phase 2 model to predict anxiety

---

## 📁 Your Folder/File Structure Should Look Like:

```
/data/
  └── full_dataset.csv
  └── labels_phase1.csv     # phq8_binary labels
  └── labels_phase2.csv     # only depressed + anxiety labels

/models/
  └── depression_model.pkl
  └── anxiety_model.pkl

/EDA/
  └── feature_extraction.py
  └── phase2_model_train.py
```

---

## 🔚 Summary: Phase 2 Key Points

| Component       | Description                                              |
| --------------- | -------------------------------------------------------- |
| Dataset Filter  | Only participants with `phq8_binary == 1`                |
| Target Variable | Pseudo-anxiety: `anxiety_binary` or `anxiety_score`      |
| Input Features  | Transcript-based: Empath features, embeddings, etc.      |
| Label File      | `labels_phase2.csv` only includes depressed participants |
| Final Output    | Anxiety prediction for depressed participants            |

---