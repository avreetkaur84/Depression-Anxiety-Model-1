# Anxiety‑Model Project – EDA Pipeline Documentation

*Last updated: 11 Jun 2025*

---

## 1. Purpose

This document records every data‑preparation step you have completed inside the `` folder so far.  It explains **what each script does, what files it produces, and how these pieces fit together** so that anyone (including future‑you) can reproduce or extend the pipeline.

---

## 2. Current Directory Layout

```
project‑root/
├── data/                     # Raw DAIC‑WOZ ZIPs + metadata ↓
│   └── …
├── EDA/
│   ├── clean_text/           # Cleaned transcript .txt files (per participant)
│   │   ├── 301_TRANSCRIPT_clean.txt
│   │   └── …
│   ├── clean_transcripts.py  # Step 2.1 – transcript cleaning
│   ├── text_feature_extraction.py   # Step 2.2A/B – LIWC + lexical
│   ├── extract_nlp_features.py      # wrapper / helper (tokenization etc.)
│   ├── filter_emotion_features.py   # optional extra emotion filtering
│   ├── sentence_embedding_extraction.py  # Step 2.2C – utterance embeddings → CSV
│   ├── aggregate_embeddings.py      # Averages embeddings per participant
│   ├── features.csv                 # LIWC + lexical features (participant‑level)
│   ├── nlp_features.csv             # Same as above but interim
│   ├── emotion_features.csv         # Emotion counts (optional)
│   ├── sentence_embeddings.csv      # Utterance‑level embeddings (384‑d)
│   ├── participant_embeddings.csv   # 384‑d mean vector per participant
│   └── final_features_with_labels.csv   # <— to be produced in next step
├── venv/ (or conda env)
└── requirements.txt
```

---

## 3. Environment Setup

```bash
# create & activate venv (example)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# install core libraries
pip install -r requirements.txt
#   key packages: pandas, nltk, empath, sentence‑transformers, scikit‑learn
```

---

## 4. Step‑by‑Step Pipeline

| Step        | Script / Action                                                | Input                                                                              | Key Operations                                                                                                  | Output                                      |
| ----------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| **2.1**     | `clean_transcripts.py`                                         | Raw `*_TRANSCRIPT.txt`                                                             | • Remove timestamps, filler chars• Lower‑case, strip speaker tags• Save per‑participant clean file              | `EDA/clean_text/XXX_TRANSCRIPT_clean.txt`   |
| **2.2 A/B** | `text_feature_extraction.py` (calls `extract_nlp_features.py`) | Clean text files                                                                   | • LIWC category counts via Empath• Lexical diversity (TTR, MTLD, etc.)• Hedge frequency, cognitive‑load proxies | `features.csv` (participant‑level)          |
| (opt.)      | `filter_emotion_features.py`                                   | `features.csv`                                                                     | • Keep emotion‑relevant LIWC rows only                                                                          | `emotion_features.csv`                      |
| **2.2 C**   | `sentence_embedding_extraction.py`                             | Clean text files                                                                   | • Load `all‑MiniLM‑L6‑v2` model• Encode each utterance (384 dims)                                               | `sentence_embeddings.csv` (utterance‑level) |
|             | `aggregate_embeddings.py`                                      | `sentence_embeddings.csv`                                                          | • Mean‑pool embeddings per participant                                                                          | `participant_embeddings.csv`                |
| **2.3**     | *(to build)* merge script                                      | `features.csv`, `participant_embeddings.csv`, `emotion_features.csv`, `labels.csv` | • Inner‑join on `participant_id`                                                                                | `final_features_with_labels.csv`            |
| **2.4**     | model training notebook / script                               | `final_features_with_labels.csv`                                                   | • Logistic Regression, Random Forest, SVM• Train/val/test split (stratified)                                    | Saved model + metrics                       |
| **2.5**     | anxiety‑risk clf (future)                                      | Depressed subset from 2.4                                                          | • Predict `anxiety` within depressed                                                                            | Metrics + model                             |

---

## 5. Detailed Script Descriptions

### 5.1 `clean_transcripts.py`

- **Goal:** standardise raw DAIC transcripts for NLP.
- **Highlights:**
  - Uses regex to drop `[laugh]`, timestamps, XML tags.
  - Tokenises into sentences with NLTK (for later embedding granularity).

### 5.2 `text_feature_extraction.py` & `extract_nlp_features.py`

- **Goal:** produce participant‑level linguistic features inspired by prior depression‑detection papers.
- **Key Features Generated:**
  - **LIWC‑style counts** (emotion, cognitive process, social words) via Empath categories.
  - **Lexical diversity:** Type‑Token Ratio (TTR), Hapax Legomena %, MTLD.
  - **Hedging & cognitive‑load cues:** frequency of "I guess", "maybe", pronoun shifts, average sentence length.

### 5.3 `sentence_embedding_extraction.py`

- **Goal:** get semantic representation of every utterance.
- **Model:** `all‑MiniLM‑L6‑v2` (Sentence‑Transformers; 384‑d; 90 M params).
- **Process:** loops through each cleaned file, encodes lines, appends to list, saves `sentence_embeddings.csv`.

### 5.4 `aggregate_embeddings.py`

- **Goal:** convert utterance‑level embeddings to a single participant vector.
- **Pooling:** simple mean across utterances (empirically solid baseline; other options = CLS token, max‑pool).

### 5.5 Feature Merge Script (TBD)

Will:

```python
merged = (pd.read_csv('features.csv')
            .merge(pd.read_csv('participant_embeddings.csv'), on='participant_id')
            .merge(pd.read_csv('labels.csv'), on='participant_id'))
merged.to_csv('final_features_with_labels.csv', index=False)
```

---

## 6. Data Provenance Summary

| File                             | Rows                 | Level       | Notes                           |
| -------------------------------- | -------------------- | ----------- | ------------------------------- |
| `clean_text/*.txt`               | \~3–8 k lines / file | Utterance   | Raw text cleaned                |
| `sentence_embeddings.csv`        | Σ utterances         | Utterance   | 384‑d vectors + metadata        |
| `participant_embeddings.csv`     | \~150 participants   | Participant | Mean of embeddings              |
| `features.csv`                   | same                 | Participant | LIWC + lexical (\~50 dims)      |
| `emotion_features.csv`           | same                 | Participant | Optional subset of features     |
| `final_features_with_labels.csv` | same                 | Participant | (+) depression / anxiety labels |

---

## 7. Next To‑Dos

1. **Generate **`` using PHQ‑8 & GAD‑7 as discussed.
2. **Merge features** (§5.5) to create `final_features_with_labels.csv`.
3. **Train baseline classifiers** (LogReg / SVM / RandomForest); log metrics (accuracy, F1, AUC).
4. **Hyper‑tune models**; feature importance analysis.
5. (**Optional**) Visualise embeddings with t‑SNE/PCA for cluster sanity‑check.
6. **Sub‑task 2.5:** Train anxiety‑risk model on depressed subset.

---

## 8. Repro Tips

- Always fix random seeds (`numpy`, `torch`, `sklearn`) for reproducible splits.
- Cache heavy models in `~/.cache/` or project root `.models` to avoid re‑download.
- Track package versions in `requirements.txt` after each new install.

---

## 9. Changelog

| Date        | Author           | Change                                                                       |
| ----------- | ---------------- | ---------------------------------------------------------------------------- |
| 10 Jun 2025 | Avreet + ChatGPT | Added LIWC + lexical extraction pipeline.                                    |
| 11 Jun 2025 | Avreet + ChatGPT | Added sentence embeddings + participant aggregation; initial docs generated. |

---

> **Legend:** Steps correspond to Avreet’s original project outline (2.1 – 2.5).  🏁

