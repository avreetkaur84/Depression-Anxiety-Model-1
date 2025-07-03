# 🧠 Transcript-Based Mental Health Screening App

This project is an AI-powered web application that allows users to answer a short questionnaire. Their responses are analyzed using NLP and machine learning models to **predict signs of depression and anxiety**. It's designed for early mental health screening, inspired by PHQ-8 and real-world conversational data.

---

## 📁 Project Structure

```

Depression-Anxiety-Model-1/
│
├── frontend/                  ← Next.js frontend (React)
│   └── app/                   ← Pages and components
│       ├── page.js            ← Main UI with questionnaire + results
│       └── components/        ← Reusable UI component (QuestionForm)
│
├── model/                     ← Python backend (FastAPI)
│   ├── main.py                ← FastAPI app
│   ├── predictor.py           ← Feature engineering + model prediction
│   ├── depression\_model.pkl   ← Trained depression model (Random Forest)
│   ├── anxiety\_model.pkl      ← Trained anxiety model (Voting Ensemble)
│   ├── minmax\_scaler.pkl      ← Scaler for anxiety model
│   ├── requirements.txt       ← Python dependencies
│   ├── depression\_feature\_order.csv
│   └── anxiety\_feature\_order.csv
│
├── EDA/                       ← Local exploratory data analysis
├── data/                      ← Raw transcript data (not uploaded)
└── README.md

````

---

## 💻 How It Works

1. **User answers questions** (as free-text input)
2. Text is combined into a single transcript
3. Sent to backend via API
4. Backend extracts:
   - Empath emotional features
   - Sentence embeddings (384-dim)
   - Lexical features (word count, negations, etc.)
5. Trained models predict:
   - Depression (binary)
   - Anxiety (only if depressed)

---

## 🧠 Models Used

| Task        | Model               |
|-------------|---------------------|
| Depression  | Random Forest       |
| Anxiety     | Voting Classifier   |
| Embeddings  | all-MiniLM-L6-v2 (sentence-transformers) |
| Emotional Features | Empath         |

---

## 🚀 Running the Project Locally

### 1. Backend (FastAPI)
```bash
cd model
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn main:app --reload
````

This starts the API server at `http://127.0.0.1:8000`.

### 2. Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

This starts the UI at `http://localhost:3000`.

---

## 🌐 Deployment Plan

* **Frontend:** [Vercel](https://vercel.com) — deploys the Next.js app
* **Backend:** [Railway](https://railway.app) — deploys the FastAPI service
  (`uvicorn main:app --host 0.0.0.0 --port $PORT`)

Make sure to update the fetch URL in `frontend/app/page.js`:

```js
fetch("https://your-backend-url.onrailway.app/predict", ...)
```

---

## 📊 Dataset & Features

* Based on real conversation-style **transcripts**
* Extracted **399+ features**:

  * Emotional categories (via Empath)
  * 384-D sentence embeddings
  * Lexical stats (hedges, negations, etc.)

---

## 📦 Dependencies

* `fastapi`, `uvicorn`
* `pandas`, `numpy`, `scikit-learn`
* `empath`, `nltk`, `sentence-transformers`, `joblib`

---

## 📌 Future Improvements

* Add severity scoring (PHQ-8 regression)
* Improve response clarity and feedback
* Build user account system
* Switch to a more memory-efficient embedding model
* Host both backend & frontend together (Docker or serverless)

---

## 🙌 Authors & Credits

* **Avreet Kaur** — AI, backend & frontend
* Built for submission to **Intel AI for Youth** 🧠🚀

---

> ⚠️ This tool is **not a medical diagnosis system**. It is intended for research and educational use only.

