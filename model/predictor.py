import os
import joblib
import pandas as pd
import numpy as np
import re
from empath import Empath
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer, TreebankWordTokenizer
from sentence_transformers import SentenceTransformer

# 🔹 Load models
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model"))

depression_model = joblib.load(os.path.join(BASE_PATH, "depression_model.pkl"))
anxiety_model = joblib.load(os.path.join(BASE_PATH, "anxiety_model.pkl"))
scaler = joblib.load(os.path.join(BASE_PATH, "minmax_scaler.pkl"))

# 🔹 Initialize tools
lexicon = Empath()
sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2")
sent_tokenize = PunktSentenceTokenizer().tokenize
word_tokenize = TreebankWordTokenizer().tokenize

# 🔹 Predefined feature sets
hedges = {"maybe", "perhaps", "possibly", "sort of", "kind of", "somewhat", "seems", "likely"}
negations = {"no", "not", "never", "don't", "can't", "isn't", "won't", "didn't", "doesn't"}
cognitive_load = {"think", "know", "believe", "understand", "realize", "consider", "assume"}
anxiety_features = ['nervousness', 'fear', 'confusion', 'sadness', 'negative_emotion', 'anger']

# 🔹 Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 🔹 Predict function
def predict_transcript(raw_text: str):
    if not raw_text.strip():
        return {"error": "Empty input."}

    cleaned_text = clean_text(raw_text)

    # ✳️ Lexical features
    sentences = sent_tokenize(cleaned_text)
    words = word_tokenize(cleaned_text)

    total_words = len(words)
    unique_words = len(set(words))
    num_sentences = len(sentences)

    lexical_diversity = unique_words / total_words if total_words else 0
    avg_sentence_length = total_words / num_sentences if num_sentences else 0

    word_list = [w.lower() for w in words]
    hedge_count = sum(1 for h in hedges if h in word_list)
    negation_count = sum(word_list.count(w) for w in negations)
    cogload_count = sum(word_list.count(w) for w in cognitive_load)

    lexical = {
        "word_count": total_words,
        "lexical_diversity": round(lexical_diversity, 4),
        "avg_sentence_length": round(avg_sentence_length, 2),
        "hedge_count": hedge_count,
        "negation_count": negation_count,
        "cognitive_load": cogload_count,
    }

    # ✳️ Empath features
    empath = lexicon.analyze(cleaned_text, normalize=True)

    # ✳️ Sentence Embedding
    sentences_cleaned = [s for s in sentences if s.strip()]
    if not sentences_cleaned:
        return {"error": "Insufficient content for embedding."}

    embeddings = sentence_encoder.encode(sentences_cleaned)
    embedding_mean = np.mean(embeddings, axis=0)
    embed_dict = {f"embed_{i}": val for i, val in enumerate(embedding_mean)}

    # 🔹 Combine all features
    full_input = {**lexical, **empath, **embed_dict}
    input_df = pd.DataFrame([full_input])
    input_df.fillna(0, inplace=True)

    # 🔹 Manually enforce depression feature order
    depression_feature_order = [
        'sadness', 'anger', 'joy', 'fear', 'nervousness', 'affection',
        'negative_emotion', 'positive_emotion', 'confusion', 'cheerfulness',
        'disgust', 'torment', 'pain', 'shame', 'exasperation'
    ] + [f"embed_{i}" for i in range(384)]

    # Filter and reorder columns
    missing_cols = set(depression_feature_order) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0.0

    input_df = input_df[depression_feature_order]

    # 🔹 Depression Prediction
    depression_pred = depression_model.predict(input_df)[0]


    result = {
        "depression_predicted": int(depression_pred),
    }

    # 🔹 Anxiety Prediction (only if depressed)
    if depression_pred == 1:
        anxiety_feature_order = [
            "nervousness", "fear", "confusion", "sadness", "negative_emotion", "anger"
        ]
        anx_input = input_df[anxiety_feature_order].copy()
        anx_scaled = scaler.transform(anx_input)

        anxiety_pred = anxiety_model.predict(anx_scaled)[0]
        result["anxiety_predicted"] = int(anxiety_pred)
    else:
        result["anxiety_predicted"] = "Not evaluated (not depressed)"

    return result


if __name__ == "__main__":
    sample_input = "I don't feel like doing anything and I can't sleep well lately."
    output = predict_transcript(sample_input)
    print(output)