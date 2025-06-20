import os
from pathlib import Path
import pandas as pd
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
import nltk

# Tokenizers
sent_tokenize = PunktSentenceTokenizer().tokenize
word_tokenize = TreebankWordTokenizer().tokenize

# Paths
CLEAN_TEXT_FOLDER = Path("EDA/clean_text")
OUTPUT_PATH = Path("EDA/nlp_features.csv")

# Feature lexicons
hedges = {"maybe", "perhaps", "possibly", "sort of", "kind of", "somewhat", "seems", "likely"}
negations = {"no", "not", "never", "don't", "can't", "isn't", "won't", "didn't", "doesn't"}
cognitive_load = {"think", "know", "believe", "understand", "realize", "consider", "assume"}

# Collect feature rows
rows = []

for file in CLEAN_TEXT_FOLDER.glob("*.txt"):
    participant_id = file.stem.replace("_clean", "")
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        print(f"Skipping {file.name}: empty text")
        continue

    # Tokenize
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    total_words = len(words)
    unique_words = len(set(words))
    num_sentences = len(sentences)

    # Extract features
    lexical_diversity = unique_words / total_words if total_words else 0
    avg_sentence_length = total_words / num_sentences if num_sentences else 0

    # Lowercased word list for matching
    word_list = [w.lower() for w in words]

    hedge_count = sum(1 for h in hedges if h in text.lower())
    negation_count = sum(word_list.count(w) for w in negations)
    cogload_count = sum(word_list.count(w) for w in cognitive_load)

    # Store features
    rows.append({
        "participant": participant_id,
        "word_count": total_words,
        "lexical_diversity": round(lexical_diversity, 4),
        "avg_sentence_length": round(avg_sentence_length, 2),
        "hedge_count": hedge_count,
        "negation_count": negation_count,
        "cognitive_load": cogload_count,
    })

# Save to CSV
df_features = pd.DataFrame(rows)
df_features.to_csv(OUTPUT_PATH, index=False)
print(f"✅ NLP features extracted and saved to {OUTPUT_PATH}")