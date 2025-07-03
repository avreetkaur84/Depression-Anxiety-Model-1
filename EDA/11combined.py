import os
import pandas as pd
import re
from pathlib import Path
from empath import Empath
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer, TreebankWordTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import nltk

# Download NLTK data if not already present
nltk.download('punkt')

# ----- Paths -----
DATA_FOLDER = Path("data/transcript")
CLEAN_TEXT_FOLDER = Path("EDA/clean_text")
CLEAN_TEXT_FOLDER.mkdir(parents=True, exist_ok=True)

# ----- Helper Function: Clean Text -----
def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove non-alphabetic characters (keep spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ----- Process Each Transcript -----
for file in DATA_FOLDER.glob("*.csv"):
    df = pd.read_csv(file, sep='\t')

   # Clean column names (remove leading/trailing spaces or BOM)
    df.columns = df.columns.str.strip().str.lower()

    if not {'speaker', 'value'}.issubset(df.columns):
        print(f"Skipping {file.name}: missing or misnamed columns: {df.columns.tolist()}")
        continue


    # Filter only participant's speech
    df = df[df['speaker'].str.lower() == 'participant']
    print(f"{file.name} → Participant rows: {len(df)}")

    # Clean each utterance
    df['cleaned'] = df['value'].astype(str).apply(clean_text)

    # Drop empty utterances (just in case)
    df = df[df['cleaned'].str.strip() != '']

    # Combine all cleaned text into one file
    full_text = " ".join(df['cleaned'].tolist())
    print(f"{file.name} → Final cleaned length: {len(full_text.split())} words")


    # Save to clean_text/ folder
    participant_id = file.stem  # e.g., "P101"
    output_path = CLEAN_TEXT_FOLDER / f"{participant_id}_clean.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"✅ Cleaned transcript saved: {output_path.name}")


#  Tokenizers
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



# Setup
lexicon = Empath()
CLEAN_TEXT_FOLDER = Path("EDA/clean_text")
FEATURE_OUTPUT = Path("EDA/features.csv")

# Prepare results
rows = []

# Process each cleaned text file
for file in CLEAN_TEXT_FOLDER.glob("*.txt"):
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()

    # Use Empath to analyze text
    empath_features = lexicon.analyze(text, normalize=True)

    # Add filename as participant ID
    participant_id = file.stem.replace("_clean", "")
    row = {"participant": participant_id, **empath_features}
    rows.append(row)

# Convert to DataFrame and save
df = pd.DataFrame(rows)
df.to_csv(FEATURE_OUTPUT, index=False)
print(f"✅ Empath features saved to {FEATURE_OUTPUT}")


# Load original Empath features
empath_df = pd.read_csv("EDA/features.csv")

# Define selected emotion-related features
emotion_features = [
    'participant',
    'sadness', 'anger', 'joy', 'fear', 'nervousness', 'affection',
    'negative_emotion', 'positive_emotion', 'confusion', 'cheerfulness',
    'disgust', 'torment', 'pain', 'shame', 'exasperation'
]

# Filter the dataframe
filtered_df = empath_df[[col for col in emotion_features if col in empath_df.columns]]

# Save filtered file
filtered_df.to_csv("EDA/emotion_features.csv", index=False)
print("✅ Filtered emotion features saved to EDA/emotion_features.csv")


# Initialize model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Folder with cleaned transcripts
clean_text_folder = Path("EDA/clean_text")

# Prepare to collect all data
all_embeddings = []
all_metadata = []

# Iterate through each file
for file in clean_text_folder.glob("*_TRANSCRIPT_clean.txt"):
    participant_id = file.stem.split("_")[0]

    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]  # Remove empty lines

        # Generate embeddings for each line (utterance)
        embeddings = model.encode(lines)

        for i, (text, embed) in enumerate(zip(lines, embeddings)):
            all_metadata.append({
                "participant_id": participant_id+"_TRANSCRIPT",
                "sentence_id": i,
                "text": text
            })
            all_embeddings.append(embed)

# Convert to DataFrames
metadata_df = pd.DataFrame(all_metadata)
embedding_df = pd.DataFrame(all_embeddings)
embedding_df.columns = [f"embed_{i}" for i in range(embedding_df.shape[1])]

# Combine metadata + embeddings
final_df = pd.concat([metadata_df, embedding_df], axis=1)

# Save
final_df.to_csv("EDA/sentence_embeddings.csv", index=False)
print("✅ Sentence embeddings saved to: EDA/sentence_embeddings.csv")


# Load sentence embeddings
df = pd.read_csv("EDA/sentence_embeddings.csv")

# Drop text and sentence_id
embedding_cols = [col for col in df.columns if col.startswith("embed_")]

# Average all sentence embeddings per participant
agg_df = df.groupby("participant_id")[embedding_cols].mean().reset_index()

# Save
agg_df.to_csv("EDA/participant_embeddings.csv", index=False)
print("✅ Participant-level embeddings saved to: EDA/participant_embeddings.csv")


# --- Load data ---
features = pd.read_csv("EDA/emotion_features.csv")  # LIWC + lexical
embeddings = pd.read_csv("EDA/participant_embeddings.csv")  # 384-d vectors
labels = pd.read_csv("EDA/labels.csv")  # contains only valid labeled participants

print("✅ Columns in emotion_features.csv:", features.columns.tolist(), end="\n\n")
print("✅ Columns in participant_embeddings.csv:", embeddings.columns.tolist(), end="\n\n")
print("✅ Columns in labels.csv:", labels.columns.tolist(), end="\n\n")

features.rename(columns={'participant': 'participant_id'}, inplace=True)

features.columns = features.columns.str.strip().str.lower()
embeddings.columns = embeddings.columns.str.strip().str.lower()
labels.columns = labels.columns.str.strip().str.lower()


# --- Sanity check ---
common_ids = set(features['participant_id']) & set(embeddings['participant_id']) & set(labels['participant_id'])

# Warn if some IDs are missing
missing_in_labels = set(features['participant_id']) - set(labels['participant_id'])
if missing_in_labels:
    print(f"⚠️ Warning: {len(missing_in_labels)} participants missing in labels.csv — they will be dropped.")
    print(f"Missing IDs: {sorted(list(missing_in_labels))[:5]}{' ...' if len(missing_in_labels) > 5 else ''}")

# --- Filter all dataframes to common IDs only ---
features = features[features['participant_id'].isin(common_ids)]
embeddings = embeddings[embeddings['participant_id'].isin(common_ids)]
labels = labels[labels['participant_id'].isin(common_ids)]

# --- Merge ---
merged = features.merge(embeddings, on="participant_id").merge(labels, on="participant_id")

# --- Save final dataset ---
merged.to_csv("EDA/final_features_with_labels.csv", index=False)

print(f"✅ Done. Final dataset saved to EDA/final_features_with_labels.csv ({len(merged)} rows)")



# Assuming this is your full data file (with embeddings + emotional features + labels)
df = pd.read_csv("EDA/final_features_with_labels.csv")

# Keep only rows where PHQ8_Binary == 1 (i.e., depressed)
depressed_df = df[df['phq8_binary'] == 1].copy()

# Save to a new CSV file for phase 2 (anxiety detection)
depressed_df.to_csv("EDA/depressed_only_features.csv", index=False)


# Load the depressed participants dataset
depressed_df = pd.read_csv("EDA/depressed_only_features.csv")

# 🔹 Step 1: Define anxiety-related emotional features
anxiety_features = ['nervousness', 'fear', 'confusion', 'sadness', 'negative_emotion', 'anger']

# 🔹 Step 2: Normalize these features to 0–1 using MinMaxScaler
scaler = MinMaxScaler()
depressed_df[anxiety_features] = scaler.fit_transform(depressed_df[anxiety_features])

# 🔹 Step 3: Create anxiety_score (row-wise mean of anxiety features)
depressed_df['anxiety_score'] = depressed_df[anxiety_features].mean(axis=1)

# 🔹 Step 4: Create binary label using threshold (default: 0.5)
# Calculate the threshold so that top 60% are labeled as 'anxious'
threshold = depressed_df['anxiety_score'].quantile(0.4)  # bottom 40% → 0, top 60% → 1

# Apply binary labeling based on threshold
depressed_df['anxiety_binary'] = depressed_df['anxiety_score'].apply(lambda x: 1 if x > threshold else 0)


# 🔹 Optional: Save the updated file
depressed_df.to_csv("EDA/depressed_with_anxiety_labels.csv", index=False)


