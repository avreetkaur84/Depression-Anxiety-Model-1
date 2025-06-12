import os
import pandas as pd
import re
from pathlib import Path
from nltk.tokenize import sent_tokenize, word_tokenize
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
