import os
import pandas as pd
from pathlib import Path
from empath import Empath

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