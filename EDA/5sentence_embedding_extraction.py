import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path

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