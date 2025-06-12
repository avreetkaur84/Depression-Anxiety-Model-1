import pandas as pd

# Load sentence embeddings
df = pd.read_csv("EDA/sentence_embeddings.csv")

# Drop text and sentence_id
embedding_cols = [col for col in df.columns if col.startswith("embed_")]

# Average all sentence embeddings per participant
agg_df = df.groupby("participant_id")[embedding_cols].mean().reset_index()

# Save
agg_df.to_csv("EDA/participant_embeddings.csv", index=False)
print("✅ Participant-level embeddings saved to: EDA/participant_embeddings.csv")