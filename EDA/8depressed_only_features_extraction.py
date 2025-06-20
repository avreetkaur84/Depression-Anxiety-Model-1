import pandas as pd

# Assuming this is your full data file (with embeddings + emotional features + labels)
df = pd.read_csv("EDA/final_features_with_labels.csv")

# Keep only rows where PHQ8_Binary == 1 (i.e., depressed)
depressed_df = df[df['phq8_binary'] == 1].copy()

# Save to a new CSV file for phase 2 (anxiety detection)
depressed_df.to_csv("EDA/depressed_only_features.csv", index=False)

