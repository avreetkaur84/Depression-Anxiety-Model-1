import pandas as pd

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