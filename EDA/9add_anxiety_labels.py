import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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
