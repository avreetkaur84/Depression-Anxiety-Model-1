import pandas as pd
from pathlib import Path

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