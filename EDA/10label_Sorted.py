import pandas as pd

# Load the label file
df = pd.read_csv("EDA/labels.csv")

# Extract numeric part from participant_id and store it in a new column
df["participant_num"] = df["participant_id"].str.extract(r"(\d+)").astype(int)

# Sort by the numeric value
df_sorted = df.sort_values("participant_num")

# Drop the temporary column if not needed
df_sorted = df_sorted.drop(columns=["participant_num"])

# Save the sorted file (optional)
df_sorted.to_csv("label_sorted.csv", index=False)

# Display the sorted DataFrame
print(df_sorted)