import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

# ----- PHASE 1: Depression Model (Random Forest) -----

print("📥 Loading depression dataset...")
df = pd.read_csv("../EDA/final_features_with_labels.csv")

# Drop identifier columns
X_depression = df.drop(columns=["participant_id", "phq8_score", "phq8_binary"])
y_depression = df["phq8_binary"]

# Train Random Forest
print("🌲 Training Random Forest for depression classification...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_depression, y_depression)

# Save the model
joblib.dump(rf_model, "depression_model.pkl")
print("✅ Depression model saved as models/depression_model.pkl")


# ----- PHASE 2: Anxiety Model (Voting Classifier) -----

print("\n📥 Loading anxiety dataset...")
df_anx = pd.read_csv("../EDA/depressed_with_anxiety_labels.csv")

# Anxiety features used (same ones you normalized)
anxiety_features = ['nervousness', 'fear', 'confusion', 'sadness', 'negative_emotion', 'anger']
X_anxiety = df_anx[anxiety_features]
y_anxiety = df_anx['anxiety_binary']

# Normalize with MinMaxScaler
print("📊 Normalizing anxiety features...")
scaler = MinMaxScaler()
X_anxiety_scaled = scaler.fit_transform(X_anxiety)

# Save scaler
joblib.dump(scaler, "minmax_scaler.pkl")
print("✅ Scaler saved as models/minmax_scaler.pkl")

# Voting Classifier
print("🗳️ Training VotingClassifier for anxiety...")
clf1 = LogisticRegression()
clf2 = RandomForestClassifier(n_estimators=100)
clf3 = SVC(probability=True)

voting_clf = VotingClassifier(estimators=[
    ('lr', clf1),
    ('rf', clf2),
    ('svc', clf3)
], voting='soft')

voting_clf.fit(X_anxiety_scaled, y_anxiety)

# Save the model
joblib.dump(voting_clf, "anxiety_model.pkl")
print("✅ Anxiety model saved as models/anxiety_model.pkl")


# Save feature order
pd.Series(X_depression.columns).to_csv("depression_feature_order.csv", index=False)

pd.Series(anxiety_features).to_csv("anxiety_feature_order.csv", index=False)
