# src/train_model.py

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ------------------------------------------------------------------
# 1. Load features & labels
# ------------------------------------------------------------------
X = np.load("data/processed/X_features.npy")
y = np.load("data/processed/y.npy")

# ------------------------------------------------------------------
# 2. Train-test split (stratified)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------------------------------------------------
# 3. Build pipeline: StandardScaler → RandomForest
# ------------------------------------------------------------------
clf = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",   # handles slight label imbalance
        n_jobs=-1,
        random_state=42,
    ),
)

# ------------------------------------------------------------------
# 4. Train
# ------------------------------------------------------------------
clf.fit(X_train, y_train)

# ------------------------------------------------------------------
# 5. Evaluate
# ------------------------------------------------------------------
y_pred = clf.predict(X_test)
print("✅ Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

Path("reports/figures").mkdir(parents=True, exist_ok=True)
plt.savefig("reports/figures/confusion_matrix.png")
plt.show()

# ------------------------------------------------------------------
# 6. Save model
# ------------------------------------------------------------------
Path("models").mkdir(exist_ok=True)
joblib.dump(clf, "models/random_forest_eeg.pkl")
print("✅ Model saved to models/random_forest_eeg.pkl")
