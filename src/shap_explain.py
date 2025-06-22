import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------------
# 1. Load data and trained pipeline
# ------------------------------------------------------------------
X = np.load("data/processed/X_features.npy")
feature_names = np.load("data/processed/feature_names.npy")
y = np.load("data/processed/y.npy")
pipeline = joblib.load("models/random_forest_eeg.pkl")

scaler = pipeline.named_steps["standardscaler"]
model  = pipeline.named_steps["randomforestclassifier"]

X_scaled = scaler.transform(X)

# ------------------------------------------------------------------
# 2. SHAP for tree model
# ------------------------------------------------------------------
explainer = shap.TreeExplainer(model)
shap_values_full = explainer.shap_values(X_scaled)

# Handle binary vs multi-class return types
# For RandomForestClassifier (binary) shap_values is usually a list of length 2
if isinstance(shap_values_full, list):
    shap_values = shap_values_full[1]          # class 1 (eye open / high load)
else:
    shap_values = shap_values_full            # already an array

# ------------------------------------------------------------------
# 3. Fix possible bias-column mismatch
# ------------------------------------------------------------------
if shap_values.shape[1] == X_scaled.shape[1] + 1:
    print("ℹ️  Detected extra bias column in SHAP values. Removing last column.")
    shap_values = shap_values[:, :-1]

assert shap_values.shape[1] == X_scaled.shape[1], \
    "SHAP values still mismatch feature matrix after adjustment!"

# ------------------------------------------------------------------
# 4. Plots
# ------------------------------------------------------------------
Path("reports/figures").mkdir(parents=True, exist_ok=True)

plt.figure()
shap.summary_plot(shap_values, X_scaled,feature_names=feature_names, show=False)
plt.title("SHAP Summary – Class 1")
plt.tight_layout()
plt.savefig("reports/figures/shap_summary_class1.png")
plt.close()

plt.figure()
shap.summary_plot(shap_values, X_scaled,feature_names=feature_names, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("reports/figures/shap_feature_importance_class1.png")
plt.close()
print("✅ SHAP explainability plots saved to reports/figures/")
