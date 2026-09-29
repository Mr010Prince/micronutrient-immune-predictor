import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

PIPELINE_PATH = os.path.join("backend", "ml", "model.joblib")
TEST_PATH = os.path.join("data", "processed", "test.csv")
OUT_PATH = os.path.join("ml", "results", "shap_summary.png")

pipe = joblib.load(PIPELINE_PATH)
print(f"Loaded pipeline: {PIPELINE_PATH}")

df = pd.read_csv(TEST_PATH)
with open(os.path.join("backend", "ml", "feature_order.json")) as f:
    import json
    feature_order = json.load(f)

X = df[feature_order]
print(f"Test data: {X.shape}")

# Impute + scale with the pipeline steps, then run TreeExplainer on the XGBoost model
X_trans = pipe[:-1].transform(X)
model = pipe.named_steps["model"]
print(f"Explaining XGBoost step: {type(model).__name__}")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_trans)
sv = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values

mean_abs = np.abs(sv).mean(axis=0)
top10 = pd.Series(mean_abs, index=feature_order).sort_values(ascending=False).head(10)
print("\nTop 10 features by mean absolute SHAP value:")
print(top10.round(5).to_string())

plt.figure(figsize=(8, 6))
plt.barh(top10.index[::-1], top10.values[::-1], color="tab:blue")
plt.xlabel("Mean |SHAP value|")
plt.title("SHAP feature importance (XGBoost, test set)")
plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=150)
plt.close()
print(f"\nSaved SHAP summary plot: {OUT_PATH}")