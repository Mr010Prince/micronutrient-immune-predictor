import os
import json
import warnings

import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, roc_curve,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# --- Column mapping (verified against data/interim/merged.csv header) ---
FEATURE_COLS = {
    "RIDAGEYR": "age",
    "RIAGENDR": "gender",
    "RIDRETH3": "race_ethnicity",
    "INDFMPIR": "income_poverty_ratio",
    "LBXVIA": "vitamin_A",
    "LBXVIE": "vitamin_E",
    "LBXVIC": "vitamin_C",
    "LBXVIDMS": "vitamin_D_25OH",
    "LBXFER": "ferritin",
    "LBXIRN": "serum_iron",
    "LBDTIB": "TIBC",
    "LBDPCT": "transferrin_saturation",
    "LBXTFR": "transferrin_receptor",
    "LBDRFO": "RBC_folate",
    "LBDFOT": "serum_total_folate",
    "LBXBSE": "blood_selenium",
}
CRP_COL = "LBXHSCRP"


def build_target(crp):
    """1 if CRP >= 3 and < 10 (elevated, non-acute); 0 if CRP < 1; else NaN (drop)."""
    t = pd.Series(np.nan, index=crp.index)
    t[(crp >= 3) & (crp < 10)] = 1
    t[crp < 1] = 0
    return t


def sens_spec(y, yhat):
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return tp / (tp + fn), tn / (tn + fp)


def main():
    print("=" * 70)
    print("Micronutrient Immune Dysfunction Predictor - training")
    print("=" * 70)

    df = pd.read_csv(os.path.join("data", "interim", "merged.csv"))
    print(f"Loaded merged.csv: {df.shape[0]} rows, {df.shape[1]} columns")

    present = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    print(f"\nFeature columns used ({len(present)}): {present}")
    if missing:
        print(f"NOT found in CSV (skipped): {missing}")
    assert CRP_COL in df.columns, f"{CRP_COL} not in CSV"

    # Adults only
    df = df[df["RIDAGEYR"] >= 18].copy()
    print(f"\nAdults (RIDAGEYR >= 18): {len(df)} rows")

    # Target
    df["immune_dysfunction"] = build_target(df[CRP_COL].astype(float))
    df = df[df["immune_dysfunction"].notna()].copy()
    df["immune_dysfunction"] = df["immune_dysfunction"].astype(int)
    print(f"After target filtering (CRP < 1 -> 0, 3 <= CRP < 10 -> 1): {len(df)} rows")
    print(f"Class counts: {df['immune_dysfunction'].value_counts().to_dict()}")

    # Features (explicitly excluding LBXHSCRP and all CRP/weight/design/SEQN/comment columns)
    X = df[present].rename(columns=FEATURE_COLS)
    y = df["immune_dysfunction"]
    print(f"\nCRP column '{CRP_COL}' excluded from features. "
          f"Weights/design/SEQN/comment columns excluded.")

    # Drop features with > 40% missing
    miss_frac = X.isna().mean()
    dropped = miss_frac[miss_frac > 0.40].index.tolist()
    X = X.drop(columns=dropped)
    print(f"Features dropped (>40% missing): {dropped if dropped else 'none'}")
    print(f"Final feature list ({X.shape[1]}): {list(X.columns)}")
    print(f"\nMissingness per feature:\n{X.isna().mean().sort_values().round(3)}")

    feature_names = list(X.columns)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train class balance: {np.bincount(y_train)} | Test: {np.bincount(y_test)}")

    # Models (fixed hyperparameters)
    def make_pipeline(clf):
        return Pipeline([
            ("imputer", IterativeImputer(random_state=RANDOM_STATE, max_iter=10)),
            ("scaler", StandardScaler()),
            ("model", clf),
        ])

    models = {
        "LogisticRegression": make_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
        ),
        "RandomForest": make_pipeline(
            RandomForestClassifier(
                n_estimators=500, max_depth=None, min_samples_leaf=5,
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
            )
        ),
        "XGBoost": make_pipeline(
            XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8, scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()),
                eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
            )
        ),
        "SVC": make_pipeline(
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)
        ),
        "KNN": make_pipeline(
            KNeighborsClassifier(n_neighbors=15, n_jobs=-1)
        ),
        "MLP": make_pipeline(
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_STATE)
        ),
    }

    # 10-fold stratified CV ROC-AUC on train
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    print("\n--- Stratified 10-fold CV ROC-AUC (training set) ---")
    cv_scores = {}
    for name, pipe in models.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_scores[name] = scores
        print(f"{name:>20}: {scores.mean():.4f} +/- {scores.std():.4f}")

    # Test metrics
    print("\n--- Test set metrics ---")
    results, roc_data, fitted = [], {}, {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, proba)
        acc = accuracy_score(y_test, pred)
        sens, spec = sens_spec(y_test, pred)
        cm = confusion_matrix(y_test, pred)
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_data[name] = (fpr, tpr, auc)
        fitted[name] = pipe
        results.append({
            "model": name, "cv_auc_mean": round(cv_scores[name].mean(), 4),
            "cv_auc_std": round(cv_scores[name].std(), 4), "test_roc_auc": round(auc, 4),
            "test_accuracy": round(acc, 4), "test_sensitivity": round(sens, 4),
            "test_specificity": round(spec, 4),
        })
        print(f"{name}: AUC={auc:.4f} acc={acc:.4f} sens={sens:.4f} spec={spec:.4f}")
        print(f"    confusion matrix [[TN FP][FN TP]]:\n{cm}")

    # Metrics table
    os.makedirs(os.path.join("ml", "results"), exist_ok=True)
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join("ml", "results", "metrics.csv"), index=False)
    print("\nMetrics table:")
    print(res_df.to_string(index=False))

    # ROC plot
    plt.figure(figsize=(7, 6))
    colors = {"LogisticRegression": "tab:blue", "RandomForest": "tab:green", "XGBoost": "tab:red",
              "SVC": "tab:purple", "KNN": "tab:orange", "MLP": "tab:brown"}
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=colors[name])
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves - immune dysfunction (CRP-based target)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join("ml", "results", "roc_curves.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"\nSaved ROC plot: {roc_path}")

    # Best model by test ROC-AUC -> backend/ml/model.joblib
    best_name = res_df.sort_values("test_roc_auc", ascending=False).iloc[0]["model"]
    print(f"\nBest model by test ROC-AUC: {best_name}")

    os.makedirs(os.path.join("backend", "ml"), exist_ok=True)
    joblib.dump(fitted[best_name], os.path.join("backend", "ml", "model.joblib"))
    with open(os.path.join("backend", "ml", "feature_order.json"), "w") as f:
        json.dump(feature_names, f, indent=2)
    print("Saved model + feature order to backend/ml/")

    # Processed data
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    pd.concat([X_train, y_train], axis=1).to_csv(
        os.path.join("data", "processed", "train.csv"), index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(
        os.path.join("data", "processed", "test.csv"), index=False)
    print("Saved processed train/test to data/processed/")


if __name__ == "__main__":
    main()
