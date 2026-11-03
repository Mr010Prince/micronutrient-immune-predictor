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
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, roc_curve,
)
from xgboost import XGBClassifier

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical

    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution
    from optuna.integration import OptunaSearchCV

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

    # Models (hyperparameter tuning)
    def make_pipeline(clf):
        return Pipeline([
            ("imputer", IterativeImputer(random_state=RANDOM_STATE, max_iter=10)),
            ("scaler", StandardScaler()),
            ("model", clf),
        ])

    def pl(clf):
        return make_pipeline(clf)

    tune_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid_searches = {}

    # --- GridSearchCV models ---
    pipe = pl(LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE))
    grid_searches["LogisticRegression"] = (
        pipe,
        {
            f"model__C": [0.01, 0.1, 1, 10, 100],
            f"model__penalty": ["l1", "l2"],
            f"model__solver": ["liblinear", "saga"],
        },
    )
    pipe = pl(RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))
    grid_searches["RandomForest"] = (
        pipe,
        {
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [1, 5, 10],
            "model__min_samples_split": [2, 10],
        },
    )
    pipe = pl(SVC(
        kernel="rbf", probability=True, class_weight="balanced",
        random_state=RANDOM_STATE))
    grid_searches["SVC"] = (
        pipe,
        {
            "model__C": [0.1, 1, 10, 100],
            "model__gamma": ["scale", "auto", 0.01, 0.1],
        },
    )
    pipe = pl(KNeighborsClassifier(n_jobs=-1))
    grid_searches["KNN"] = (
        pipe,
        {
            "model__n_neighbors": [5, 10, 15, 25, 35],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
    )

    spw = float((y_train == 0).sum() / (y_train == 1).sum())

    def tune_bayes(name, clf, spaces, n_trials=40):
        """Bayesian optimization via skopt BayesSearchCV, or optuna fallback."""
        pipe = make_pipeline(clf)
        if HAS_SKOPT:
            search = BayesSearchCV(
                pipe, spaces, n_iter=n_trials, cv=tune_cv,
                scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE, verbose=0,
            )
        else:
            optuna_sp = {}
            for pname, (dist, *_) in spaces.items():
                optuna_sp[f"model__{pname}"] = dist
            search = OptunaSearchCV(
                pipe, optuna_sp, cv=tune_cv, scoring="roc_auc",
                n_trials=n_trials, timeout=3600, n_jobs=-1,
                random_state=RANDOM_STATE, verbose=0,
            )
        search.fit(X_train, y_train)
        print(f"{name} best hyperparameters (Bayesian): {search.best_params_}")
        print(f"    tuning CV AUC: {search.best_score_:.4f}")
        return search.best_estimator_

    # XGBoost
    xgb_spaces_optuna = {
        "n_estimators": (IntDistribution(100, 600),),
        "learning_rate": (FloatDistribution(0.005, 0.3, log=True),),
        "max_depth": (IntDistribution(2, 10),),
        "subsample": (FloatDistribution(0.5, 1.0),),
        "colsample_bytree": (FloatDistribution(0.5, 1.0),),
        "min_child_weight": (IntDistribution(1, 10),),
    }
    xgb_clf = XGBClassifier(
        scale_pos_weight=spw, eval_metric="logloss", use_label_encoder=False,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )
    models_xgb = tune_bayes("XGBoost", xgb_clf, xgb_spaces_optuna)

    # MLP
    mlp_spaces_optuna = {
        "hidden_layer_sizes": (CategoricalDistribution(["(32,)", "(64,)", "(64, 32)", "(128, 64)", "(128, 64, 32)"]),),
        "alpha": (FloatDistribution(1e-5, 1e-1, log=True),),
        "learning_rate_init": (FloatDistribution(1e-4, 1e-1, log=True),),
        "activation": (CategoricalDistribution(["relu", "tanh"]),),
    }
    mlp_clf = MLPClassifier(max_iter=500, random_state=RANDOM_STATE)
    if not HAS_SKOPT:
        mlp_pipe = make_pipeline(mlp_clf)
        # optuna categorical must be actual tuples, decoded post-fit
        class _MLPEstimator(mlp_clf.__class__):
            def set_params(self, **params):
                v = params.get("hidden_layer_sizes")
                if isinstance(v, str):
                    params["hidden_layer_sizes"] = tuple(int(x) for x in v.replace("(", "").replace(")", "").replace(" ", "").split(",") if x)
                return super().set_params(**params)
        mlp_pipe.set_params(model=_MLPEstimator(max_iter=500, random_state=RANDOM_STATE))
        pipe = mlp_pipe
        optuna_sp = {f"model__{k2}": dist[0] for k2, dist in mlp_spaces_optuna.items()}
        search = OptunaSearchCV(pipe, optuna_sp, cv=tune_cv, scoring="roc_auc",
                                n_trials=40, timeout=3600, n_jobs=-1,
                                random_state=RANDOM_STATE, verbose=0)
        search.fit(X_train, y_train)
        print(f"MLP best hyperparameters (Bayesian): {search.best_params_}")
        print(f"    tuning CV AUC: {search.best_score_:.4f}")
        models_mlp = search.best_estimator_
    else:
        models_mlp = tune_bayes("MLP", mlp_clf, mlp_spaces_skopt)

    # Run GridSearchCV for the four grid models
    tuned = {"XGBoost": models_xgb, "MLP": models_mlp}
    for name, (pipe, params) in grid_searches.items():
        gs = GridSearchCV(pipe, params, cv=tune_cv, scoring="roc_auc", n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)
        print(f"{name} best hyperparameters (Grid): {gs.best_params_}")
        print(f"    tuning CV AUC: {gs.best_score_:.4f}")
        tuned[name] = gs.best_estimator_
    models = tuned

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
