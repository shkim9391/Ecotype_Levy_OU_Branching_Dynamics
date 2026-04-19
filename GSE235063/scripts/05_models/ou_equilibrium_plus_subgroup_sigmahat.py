from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="One or more of the test scores are non-finite")

ROOT = Path("/GSE235063/derived_dx_primary_training")
df = pd.read_csv(ROOT / "dx_ou_ilr_branch_ready.csv")

responses = [
    "ilr_stem_vs_committed",
    "ilr_prog_vs_mature",
    "ilr_gmp_vs_monodc",
    "log_aux_erybaso",
    "log_aux_clp",
]

alphas = np.logspace(-3, 4, 71)

# fixed subgroup order, KMT2A as reference
subgroup_levels = ["KMT2A", "RUNX", "CBFB", "FLT", "Other"]
df["Subgroup"] = pd.Categorical(df["Subgroup"], categories=subgroup_levels)

print("\n=== SUBGROUP COUNTS: full dataset ===")
print(df["Subgroup"].value_counts(dropna=False).to_string())

def make_preprocess():
    return ColumnTransformer(
        transformers=[
            ("scale_pc", StandardScaler(), ["PC1", "PC2"]),
            ("keep_bin", "passthrough", ["is_blood"]),
            ("subgroup", OneHotEncoder(
                categories=[subgroup_levels],
                drop="first",
                handle_unknown="ignore"
            ), ["Subgroup"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

def fit_suite(df_sub: pd.DataFrame, tag: str):
    df_sub = df_sub.copy().reset_index(drop=True)
    df_sub["Subgroup"] = pd.Categorical(df_sub["Subgroup"], categories=subgroup_levels)

    X = df_sub[["PC1", "PC2", "is_blood", "Subgroup"]].copy()
    loo = LeaveOneOut()

    perf_rows = []
    coef_rows = []
    pred_df = df_sub[["sample_id", "sample", "Biopsy_Origin", "Subgroup"]].copy()
    fitted_df = df_sub[["sample_id", "sample"]].copy()

    print(f"\n=== SUBGROUP COUNTS: {tag} ===")
    print(df_sub["Subgroup"].value_counts(dropna=False).to_string())

    for ycol in responses:
        y = df_sub[ycol].to_numpy()

        y_pred_loo = np.zeros_like(y, dtype=float)
        alpha_used = []

        for train_idx, test_idx in loo.split(X):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y[train_idx]

            pipe = Pipeline([
                ("preprocess", make_preprocess()),
                ("model", RidgeCV(
                    alphas=alphas,
                    cv=LeaveOneOut(),
                    scoring="neg_mean_squared_error"
                ))
            ])

            pipe.fit(X_train, y_train)
            y_pred_loo[test_idx[0]] = pipe.predict(X_test)[0]
            alpha_used.append(float(pipe.named_steps["model"].alpha_))

        rmse = float(np.sqrt(mean_squared_error(y, y_pred_loo)))
        r2 = float(r2_score(y, y_pred_loo))
        corr = float(np.corrcoef(y, y_pred_loo)[0, 1]) if np.std(y_pred_loo) > 0 else np.nan

        perf_rows.append({
            "analysis": tag,
            "response": ycol,
            "n_samples": len(df_sub),
            "loo_rmse": rmse,
            "loo_r2": r2,
            "loo_corr": corr,
            "median_alpha_across_outer_folds": float(np.median(alpha_used)),
        })

        pred_df[f"true__{ycol}"] = y
        pred_df[f"predloo__{ycol}"] = y_pred_loo
        pred_df[f"residloo__{ycol}"] = y - y_pred_loo

        pipe_full = Pipeline([
            ("preprocess", make_preprocess()),
            ("model", RidgeCV(
                alphas=alphas,
                cv=LeaveOneOut(),
                scoring="neg_mean_squared_error"
            ))
        ])

        pipe_full.fit(X, y)
        y_fitted = pipe_full.predict(X)

        fitted_df[f"fitted__{ycol}"] = y_fitted
        fitted_df[f"resid__{ycol}"] = y - y_fitted

        feature_names = pipe_full.named_steps["preprocess"].get_feature_names_out()
        coef = pipe_full.named_steps["model"].coef_

        coef_rows.append({
            "analysis": tag,
            "response": ycol,
            "term": "intercept",
            "coefficient": float(pipe_full.named_steps["model"].intercept_),
            "alpha_full_fit": float(pipe_full.named_steps["model"].alpha_),
        })

        for name, c in zip(feature_names, coef):
            coef_rows.append({
                "analysis": tag,
                "response": ycol,
                "term": str(name),
                "coefficient": float(c),
                "alpha_full_fit": float(pipe_full.named_steps["model"].alpha_),
            })

    perf_df = pd.DataFrame(perf_rows)
    coef_df = pd.DataFrame(coef_rows)

    resid_cols = [c for c in fitted_df.columns if c.startswith("resid__")]
    sigma = fitted_df[resid_cols].copy()
    sigma.columns = [c.replace("resid__", "") for c in sigma.columns]
    sigma_hat = sigma.cov()

    perf_df.to_csv(ROOT / f"ou_equilibrium_plus_subgroup_performance__{tag}.csv", index=False)
    coef_df.to_csv(ROOT / f"ou_equilibrium_plus_subgroup_coefficients__{tag}.csv", index=False)
    pred_df.to_csv(ROOT / f"ou_equilibrium_plus_subgroup_predictions__{tag}.csv", index=False)
    sigma_hat.to_csv(ROOT / f"ou_equilibrium_plus_subgroup_sigmahat__{tag}.csv")

    print(f"\n=== ANALYSIS: {tag} ===")
    print("\nPerformance:")
    print(perf_df.to_string(index=False))
    print("\nCoefficients:")
    print(coef_df.to_string(index=False))
    print("\nSigma_hat:")
    print(sigma_hat.round(4).to_string())

fit_suite(df, "full19")
fit_suite(df[df["sample_id"] != "AML23_DX"].copy(), "no_AML23")

df_marrow = df[df["Biopsy_Origin"] == "Marrow"].copy()
if len(df_marrow) >= 8:
    fit_suite(df_marrow, "marrow_only")

print("\n=== FILES WRITTEN ===")
for p in sorted(ROOT.glob("ou_equilibrium_plus_subgroup_*")):
    print(p)
