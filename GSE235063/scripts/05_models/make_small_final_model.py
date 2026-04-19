from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
OUTDIR = ROOT / "final_small_model"
OUTDIR.mkdir(exist_ok=True, parents=True)

df = pd.read_csv(ROOT / "dx_ou_ilr_branch_ready.csv")

responses = [
    "ilr_stem_vs_committed",
    "log_aux_erybaso",
]

alphas = np.logspace(-3, 4, 71)

subgroup_levels = ["KMT2A", "RUNX", "CBFB", "FLT", "Other"]
df["Subgroup"] = pd.Categorical(df["Subgroup"], categories=subgroup_levels)

def make_preprocess():
    return ColumnTransformer(
        transformers=[
            ("scale_pc", StandardScaler(), ["PC1", "PC2"]),
            ("keep_bin", "passthrough", ["is_blood"]),
            ("subgroup", OneHotEncoder(
                categories=[subgroup_levels],
                drop="first",
                handle_unknown="ignore",
                sparse_output=False
            ), ["Subgroup"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

def fit_small_model(df_sub: pd.DataFrame, tag: str):
    df_sub = df_sub.copy().reset_index(drop=True)
    df_sub["Subgroup"] = pd.Categorical(df_sub["Subgroup"], categories=subgroup_levels)

    X = df_sub[["PC1", "PC2", "is_blood", "Subgroup"]].copy()
    loo = LeaveOneOut()

    perf_rows = []
    coef_rows = []
    pred_df = df_sub[["sample_id", "sample", "Biopsy_Origin", "Subgroup"]].copy()
    fitted_df = df_sub[["sample_id", "sample"]].copy()

    print(f"\n=== ANALYSIS: {tag} ===")
    print("\nSubgroup counts:")
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

    perf_df.to_csv(OUTDIR / f"small_model_performance__{tag}.csv", index=False)
    coef_df.to_csv(OUTDIR / f"small_model_coefficients__{tag}.csv", index=False)
    pred_df.to_csv(OUTDIR / f"small_model_predictions__{tag}.csv", index=False)
    sigma_hat.to_csv(OUTDIR / f"small_model_sigmahat__{tag}.csv")

    print("\nPerformance:")
    print(perf_df.to_string(index=False))
    print("\nCoefficients:")
    print(coef_df.to_string(index=False))
    print("\nSigma_hat:")
    print(sigma_hat.round(4).to_string())

    # ---------- plots ----------
    for ycol in responses:
        # observed vs predicted
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        x = pred_df[f"true__{ycol}"].to_numpy()
        yhat = pred_df[f"predloo__{ycol}"].to_numpy()
        ax.scatter(x, yhat)
        lo = min(np.min(x), np.min(yhat))
        hi = max(np.max(x), np.max(yhat))
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        for _, r in pred_df.iterrows():
            ax.text(r[f"true__{ycol}"], r[f"predloo__{ycol}"], r["sample"], fontsize=8)
        ax.set_xlabel(f"Observed {ycol}")
        ax.set_ylabel(f"LOO predicted {ycol}")
        ax.set_title(f"{tag}: observed vs predicted")
        fig.tight_layout()
        fig.savefig(OUTDIR / f"small_model_obs_vs_pred__{tag}__{ycol}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # coefficient plot
        cdf = coef_df[(coef_df["response"] == ycol) & (coef_df["term"] != "intercept")].copy()
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.bar(cdf["term"], cdf["coefficient"])
        ax.axhline(0, linestyle="--")
        ax.set_ylabel("Coefficient")
        ax.set_title(f"{tag}: coefficients for {ycol}")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(OUTDIR / f"small_model_coefficients__{tag}__{ycol}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

# primary
fit_small_model(df, "full19")

# required sensitivity
fit_small_model(df[df["sample_id"] != "AML23_DX"].copy(), "no_AML23")

print("\n=== FILES WRITTEN ===")
for p in sorted(OUTDIR.glob("*")):
    print(p)
