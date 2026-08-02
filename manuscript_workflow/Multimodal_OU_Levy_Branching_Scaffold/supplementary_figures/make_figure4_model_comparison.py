from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, t


# ----------------------------- styling ------------------------------------- #

COLOR_STABLE = "#4C9ED9"
COLOR_SWITCHING = "#F28E2B"
COLOR_GAUSSIAN = "#6E6E6E"
COLOR_HEAVY = "#B22222"
COLOR_BEST = "#B22222"
COLOR_OTHER = "#9A9A9A"
COLOR_REF = "#7A7A7A"

PANEL_FONT_SIZE = 18
TITLE_FONT_SIZE = 15
LABEL_FONT_SIZE = 12
TICK_FONT_SIZE = 11
ANNOT_FONT_SIZE = 10

DEFAULT_FIGSIZE = (16, 10)
DEFAULT_DPI = 600
EPS = 1e-8


# ----------------------------- data model ---------------------------------- #

@dataclass
class FitResult:
    model_id: str
    model_label: str
    family: str
    branch_aware: bool
    k: int
    n: int
    success: bool
    message: str
    loglik: float
    aic: float
    aicc: float
    bic: float
    params: Dict[str, float]
    logpdf: np.ndarray


# ----------------------------- utilities ----------------------------------- #

def standardize_stability(value: object) -> Optional[str]:
    if pd.isna(value):
        return None

    s = str(value).strip().lower()
    if s in {"", "nan", "none", "null"}:
        return None

    if "stable" in s or "same" in s or "no_switch" in s:
        return "Stable"
    if "switch" in s or "chang" in s or "different" in s:
        return "Switching"

    if s in {"0", "false", "no"}:
        return "Stable"
    if s in {"1", "true", "yes"}:
        return "Switching"

    try:
        x = float(s)
        return "Switching" if x != 0 else "Stable"
    except ValueError:
        return None


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, sep=None, engine="python")
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def aicc_from_loglik(loglik: float, k: int, n: int) -> float:
    aic = 2 * k - 2 * loglik
    if n <= k + 1:
        return np.inf
    return aic + (2 * k * (k + 1)) / (n - k - 1)


def compute_information_criteria(loglik: float, k: int, n: int) -> Tuple[float, float, float]:
    aic = 2 * k - 2 * loglik
    aicc = aicc_from_loglik(loglik, k, n)
    bic = math.log(n) * k - 2 * loglik
    return aic, aicc, bic


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12, 1.05, label,
        transform=ax.transAxes,
        fontsize=PANEL_FONT_SIZE,
        fontweight="bold",
        va="top",
        ha="left",
    )


def safe_std(x: np.ndarray) -> float:
    s = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    return max(s, 1e-4)


# ----------------------------- likelihoods --------------------------------- #

def gaussian_pooled_logpdf(theta: np.ndarray, x: np.ndarray, g: np.ndarray) -> np.ndarray:
    mu, log_sigma = theta
    sigma = np.exp(log_sigma)
    return norm.logpdf(x, loc=mu, scale=sigma)


def gaussian_branch_logpdf(theta: np.ndarray, x: np.ndarray, g: np.ndarray) -> np.ndarray:
    mu_stable, mu_switch, log_sigma_stable, log_sigma_switch = theta
    sigma = np.where(g == 0, np.exp(log_sigma_stable), np.exp(log_sigma_switch))
    mu = np.where(g == 0, mu_stable, mu_switch)
    return norm.logpdf(x, loc=mu, scale=sigma)


def student_pooled_logpdf(theta: np.ndarray, x: np.ndarray, g: np.ndarray) -> np.ndarray:
    mu, log_sigma, log_nu_minus_2 = theta
    sigma = np.exp(log_sigma)
    nu = 2.0 + np.exp(log_nu_minus_2)
    return t.logpdf(x, df=nu, loc=mu, scale=sigma)


def student_branch_logpdf(theta: np.ndarray, x: np.ndarray, g: np.ndarray) -> np.ndarray:
    mu_stable, mu_switch, log_sigma_stable, log_sigma_switch, log_nu_minus_2 = theta
    sigma = np.where(g == 0, np.exp(log_sigma_stable), np.exp(log_sigma_switch))
    nu = 2.0 + np.exp(log_nu_minus_2)
    mu = np.where(g == 0, mu_stable, mu_switch)
    return t.logpdf(x, df=nu, loc=mu, scale=sigma)


def nll_from_logpdf_fn(logpdf_fn, theta: np.ndarray, x: np.ndarray, g: np.ndarray) -> float:
    logpdf = logpdf_fn(theta, x, g)
    if np.any(~np.isfinite(logpdf)):
        return 1e12
    return -float(np.sum(logpdf))


def optimize_model(logpdf_fn, theta0: np.ndarray, x: np.ndarray, g: np.ndarray):
    def objective(theta):
        return nll_from_logpdf_fn(logpdf_fn, theta, x, g)

    result = minimize(objective, theta0, method="L-BFGS-B")
    if result.success:
        return result

    # fallback
    result2 = minimize(objective, theta0, method="Powell")
    return result2


def fit_model(
    model_id: str,
    model_label: str,
    family: str,
    branch_aware: bool,
    x: np.ndarray,
    g: np.ndarray,
) -> FitResult:
    x = np.asarray(x, dtype=float)
    g = np.asarray(g, dtype=int)
    n = len(x)

    stable = x[g == 0]
    switch = x[g == 1]

    if family == "gaussian" and not branch_aware:
        mu0 = float(np.mean(x))
        sigma0 = safe_std(x)
        theta0 = np.array([mu0, np.log(sigma0)])
        logpdf_fn = gaussian_pooled_logpdf
        k = 2

    elif family == "gaussian" and branch_aware:
        mu_stable0 = float(np.mean(stable))
        mu_switch0 = float(np.mean(switch))
        sigma_stable0 = safe_std(stable)
        sigma_switch0 = safe_std(switch)
        theta0 = np.array([
            mu_stable0,
            mu_switch0,
            np.log(sigma_stable0),
            np.log(sigma_switch0),
        ])
        logpdf_fn = gaussian_branch_logpdf
        k = 4

    elif family == "student_t" and not branch_aware:
        mu0 = float(np.median(x))
        sigma0 = safe_std(x)
        nu0 = 8.0
        theta0 = np.array([mu0, np.log(sigma0), np.log(nu0 - 2.0)])
        logpdf_fn = student_pooled_logpdf
        k = 3

    elif family == "student_t" and branch_aware:
        mu_stable0 = float(np.median(stable))
        mu_switch0 = float(np.median(switch))
        sigma_stable0 = safe_std(stable)
        sigma_switch0 = safe_std(switch)
        nu0 = 8.0
        theta0 = np.array([
            mu_stable0,
            mu_switch0,
            np.log(sigma_stable0),
            np.log(sigma_switch0),
            np.log(nu0 - 2.0),
        ])
        logpdf_fn = student_branch_logpdf
        k = 5

    else:
        raise ValueError("Unsupported model specification.")

    opt = optimize_model(logpdf_fn, theta0, x, g)
    theta_hat = opt.x
    logpdf = logpdf_fn(theta_hat, x, g)
    loglik = float(np.sum(logpdf))
    aic, aicc, bic = compute_information_criteria(loglik, k, n)

    if family == "gaussian" and not branch_aware:
        params = {
            "mu": float(theta_hat[0]),
            "sigma": float(np.exp(theta_hat[1])),
        }

    elif family == "gaussian" and branch_aware:
        params = {
            "mu_stable": float(theta_hat[0]),
            "mu_switching": float(theta_hat[1]),
            "sigma_stable": float(np.exp(theta_hat[2])),
            "sigma_switching": float(np.exp(theta_hat[3])),
        }

    elif family == "student_t" and not branch_aware:
        params = {
            "mu": float(theta_hat[0]),
            "sigma": float(np.exp(theta_hat[1])),
            "nu": float(2.0 + np.exp(theta_hat[2])),
        }

    else:
        params = {
            "mu_stable": float(theta_hat[0]),
            "mu_switching": float(theta_hat[1]),
            "sigma_stable": float(np.exp(theta_hat[2])),
            "sigma_switching": float(np.exp(theta_hat[3])),
            "nu": float(2.0 + np.exp(theta_hat[4])),
        }

    return FitResult(
        model_id=model_id,
        model_label=model_label,
        family=family,
        branch_aware=branch_aware,
        k=k,
        n=n,
        success=bool(opt.success),
        message=str(opt.message),
        loglik=loglik,
        aic=aic,
        aicc=aicc,
        bic=bic,
        params=params,
        logpdf=logpdf,
    )


# ----------------------------- preparation --------------------------------- #

def prepare_data(
    df: pd.DataFrame,
    metric_column: str,
    sample_column: str,
    dx_branch_column: str,
    rel_branch_column: str,
    stability_column: Optional[str],
) -> pd.DataFrame:
    required = [metric_column, sample_column, dx_branch_column, rel_branch_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = df.copy()
    out["sample_std"] = out[sample_column].astype(str).str.strip()
    out["dx_branch_std"] = out[dx_branch_column].astype(str).str.strip()
    out["rel_branch_std"] = out[rel_branch_column].astype(str).str.strip()
    out["disp_std"] = pd.to_numeric(out[metric_column], errors="coerce")
    out["transition_std"] = out["dx_branch_std"] + "→" + out["rel_branch_std"]

    if stability_column and stability_column in out.columns:
        out["stability_std"] = out[stability_column].map(standardize_stability)
    else:
        out["stability_std"] = None

    fallback = np.where(
        out["dx_branch_std"] == out["rel_branch_std"],
        "Stable",
        "Switching",
    )
    out["stability_std"] = out["stability_std"].fillna(pd.Series(fallback, index=out.index))
    out["stability_std"] = out["stability_std"].astype(str).str.strip().str.title()

    out = out.dropna(subset=["sample_std", "disp_std", "dx_branch_std", "rel_branch_std", "stability_std"]).copy()
    out = out[out["stability_std"].isin(["Stable", "Switching"])].copy()

    if out.empty:
        raise ValueError("No valid rows remain after preprocessing.")

    n_stable = int((out["stability_std"] == "Stable").sum())
    n_switch = int((out["stability_std"] == "Switching").sum())
    if n_stable < 2 or n_switch < 2:
        raise ValueError(f"Need at least 2 Stable and 2 Switching rows. Found Stable={n_stable}, Switching={n_switch}.")

    out["group_code"] = np.where(out["stability_std"] == "Stable", 0, 1)
    return out


# ----------------------------- summaries ----------------------------------- #

def results_to_tables(results: Dict[str, FitResult]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    comp_rows = []
    param_rows = []

    for model_id, fit in results.items():
        comp_rows.append({
            "model_id": model_id,
            "model_label": fit.model_label,
            "family": fit.family,
            "branch_aware": fit.branch_aware,
            "k": fit.k,
            "n": fit.n,
            "success": fit.success,
            "message": fit.message,
            "loglik": fit.loglik,
            "aic": fit.aic,
            "aicc": fit.aicc,
            "bic": fit.bic,
        })
        for pname, pval in fit.params.items():
            param_rows.append({
                "model_id": model_id,
                "model_label": fit.model_label,
                "parameter": pname,
                "value": pval,
            })

    comp_df = pd.DataFrame(comp_rows).sort_values("aicc", ascending=True).reset_index(drop=True)
    best_aicc = float(comp_df["aicc"].min())
    comp_df["delta_aicc"] = comp_df["aicc"] - best_aicc

    weights = np.exp(-0.5 * comp_df["delta_aicc"].to_numpy(dtype=float))
    weights = weights / np.sum(weights)
    comp_df["aicc_weight"] = weights

    param_df = pd.DataFrame(param_rows)
    return comp_df, param_df


def best_model_by_family(comp_df: pd.DataFrame, family: str) -> str:
    sub = comp_df[comp_df["family"] == family].sort_values("aicc", ascending=True)
    if sub.empty:
        raise ValueError(f"No models found for family={family}")
    return str(sub.iloc[0]["model_id"])


def fit_survival_curve(
    fit: FitResult,
    x_grid: np.ndarray,
    p_stable: float,
    p_switch: float,
) -> np.ndarray:
    if fit.family == "gaussian" and not fit.branch_aware:
        mu = fit.params["mu"]
        sigma = fit.params["sigma"]
        return norm.sf(x_grid, loc=mu, scale=sigma)

    if fit.family == "gaussian" and fit.branch_aware:
        mu0 = fit.params["mu_stable"]
        mu1 = fit.params["mu_switching"]
        sigma0 = fit.params["sigma_stable"]
        sigma1 = fit.params["sigma_switching"]
        return (
            p_stable * norm.sf(x_grid, loc=mu0, scale=sigma0) +
            p_switch * norm.sf(x_grid, loc=mu1, scale=sigma1)
        )

    if fit.family == "student_t" and not fit.branch_aware:
        mu = fit.params["mu"]
        sigma = fit.params["sigma"]
        nu = fit.params["nu"]
        return t.sf(x_grid, df=nu, loc=mu, scale=sigma)

    if fit.family == "student_t" and fit.branch_aware:
        mu0 = fit.params["mu_stable"]
        mu1 = fit.params["mu_switching"]
        sigma0 = fit.params["sigma_stable"]
        sigma1 = fit.params["sigma_switching"]
        nu = fit.params["nu"]
        return (
            p_stable * t.sf(x_grid, df=nu, loc=mu0, scale=sigma0) +
            p_switch * t.sf(x_grid, df=nu, loc=mu1, scale=sigma1)
        )

    raise ValueError("Unsupported fit for survival curve.")


def empirical_survival(x: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.array([(x >= v).mean() for v in x_grid], dtype=float)


# ----------------------------- plotting ------------------------------------ #

def draw_model_box(ax: plt.Axes, xy: Tuple[float, float], w: float, h: float, title: str, body: str) -> None:
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.3,
        edgecolor=COLOR_REF,
        facecolor="white",
    )
    ax.add_patch(box)
    ax.text(xy[0] + 0.03, xy[1] + h - 0.08, title, fontsize=12, fontweight="bold", ha="left", va="top")
    ax.text(xy[0] + 0.03, xy[1] + h - 0.18, body, fontsize=11, ha="left", va="top")


def plot_panel_a_model_ladder(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_title("Nested Gaussian and heavy-tail model ladder", fontsize=TITLE_FONT_SIZE)

    draw_model_box(
        ax, (0.03, 0.56), 0.42, 0.32,
        "M0  Gaussian pooled",
        r"$\Delta_i \sim \mathcal{N}(\mu,\sigma^2)$",
    )
    draw_model_box(
        ax, (0.53, 0.56), 0.42, 0.32,
        "M1  Gaussian branch-aware",
        r"$\Delta_i \sim \mathcal{N}(\mu_g,\sigma_g^2)$" + "\n" + r"$g \in \{\mathrm{Stable, Switching}\}$",
    )
    draw_model_box(
        ax, (0.03, 0.12), 0.42, 0.32,
        "M2  Student-t pooled",
        r"$\Delta_i \sim t_{\nu}(\mu,\sigma)$",
    )
    draw_model_box(
        ax, (0.53, 0.12), 0.42, 0.32,
        "M3  Student-t branch-aware",
        r"$\Delta_i \sim t_{\nu}(\mu_g,\sigma_g)$" + "\n" + r"$g \in \{\mathrm{Stable, Switching}\}$",
    )

    ax.text(0.49, 0.735, "add branch structure", fontsize=11, ha="center", va="center", color=COLOR_REF)
    ax.text(0.49, 0.295, "add branch structure", fontsize=11, ha="center", va="center", color=COLOR_REF)
    ax.text(0.24, 0.49, "add heavy tails", fontsize=11, ha="center", va="center", color=COLOR_REF)
    ax.text(0.75, 0.49, "add heavy tails", fontsize=11, ha="center", va="center", color=COLOR_REF)

    ax.annotate("", xy=(0.53, 0.72), xytext=(0.45, 0.72),
                arrowprops=dict(arrowstyle="->", color=COLOR_REF, lw=1.3))
    ax.annotate("", xy=(0.53, 0.28), xytext=(0.45, 0.28),
                arrowprops=dict(arrowstyle="->", color=COLOR_REF, lw=1.3))
    ax.annotate("", xy=(0.24, 0.44), xytext=(0.24, 0.56),
                arrowprops=dict(arrowstyle="->", color=COLOR_REF, lw=1.3))
    ax.annotate("", xy=(0.75, 0.44), xytext=(0.75, 0.56),
                arrowprops=dict(arrowstyle="->", color=COLOR_REF, lw=1.3))


def plot_panel_b_model_comparison(ax: plt.Axes, comp_df: pd.DataFrame) -> None:
    plot_df = comp_df.sort_values("delta_aicc", ascending=True).reset_index(drop=True)

    y = np.arange(len(plot_df))[::-1]
    colors = [COLOR_BEST if d == 0 else COLOR_OTHER for d in plot_df["delta_aicc"]]

    ax.barh(y, plot_df["delta_aicc"], color=colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["model_label"], fontsize=TICK_FONT_SIZE)
    ax.set_xlabel(r"$\Delta$AICc  (lower is better)", fontsize=LABEL_FONT_SIZE)
    ax.set_title("Branch-aware heavy-tail model provides the best predictive fit", fontsize=TITLE_FONT_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)

    for yi, (_, row) in zip(y, plot_df.iterrows()):
        ax.text(
            row["delta_aicc"] + 0.08,
            yi,
            f"AICc={row['aicc']:.2f}",
            fontsize=10,
            ha="left",
            va="center",
        )

    ax.axvline(0, color=COLOR_REF, linestyle="--", linewidth=1.2)
    ax.set_xlim(0, float(plot_df["delta_aicc"].max()) + 1.3)


def plot_panel_c_tail_fit(
    ax: plt.Axes,
    df: pd.DataFrame,
    best_gaussian: FitResult,
    best_student_t: FitResult,
) -> pd.DataFrame:
    switch_df = df[df["stability_std"] == "Switching"].copy()
    x = switch_df["disp_std"].to_numpy(dtype=float)

    x_min = max(0.0, float(np.min(x)) * 0.95)
    x_max = float(np.max(x)) * 1.15
    x_grid = np.linspace(x_min, x_max, 300)

    obs_surv = empirical_survival(x, x_grid)

    floor = 0.5 / len(x)

    # Switching-group fitted survival
    if best_gaussian.branch_aware:
        gauss_surv = norm.sf(
            x_grid,
            loc=best_gaussian.params["mu_switching"],
            scale=best_gaussian.params["sigma_switching"],
        )
    else:
        gauss_surv = fit_survival_curve(best_gaussian, x_grid, p_stable=0.0, p_switch=1.0)

    if best_student_t.branch_aware:
        student_surv = t.sf(
            x_grid,
            df=best_student_t.params["nu"],
            loc=best_student_t.params["mu_switching"],
            scale=best_student_t.params["sigma_switching"],
        )
    else:
        student_surv = fit_survival_curve(best_student_t, x_grid, p_stable=0.0, p_switch=1.0)

    ax.plot(
        x_grid, np.maximum(obs_surv, floor),
        color="black", linewidth=2.2, label="Observed Switching survival"
    )
    ax.plot(
        x_grid, np.maximum(gauss_surv, floor),
        color=COLOR_GAUSSIAN, linewidth=2.0, linestyle="--",
        label=f"Switching Gaussian fit: {best_gaussian.model_label}"
    )
    ax.plot(
        x_grid, np.maximum(student_surv, floor),
        color=COLOR_HEAVY, linewidth=2.0,
        label=f"Switching heavy-tail fit: {best_student_t.model_label}"
    )

    ax.set_yscale("log")
    ax.set_xlabel("DX→REL total displacement (6D)", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(r"Switching survival  $P(\Delta \geq x)$", fontsize=LABEL_FONT_SIZE)
    ax.set_title("Switching-group tail fit highlights the heavy-tail comparison", fontsize=TITLE_FONT_SIZE)
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    tail_df = pd.DataFrame({
        "x": x_grid,
        "observed_switching_survival": obs_surv,
        "switching_gaussian_survival": gauss_surv,
        "switching_student_t_survival": student_surv,
        "best_gaussian_model": best_gaussian.model_id,
        "best_student_t_model": best_student_t.model_id,
    })
    return tail_df


def plot_panel_d_casewise_gain(
    ax: plt.Axes,
    case_df: pd.DataFrame,
    top_annotate: int,
) -> None:
    ranked = case_df.sort_values("delta_loglik", ascending=False).reset_index(drop=True)
    y = np.arange(len(ranked))[::-1]

    colors = ranked["stability_std"].map({"Stable": COLOR_STABLE, "Switching": COLOR_SWITCHING}).tolist()

    for yi, delta, color in zip(y, ranked["delta_loglik"], colors):
        ax.hlines(yi, 0.0, delta, color=COLOR_REF, linewidth=1.4, zorder=1)
        ax.scatter(delta, yi, s=70, color=color, edgecolor="white", linewidth=0.7, zorder=3)

    ax.axvline(0, color=COLOR_REF, linestyle="--", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(ranked["sample_std"], fontsize=TICK_FONT_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax.set_xlabel(
        r"Case-wise log-likelihood gain  "
        r"$\log p(\Delta_i \mid M3)-\log p(\Delta_i \mid M1)$",
        fontsize=LABEL_FONT_SIZE,
    )
    ax.set_title("Model improvement is concentrated in extreme displacement cases", fontsize=TITLE_FONT_SIZE)

    top = ranked.head(min(top_annotate, len(ranked)))
    for _, row in top.iterrows():
        ypos = int(ranked.index[ranked["sample_std"] == row["sample_std"]][0])
        yi = len(ranked) - 1 - ypos
        ax.annotate(
            row["sample_std"],
            xy=(row["delta_loglik"], yi),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=ANNOT_FONT_SIZE,
            ha="left",
            va="bottom",
        )

    legend_elems = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_STABLE,
                   markeredgecolor="white", markersize=8, label="Branch-continuous"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_SWITCHING,
                   markeredgecolor="white", markersize=8, label="Branch-switching"),
    ]
    ax.legend(handles=legend_elems, frameon=False, fontsize=10, loc="lower right")


# ----------------------------- main build ---------------------------------- #

def build_figure_and_outputs(
    df: pd.DataFrame,
    output_prefix: Path,
    top_annotate: int,
    dpi: int,
    figsize: Tuple[float, float],
) -> None:
    x = df["disp_std"].to_numpy(dtype=float)
    g = df["group_code"].to_numpy(dtype=int)

    results = {
        "M0": fit_model("M0", "Gaussian pooled", "gaussian", False, x, g),
        "M1": fit_model("M1", "Gaussian branch-aware", "gaussian", True, x, g),
        "M2": fit_model("M2", "Student-t pooled", "student_t", False, x, g),
        "M3": fit_model("M3", "Student-t branch-aware", "student_t", True, x, g),
    }

    comp_df, param_df = results_to_tables(results)

    best_gaussian_id = best_model_by_family(comp_df, "gaussian")
    best_student_id = best_model_by_family(comp_df, "student_t")

    best_gaussian = results[best_gaussian_id]
    best_student_t = results[best_student_id]

    # Case-wise gain uses branch-aware heavy-tail vs branch-aware Gaussian.
    m1 = results["M1"]
    m3 = results["M3"]
    case_df = df[["sample_std", "stability_std", "dx_branch_std", "rel_branch_std", "transition_std", "disp_std"]].copy()
    case_df["loglik_m1_gaussian_branch"] = m1.logpdf
    case_df["loglik_m3_student_t_branch"] = m3.logpdf
    case_df["delta_loglik"] = case_df["loglik_m3_student_t_branch"] - case_df["loglik_m1_gaussian_branch"]
    case_df = case_df.sort_values("delta_loglik", ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    plot_panel_a_model_ladder(ax_a)
    plot_panel_b_model_comparison(ax_b, comp_df)
    tail_df = plot_panel_c_tail_fit(ax_c, df, best_gaussian, best_student_t)
    plot_panel_d_casewise_gain(ax_d, case_df, top_annotate=top_annotate)

    for label, ax in zip(["E", "F", "G", "H"], [ax_a, ax_b, ax_c, ax_d]):
        panel_label(ax, label)

    fig.tight_layout()

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    comp_path = output_prefix.parent / f"{output_prefix.name}_model_comparison.csv"
    param_path = output_prefix.parent / f"{output_prefix.name}_parameter_summary.csv"
    tail_path = output_prefix.parent / f"{output_prefix.name}_tail_fit_grid.csv"
    case_path = output_prefix.parent / f"{output_prefix.name}_casewise_loglik_gain.csv"
    summary_path = output_prefix.parent / f"{output_prefix.name}_fit_summary.txt"

    comp_df.to_csv(comp_path, index=False)
    param_df.to_csv(param_path, index=False)
    tail_df.to_csv(tail_path, index=False)
    case_df.to_csv(case_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("Supplementary Figure S6E-H model comparison summary\n")
        handle.write("=" * 40 + "\n\n")
        handle.write(f"Best Gaussian model: {best_gaussian.model_label} ({best_gaussian.model_id})\n")
        handle.write(f"Best heavy-tail model: {best_student_t.model_label} ({best_student_t.model_id})\n\n")
        handle.write("Model ranking by AICc:\n")
        for _, row in comp_df.iterrows():
            handle.write(
                f"  {row['model_id']} | {row['model_label']} | "
                f"AICc={row['aicc']:.4f} | ΔAICc={row['delta_aicc']:.4f} | "
                f"weight={row['aicc_weight']:.4f}\n"
            )
        handle.write("\nParameters:\n")
        for _, row in param_df.iterrows():
            handle.write(f"  {row['model_id']} | {row['parameter']} = {row['value']:.6f}\n")

    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")
    print(f"saved: {comp_path}")
    print(f"saved: {param_path}")
    print(f"saved: {tail_path}")
    print(f"saved: {case_path}")
    print(f"saved: {summary_path}")


# ----------------------------- cli ----------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Figure 4 model comparison using Gaussian and Student-t families.")
    parser.add_argument("--input", required=True, help="Path to per-sample displacement table.")
    parser.add_argument("--output-prefix", required=True, help="Output prefix without extension.")
    parser.add_argument("--metric-column", default="disp_total_6d", help="Displacement metric to model.")
    parser.add_argument("--sample-column", default="sample", help="Sample ID column.")
    parser.add_argument("--stability-column", default="dx_to_rel_switch", help="Stable/switching column.")
    parser.add_argument("--dx-branch-column", default="DX_branch_ge50", help="Diagnosis branch column.")
    parser.add_argument("--rel-branch-column", default="REL_branch_ge50", help="Relapse branch column.")
    parser.add_argument("--top-annotate", type=int, default=5, help="Top samples to annotate in panel D.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--width", type=float, default=DEFAULT_FIGSIZE[0])
    parser.add_argument("--height", type=float, default=DEFAULT_FIGSIZE[1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    df_raw = load_table(input_path)
    df = prepare_data(
        df_raw,
        metric_column=args.metric_column,
        sample_column=args.sample_column,
        dx_branch_column=args.dx_branch_column,
        rel_branch_column=args.rel_branch_column,
        stability_column=args.stability_column,
    )

    print("Rows retained:", len(df))
    print("Stability counts:", df["stability_std"].value_counts().to_dict())
    print("Metric column:", args.metric_column)

    build_figure_and_outputs(
        df=df,
        output_prefix=output_prefix,
        top_annotate=args.top_annotate,
        dpi=args.dpi,
        figsize=(args.width, args.height),
    )


if __name__ == "__main__":
    main()
