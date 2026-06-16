from __future__ import annotations

import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from scipy import sparse
from scipy.interpolate import interp1d, splrep, splev
from scipy.signal import find_peaks, savgol_filter
from scipy.sparse.linalg import spsolve
from scipy.stats import wasserstein_distance
from sklearn.decomposition import NMF
from sklearn.metrics import auc, roc_auc_score, roc_curve


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
MILK_DIR = SRC_DIR / "Milk_AI"
OUT_ROOT = REPO_ROOT / "validation_output"
CORRECTED_DIR = OUT_ROOT / "corrected_pipeline"
NOTEBOOK3_DIR = OUT_ROOT / "notebook3"
DOWNLOADS = Path(os.environ.get("RAW_SAMPLE_DIR", r"C:\Users\Amay Kashyap Deka\Downloads"))

CONTROL_XLSX = SRC_DIR / "combined_raman_spectra.xlsx"
EXPERIMENTAL_XLSX = SRC_DIR / "combined_raman_spectra (Experimental).xlsx"

COMMON_SHIFT = np.arange(400.0, 3500.0 + 0.625, 1.25)
FINGERPRINT_BOUNDS = (400.0, 1800.0)
CH_STRETCH_BOUNDS = (2800.0, 3000.0)
THRESHOLD_PERCENTILE = 95
RANDOM_STATE = 42


LITERATURE_TABLE = pd.DataFrame(
    [
        (1016, "Aromatic C-C", "Fat", "milk"),
        (1279, "CH2 twist / ester", "Fat", "milk"),
        (1315, "CH2 twist / ester", "Fat", "milk"),
        (1416, "CH2 bending", "Fat / Lactose", "milk"),
        (1759, "CH2 bending", "Fat / Lactose", "milk"),
        (1566, "Amide I", "Proteins", "milk"),
        (1670, "Amide II", "Proteins", "milk"),
        (2865, "CH stretch", "Fatty acids", "milk"),
        (2902, "CH stretch", "Fatty acids", "milk"),
        (1160, "Carotenoids", "Bacterial-action signature", "bacterial-action"),
        (700, "Nucleic-acid region", "Bacterial-action signature", "bacterial-action"),
        (800, "Nucleic-acid region", "Bacterial-action signature", "bacterial-action"),
        (1450, "Acetyl-CoA / beta-oxidation", "Bacterial-action signature", "bacterial-action"),
        (724, "Acetyl-CoA / beta-oxidation", "Bacterial-action signature", "bacterial-action"),
    ],
    columns=["band_cm-1", "functional_group", "assignment", "source_type"],
)


@dataclass
class Spectrum:
    name: str
    group: str
    x: np.ndarray
    y: np.ndarray
    source: str


def ensure_dirs() -> None:
    CORRECTED_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK3_DIR.mkdir(parents=True, exist_ok=True)


def als_baseline(y: np.ndarray, lam: float = 1e6, p: float = 0.01, n_iter: int = 10) -> np.ndarray:
    """Asymmetric least-squares baseline used by the original preprocessing notebooks."""
    y = np.asarray(y, dtype=float)
    length = y.size
    difference = sparse.diags(
        [np.ones(length), -2.0 * np.ones(length), np.ones(length)],
        [0, -1, -2],
        shape=(length, length - 2),
        format="csc",
    )
    weights = np.ones(length)
    for _ in range(n_iter):
        weight_matrix = sparse.spdiags(weights, 0, length, length)
        system = weight_matrix + lam * difference.dot(difference.T)
        baseline = spsolve(system, weights * y)
        weights = p * (y > baseline) + (1.0 - p) * (y < baseline)
    return baseline


def preprocess_native(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Savitzky-Golay -> ALS baseline correction -> negative clipping -> L2 normalization."""
    y = np.asarray(y, dtype=float)
    smoothed = savgol_filter(y, window_length=5, polyorder=2)
    baseline = als_baseline(smoothed, lam=1e6, p=0.01, n_iter=10)
    corrected = smoothed - baseline
    clipped = np.clip(corrected, 0.0, None)
    norm = np.linalg.norm(clipped)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Preprocessing produced a zero/non-finite spectrum.")
    normalized = clipped / norm
    return {
        "raw": y,
        "smoothed": smoothed,
        "baseline": baseline,
        "baseline_corrected": clipped,
        "normalized": normalized,
    }


def fit_functional_spline(x: np.ndarray, y: np.ndarray):
    """Fit a common-order B-spline functional representation.

    s=1e-4 is applied to every normalized spectrum. This is deliberately small:
    the prior Savitzky-Golay + ALS pipeline already removes most high-frequency
    acquisition noise, so the spline primarily provides a continuous FDA object
    and consistent evaluation on a common Raman grid rather than aggressive
    smoothing.
    """
    return splrep(x, y, k=3, s=1e-4)


def spline_eval(tck, grid: np.ndarray = COMMON_SHIFT) -> np.ndarray:
    return np.clip(np.asarray(splev(grid, tck), dtype=float), 0.0, None)


def linear_eval(x: np.ndarray, y: np.ndarray, grid: np.ndarray = COMMON_SHIFT) -> np.ndarray:
    f = interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate", assume_sorted=True)
    return np.clip(f(grid), 0.0, None)


def geometric_median(points: np.ndarray, tolerance: float = 1e-6, max_iterations: int = 500) -> np.ndarray:
    estimate = np.mean(points, axis=0)
    for _ in range(max_iterations):
        distances = np.linalg.norm(points - estimate, axis=1)
        if np.any(distances < 1e-12):
            return points[np.argmin(distances)].copy()
        inv = 1.0 / distances
        updated = np.average(points, axis=0, weights=inv)
        if np.linalg.norm(updated - estimate) < tolerance:
            return updated
        estimate = updated
    return estimate


def exact_quantile_barycenter(spectra: np.ndarray, support: np.ndarray = COMMON_SHIFT, n_quantiles: int = 10001) -> np.ndarray:
    nonnegative = np.clip(spectra, 0.0, None)
    probabilities = nonnegative / nonnegative.sum(axis=1, keepdims=True)
    levels = np.linspace(0.0, 1.0, n_quantiles)
    quantiles = []
    for row in probabilities:
        positive = row > 0
        xp = support[positive]
        cp = np.cumsum(row[positive])
        cp /= cp[-1]
        quantiles.append(np.interp(levels, np.r_[0.0, cp], np.r_[xp[0], xp]))
    mean_q = np.mean(quantiles, axis=0)
    cdf = np.interp(support, mean_q, levels, left=0.0, right=1.0)
    density = np.clip(np.gradient(cdf, support), 0.0, None)
    density = np.clip(savgol_filter(density, 5, 2), 0.0, None)
    density /= density.sum()
    target_auc = np.mean(np.trapezoid(nonnegative, support, axis=1))
    density *= target_auc / np.trapezoid(density, support)
    return density


def sinkhorn_barycenter_numpy(spectra: np.ndarray, support: np.ndarray, reg: float, max_iter: int = 250) -> tuple[np.ndarray | None, dict]:
    """Small NumPy Sinkhorn barycenter trial used only to audit low-reg behavior."""
    try:
        nonnegative = np.clip(spectra, 0.0, None)
        probability = nonnegative / nonnegative.sum(axis=1, keepdims=True)
        A = probability.T
        x = ((support - support.min()) / np.ptp(support))[:, None]
        M = (x - x.T) ** 2
        M /= M.max()
        K = np.exp(-M / reg)
        if not np.isfinite(K).all() or np.count_nonzero(K) == 0:
            return None, {"reg": reg, "stable": False, "reason": "kernel underflow/non-finite"}
        UKv = K.dot((A.T / K.sum(axis=0)).T)
        u = (np.exp(np.mean(np.log(np.maximum(UKv, 1e-300)), axis=1)) / np.maximum(UKv.T, 1e-300)).T
        err = math.inf
        for iteration in range(max_iter):
            Ku = np.maximum(K.dot(u), 1e-300)
            UKv = u * K.T.dot(A / Ku)
            log_ukv = np.log(np.maximum(UKv, 1e-300))
            geo = np.exp(np.mean(log_ukv, axis=1))
            u = (u.T * geo).T / np.maximum(UKv, 1e-300)
            if iteration % 10 == 1:
                err = float(np.std(UKv, axis=1).sum())
                if not np.isfinite(err):
                    return None, {"reg": reg, "stable": False, "reason": "non-finite error", "iteration": iteration}
                if err < 1e-6:
                    break
        bary = np.exp(np.mean(np.log(np.maximum(UKv, 1e-300)), axis=1))
        bary = np.clip(bary, 0.0, None)
        if not np.isfinite(bary).all() or bary.sum() <= 0:
            return None, {"reg": reg, "stable": False, "reason": "non-finite/zero barycenter"}
        target_auc = np.mean(np.trapezoid(nonnegative, support, axis=1))
        bary *= target_auc / np.trapezoid(bary, support)
        return bary, {
            "reg": reg,
            "stable": True,
            "converged": err < 1e-6,
            "final_error": err,
            "iterations": iteration + 1,
            "peak_to_median": float(bary.max() / np.median(bary[bary > 0])),
        }
    except Exception as exc:
        return None, {"reg": reg, "stable": False, "reason": repr(exc)}


def load_tabular_controls() -> list[Spectrum]:
    main = pd.read_excel(CONTROL_XLSX)
    experimental = pd.read_excel(EXPERIMENTAL_XLSX)
    spectra: list[Spectrum] = []
    x_main = main["Raman Shift"].to_numpy(float)
    for col in main.columns:
        if col == "Raman Shift":
            continue
        spectra.append(Spectrum(col, "control", x_main, main[col].to_numpy(float), str(CONTROL_XLSX.relative_to(REPO_ROOT))))
    x_exp = experimental["Raman Shift"].to_numpy(float)
    non_bacteria = [c for c in experimental.columns if "NON BACTERIA" in c.upper()]
    for col in non_bacteria:
        spectra.append(Spectrum(col, "control", x_exp, experimental[col].to_numpy(float), str(EXPERIMENTAL_XLSX.relative_to(REPO_ROOT))))
    return spectra


def load_bacterial_tabular() -> list[Spectrum]:
    experimental = pd.read_excel(EXPERIMENTAL_XLSX)
    x_exp = experimental["Raman Shift"].to_numpy(float)
    spectra = []
    for col in experimental.columns:
        upper = col.upper()
        if col == "Raman Shift":
            continue
        if "BACTERIA" in upper and "NON BACTERIA" not in upper:
            spectra.append(Spectrum(col, "bacterial", x_exp, experimental[col].to_numpy(float), str(EXPERIMENTAL_XLSX.relative_to(REPO_ROOT))))
    return spectra


def load_uploaded_raw_samples() -> tuple[list[Spectrum], list[str]]:
    expected = [
        "Sample 1 532nm 50% 600g 50xL 10s 10times 400-3500cm-1 200hole.txt",
        "Sample 2 532nm 50% 600g 50xL 10s 10times 400-3500cm-1 200hole.txt",
        "Sample 3 532nm 50% 600g 50xL 10s 10times 400-3500cm-1 200hole.txt",
        "Sample 4 532nm 50% 600g 50xL 10s 10times 400-3500cm-1 200hole.txt",
        "Sample 5 532nm 50% 600g 50xL 8s 10times 400-3500cm-1 200hole.txt",
        "Sample 6 532nm 50% 600g 50xL 8s 10times 400-3500cm-1 200hole.txt",
    ]
    spectra = []
    missing = []
    for name in expected:
        path = DOWNLOADS / name
        if not path.exists():
            missing.append(name)
            continue
        rows = []
        skipped = 0
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                skipped += 1
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                skipped += 1
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                skipped += 1
        data = np.asarray(rows, dtype=float)
        spectra.append(Spectrum(name, "raw_unadulterated", data[:, 0], data[:, 1], str(path)))
    return spectra, missing


def process_spectra(spectra: list[Spectrum]) -> tuple[pd.DataFrame, np.ndarray, dict[str, dict[str, np.ndarray]], np.ndarray]:
    records = []
    stages_by_name = {}
    spline_matrix = []
    linear_matrix = []
    for spectrum in spectra:
        stages = preprocess_native(spectrum.x, spectrum.y)
        stages_by_name[spectrum.name] = stages
        tck = fit_functional_spline(spectrum.x, stages["normalized"])
        spline_y = spline_eval(tck)
        linear_y = linear_eval(spectrum.x, stages["normalized"])
        spline_matrix.append(spline_y)
        linear_matrix.append(linear_y)
        raw_baseline_ratio = float(np.percentile(spectrum.y, 95) / max(np.percentile(stages["baseline_corrected"], 95), 1e-12))
        records.append(
            {
                "sample_id": spectrum.name,
                "group": spectrum.group,
                "source": spectrum.source,
                "native_points": len(spectrum.x),
                "x_min": float(spectrum.x.min()),
                "x_max": float(spectrum.x.max()),
                "median_spacing": float(np.median(np.diff(spectrum.x))),
                "raw_min": float(spectrum.y.min()),
                "raw_max": float(spectrum.y.max()),
                "baseline_ratio_p95_raw_to_corrected": raw_baseline_ratio,
            }
        )
    return pd.DataFrame(records), np.vstack(spline_matrix), stages_by_name, np.vstack(linear_matrix)


def plot_preprocessing_diagnostic(spectrum: Spectrum, stages: dict[str, np.ndarray], out_path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(spectrum.x, stages["raw"], lw=0.9)
    axes[0].set_title("Raw tabular spectrum")
    axes[1].plot(spectrum.x, stages["smoothed"], lw=0.9, color="#ff7f0e")
    axes[1].set_title("Savitzky-Golay smoothed (window=5, polyorder=2)")
    axes[2].plot(spectrum.x, stages["baseline_corrected"], lw=0.9, color="#2ca02c")
    axes[2].set_title("ALS baseline-corrected and negative-clipped")
    axes[3].plot(spectrum.x, stages["normalized"], lw=0.9, color="#9467bd")
    axes[3].set_title("L2-normalized")
    axes[-1].set_xlabel("Raman shift (cm$^{-1}$)")
    for ax in axes:
        ax.set_ylabel("Intensity")
        ax.set_xlim(400, 3500)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def peak_table(curves: dict[str, np.ndarray], out_path: Path) -> pd.DataFrame:
    rows = []
    for method, y in curves.items():
        peaks, props = find_peaks(y, prominence=max(y.max() * 0.03, 1e-12), distance=8)
        if len(peaks) == 0:
            peaks = np.array([int(np.argmax(y))])
            prominences = np.array([y.max()])
        else:
            prominences = props["prominences"]
        order = np.argsort(prominences)[::-1][:15]
        for idx in peaks[order]:
            shift = float(COMMON_SHIFT[idx])
            lit = nearest_literature(shift)
            rows.append({"method": method, "peak_cm-1": shift, **lit})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def nearest_literature(shift: float) -> dict:
    diffs = np.abs(LITERATURE_TABLE["band_cm-1"].to_numpy(float) - shift)
    row = LITERATURE_TABLE.iloc[int(np.argmin(diffs))]
    delta = float(diffs.min())
    if delta <= 15:
        confidence = "close match"
    elif delta <= 40:
        confidence = "approximate"
    else:
        confidence = "unassigned"
    return {
        "matched_band_cm-1": float(row["band_cm-1"]) if confidence != "unassigned" else np.nan,
        "functional_group": row["functional_group"] if confidence != "unassigned" else "Unassigned",
        "assignment": row["assignment"] if confidence != "unassigned" else "Unassigned",
        "source_type": row["source_type"] if confidence != "unassigned" else "Unassigned",
        "match_delta_cm-1": delta,
        "match_confidence": confidence,
    }


def distance_metrics(matrix: np.ndarray, consensus: np.ndarray, weights: np.ndarray) -> pd.DataFrame:
    rows = []
    fp_mask = (COMMON_SHIFT >= FINGERPRINT_BOUNDS[0]) & (COMMON_SHIFT <= FINGERPRINT_BOUNDS[1])
    ch_mask = (COMMON_SHIFT >= CH_STRETCH_BOUNDS[0]) & (COMMON_SHIFT <= CH_STRETCH_BOUNDS[1])
    cprob = np.clip(consensus, 0, None)
    cprob = cprob / cprob.sum()
    for row in matrix:
        rprob = np.clip(row, 0, None)
        rprob = rprob / rprob.sum()
        rows.append(
            {
                "l2": float(np.linalg.norm(row - consensus)),
                "weighted_l2": float(np.sqrt(np.sum(weights * (row - consensus) ** 2))),
                "auc": float(abs(np.trapezoid(row, COMMON_SHIFT) - np.trapezoid(consensus, COMMON_SHIFT))),
                "wasserstein_1d": float(wasserstein_distance(COMMON_SHIFT, COMMON_SHIFT, u_weights=rprob, v_weights=cprob)),
                "fingerprint_distance": float(np.linalg.norm(row[fp_mask] - consensus[fp_mask])),
                "ch_stretch_distance": float(np.linalg.norm(row[ch_mask] - consensus[ch_mask])),
            }
        )
    return pd.DataFrame(rows)


def save_matrix_csv(path: Path, matrix: np.ndarray, sample_names: list[str]) -> None:
    pd.DataFrame(matrix, index=pd.Index(sample_names, name="sample_id"), columns=[f"I_{x:.2f}" for x in COMMON_SHIFT]).to_csv(path)


def plot_consensus(curves: dict[str, np.ndarray], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    for name, y in curves.items():
        ax.plot(COMMON_SHIFT, y, lw=1.4, label=name)
    ax.set(title="Corrected FDA Consensus Spectra", xlabel="Raman shift (cm$^{-1}$)", ylabel="Intensity (a.u.)", xlim=(400, 3500))
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_variance(std: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(COMMON_SHIFT, std, lw=1.2)
    ax.axvspan(1800, 2400, color="#ff7f0e", alpha=0.12, label="High natural variance noted earlier")
    ax.axvspan(2700, 3500, color="#d62728", alpha=0.10, label="High natural variance noted earlier")
    ax.set(title="Corrected Control Pointwise Standard Deviation", xlabel="Raman shift (cm$^{-1}$)", ylabel="Std dev", xlim=(400, 3500))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_corrected_pipeline() -> dict:
    ensure_dirs()
    controls = load_tabular_controls()
    if len(controls) != 45:
        raise ValueError(f"Expected 45 controls after combining structured datasets; got {len(controls)}")
    control_manifest, functional_matrix, control_stages, linear_matrix = process_spectra(controls)
    names = [s.name for s in controls]
    control_manifest.to_csv(CORRECTED_DIR / "control_source_manifest.csv", index=False)
    save_matrix_csv(CORRECTED_DIR / "corrected_control_functional_matrix.csv", functional_matrix, names)

    plot_preprocessing_diagnostic(controls[0], control_stages[controls[0].name], CORRECTED_DIR / "preprocessing_stage_diagnostic.png")

    arithmetic_mean = linear_matrix.mean(axis=0)
    spline_mean = functional_matrix.mean(axis=0)
    robust_median = geometric_median(functional_matrix)

    sinkhorn_rows = []
    sinkhorn_curve = None
    for reg in [1e-4, 1e-5, 1e-6]:
        candidate, info = sinkhorn_barycenter_numpy(functional_matrix, COMMON_SHIFT, reg)
        sinkhorn_rows.append(info)
        if candidate is not None and info.get("stable") and not info.get("converged") is False:
            sinkhorn_curve = candidate
            break
    sinkhorn_trials = pd.DataFrame(sinkhorn_rows)
    sinkhorn_trials.to_csv(CORRECTED_DIR / "wasserstein_sinkhorn_regularization_trials.csv", index=False)

    quantile_barycenter = exact_quantile_barycenter(functional_matrix)
    wasserstein_for_peaks = sinkhorn_curve if sinkhorn_curve is not None else quantile_barycenter
    wasserstein_method = "sinkhorn_low_reg" if sinkhorn_curve is not None else "exact_quantile_fallback"

    consensus = {
        "arithmetic_mean_discrete_interpolated": arithmetic_mean,
        "spline_mean_functional": spline_mean,
        "robust_median_functional": robust_median,
        f"wasserstein_{wasserstein_method}": wasserstein_for_peaks,
        "wasserstein_exact_quantile": quantile_barycenter,
    }
    consensus_df = pd.DataFrame({"raman_shift": COMMON_SHIFT, **consensus})
    consensus_df.to_csv(CORRECTED_DIR / "corrected_consensus_spectra.csv", index=False)
    plot_consensus(consensus, CORRECTED_DIR / "corrected_consensus_overlay.png")
    peak_matches = peak_table(consensus, CORRECTED_DIR / "consensus_peak_matches.csv")

    std = functional_matrix.std(axis=0, ddof=1)
    eps = max(np.median(std) * 0.05, np.finfo(float).eps)
    weights = 1.0 / (std + eps)
    weights *= len(weights) / weights.sum()
    pd.DataFrame({"raman_shift": COMMON_SHIFT, "pointwise_std": std, "variance_weight": weights}).to_csv(
        CORRECTED_DIR / "variance_weights.csv", index=False
    )
    plot_variance(std, CORRECTED_DIR / "corrected_variance_profile.png")

    distances = distance_metrics(functional_matrix, robust_median, weights)
    distances.insert(0, "sample_id", names)
    distances.to_csv(CORRECTED_DIR / "corrected_control_distances.csv", index=False)
    thresholds = []
    for metric in ["l2", "weighted_l2", "auc", "wasserstein_1d"]:
        values = distances[metric].to_numpy(float)
        thresholds.append(
            {
                "distance_metric": metric,
                "mean": values.mean(),
                "std": values.std(ddof=1),
                "90th_percentile": np.percentile(values, 90),
                "95th_percentile": np.percentile(values, 95),
                "99th_percentile": np.percentile(values, 99),
                "chosen_threshold": np.percentile(values, THRESHOLD_PERCENTILE),
            }
        )
    thresholds_df = pd.DataFrame(thresholds)
    thresholds_df.to_csv(CORRECTED_DIR / "corrected_distance_thresholds.csv", index=False)

    # NMF component audit on corrected functional control matrix.
    nmf_rows = []
    model_rows = []
    for k in range(3, 7):
        model = NMF(n_components=k, init="nndsvda", solver="cd", max_iter=5000, tol=1e-5, random_state=RANDOM_STATE)
        W = model.fit_transform(functional_matrix)
        H = model.components_
        fig, ax = plt.subplots(figsize=(13, 6))
        matched_milk = set()
        assigned_components = 0
        for j, comp in enumerate(H, start=1):
            scaled = comp / max(comp.max(), 1e-12)
            ax.plot(COMMON_SHIFT, scaled, lw=1.0, label=f"C{j}")
            peaks, props = find_peaks(scaled, prominence=0.04, distance=8)
            if len(peaks) == 0:
                peaks = np.array([int(np.argmax(scaled))])
                prominences = np.array([scaled.max()])
            else:
                prominences = props["prominences"]
            component_assigned = False
            for idx in peaks[np.argsort(prominences)[::-1][:7]]:
                lit = nearest_literature(float(COMMON_SHIFT[idx]))
                if lit["match_confidence"] != "unassigned":
                    component_assigned = True
                    if lit["source_type"] == "milk":
                        matched_milk.add(lit["matched_band_cm-1"])
                nmf_rows.append({"n_components": k, "component": j, "peak_cm-1": float(COMMON_SHIFT[idx]), **lit})
            assigned_components += int(component_assigned)
        ax.set(title=f"Corrected NMF Basis Spectra: k={k}", xlabel="Raman shift (cm$^{-1}$)", ylabel="Scaled component intensity", xlim=(400, 3500))
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(CORRECTED_DIR / f"nmf_components_k{k}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
        model_rows.append(
            {
                "n_components": k,
                "unique_milk_bands_matched": len(matched_milk),
                "components_with_assignment": assigned_components,
                "milk_band_coverage_per_component": len(matched_milk) / k,
                "relative_reconstruction_error": float(np.linalg.norm(functional_matrix - W @ H) / np.linalg.norm(functional_matrix)),
            }
        )
    nmf_all = pd.DataFrame(nmf_rows)
    nmf_models = pd.DataFrame(model_rows).sort_values(
        ["milk_band_coverage_per_component", "unique_milk_bands_matched", "relative_reconstruction_error"],
        ascending=[False, False, True],
    )
    selected_k = int(nmf_models.iloc[0]["n_components"])
    nmf_all.to_csv(CORRECTED_DIR / "nmf_peak_assignments_all.csv", index=False)
    nmf_models.to_csv(CORRECTED_DIR / "nmf_model_comparison.csv", index=False)
    nmf_all[nmf_all["n_components"] == selected_k].to_csv(CORRECTED_DIR / "nmf_selected_component_summary.csv", index=False)

    # Wavelet decomposition of corrected robust median.
    coeffs = pywt.wavedec(robust_median, "sym8", level=4, mode="symmetric")

    def reconstruct(keep_index: int) -> np.ndarray:
        retained = [np.zeros_like(c) for c in coeffs]
        retained[keep_index] = coeffs[keep_index]
        return pywt.waverec(retained, "sym8", mode="symmetric")[: len(robust_median)]

    approximation = reconstruct(0)
    details = {level: reconstruct(4 - level + 1) for level in range(1, 5)}
    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
    axes[0].plot(COMMON_SHIFT, approximation, lw=1.0)
    axes[0].set_title("Approximation A4")
    peak_region = ((COMMON_SHIFT >= 800) & (COMMON_SHIFT <= 1800)) | ((COMMON_SHIFT >= 2800) & (COMMON_SHIFT <= 3000))
    wavelet_rows = []
    for level, ax in zip(range(1, 5), axes[1:]):
        detail = details[level]
        ax.plot(COMMON_SHIFT, detail, lw=0.8)
        ax.set_title(f"Detail D{level}")
        wavelet_rows.append(
            {
                "detail_level": level,
                "peak_region_energy_ratio": float(np.sum(detail[peak_region] ** 2) / np.sum(detail**2)),
                "total_energy": float(np.sum(detail**2)),
            }
        )
    for ax in axes:
        ax.set_xlim(400, 3500)
        ax.set_ylabel("Amplitude")
    axes[-1].set_xlabel("Raman shift (cm$^{-1}$)")
    fig.tight_layout()
    fig.savefig(CORRECTED_DIR / "wavelet_decomposition.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    wavelet_summary = pd.DataFrame(wavelet_rows).sort_values("peak_region_energy_ratio", ascending=False)
    wavelet_summary.to_csv(CORRECTED_DIR / "wavelet_level_summary.csv", index=False)

    return {
        "controls": controls,
        "control_matrix": functional_matrix,
        "control_names": names,
        "consensus": consensus,
        "robust_median": robust_median,
        "weights": weights,
        "thresholds": thresholds_df,
        "distances": distances,
        "selected_nmf_k": selected_k,
        "wasserstein_method": wasserstein_method,
        "sinkhorn_trials": sinkhorn_trials,
    }


def evaluate_validation_sets(context: dict) -> dict:
    bacterial = load_bacterial_tabular()
    raw_samples, missing_raw = load_uploaded_raw_samples()
    all_tests = bacterial + raw_samples
    test_manifest, test_matrix, test_stages, _ = process_spectra(all_tests)
    test_manifest.to_csv(NOTEBOOK3_DIR / "test_sample_manifest.csv", index=False)
    save_matrix_csv(NOTEBOOK3_DIR / "test_functional_matrix.csv", test_matrix, [s.name for s in all_tests])

    if raw_samples:
        plot_preprocessing_diagnostic(raw_samples[0], test_stages[raw_samples[0].name], NOTEBOOK3_DIR / "uploaded_raw_preprocessing_diagnostic.png")
    if bacterial:
        plot_preprocessing_diagnostic(bacterial[0], test_stages[bacterial[0].name], NOTEBOOK3_DIR / "bacterial_preprocessing_diagnostic.png")

    robust_median = context["robust_median"]
    weights = context["weights"]
    thresholds = context["thresholds"].set_index("distance_metric")["chosen_threshold"].to_dict()
    test_dist = distance_metrics(test_matrix, robust_median, weights)
    test_dist.insert(0, "sample_id", [s.name for s in all_tests])
    test_dist.insert(1, "sample_group", [s.group for s in all_tests])
    for metric in ["l2", "weighted_l2", "auc", "wasserstein_1d"]:
        test_dist[f"normalized_{metric}"] = test_dist[metric] / thresholds[metric]
    norm_cols = [f"normalized_{m}" for m in ["l2", "weighted_l2", "auc", "wasserstein_1d"]]
    test_dist["authenticity_score"] = test_dist[norm_cols].mean(axis=1)
    test_dist["above_threshold"] = test_dist["authenticity_score"] > 1.0
    test_dist.to_csv(NOTEBOOK3_DIR / "test_distance_metrics_and_scores.csv", index=False)

    control_dist = context["distances"].copy()
    control_dist["sample_group"] = "control"
    for metric in ["l2", "weighted_l2", "auc", "wasserstein_1d"]:
        control_dist[f"normalized_{metric}"] = control_dist[metric] / thresholds[metric]
    control_dist["authenticity_score"] = control_dist[norm_cols].mean(axis=1)
    control_dist["above_threshold"] = control_dist["authenticity_score"] > 1.0

    combined_for_roc = pd.concat(
        [
            control_dist[["sample_id", "sample_group", "l2", "weighted_l2", "auc", "wasserstein_1d", "authenticity_score", "above_threshold"]],
            test_dist[["sample_id", "sample_group", "l2", "weighted_l2", "auc", "wasserstein_1d", "authenticity_score", "above_threshold"]],
        ],
        ignore_index=True,
    )
    combined_for_roc.to_csv(NOTEBOOK3_DIR / "control_and_test_scores_for_roc.csv", index=False)

    metric_auc_rows = []
    roc_plot_data = {}
    bacterial_mask = combined_for_roc["sample_group"].isin(["control", "bacterial"])
    binary_df = combined_for_roc.loc[bacterial_mask].copy()
    y_true = (binary_df["sample_group"] == "bacterial").astype(int).to_numpy()
    for metric in ["authenticity_score", "l2", "weighted_l2", "auc", "wasserstein_1d"]:
        scores = binary_df[metric].to_numpy(float)
        metric_auc = roc_auc_score(y_true, scores)
        fpr, tpr, _ = roc_curve(y_true, scores)
        metric_auc_rows.append({"metric": metric, "auroc_bacterial_vs_control": metric_auc})
        roc_plot_data[metric] = (fpr, tpr, metric_auc)
    metric_aucs = pd.DataFrame(metric_auc_rows).sort_values("auroc_bacterial_vs_control", ascending=False)
    metric_aucs.to_csv(NOTEBOOK3_DIR / "bacterial_vs_control_metric_aurocs.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    for metric, (fpr, tpr, metric_auc) in roc_plot_data.items():
        ax.plot(fpr, tpr, lw=1.4, label=f"{metric} (AUC={metric_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(title="ROC: Bacterial Samples vs Corrected Controls", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(NOTEBOOK3_DIR / "roc_curves_bacterial_vs_control.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    plot_df = test_dist.copy()
    fig, ax = plt.subplots(figsize=(13, 5.5))
    colors = plot_df["sample_group"].map({"bacterial": "#d62728", "raw_unadulterated": "#2ca02c"}).fillna("#1f77b4")
    ax.bar(np.arange(len(plot_df)), plot_df["authenticity_score"], color=colors)
    ax.axhline(1.0, color="black", ls="--", lw=1.3, label="score = 1")
    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_xticklabels(plot_df["sample_id"], rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("Authenticity score")
    ax.set_title("Authenticity Scores for Bacterial and Uploaded Raw Milk Samples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(NOTEBOOK3_DIR / "authenticity_score_bar_chart.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary_rows = []
    for group, group_df in combined_for_roc.groupby("sample_group"):
        row = {
            "sample_group": group,
            "n": len(group_df),
            "mean_authenticity_score": group_df["authenticity_score"].mean(),
            "percent_above_threshold": 100.0 * group_df["above_threshold"].mean(),
            "auroc_combined_score": np.nan,
            "best_individual_metric": "",
            "best_individual_metric_auroc": np.nan,
        }
        if group == "bacterial":
            row["auroc_combined_score"] = float(metric_aucs.loc[metric_aucs["metric"] == "authenticity_score", "auroc_bacterial_vs_control"].iloc[0])
            best_individual = metric_aucs[metric_aucs["metric"] != "authenticity_score"].iloc[0]
            row["best_individual_metric"] = best_individual["metric"]
            row["best_individual_metric_auroc"] = float(best_individual["auroc_bacterial_vs_control"])
        summary_rows.append(row)
    validation_summary = pd.DataFrame(summary_rows)
    validation_summary.to_csv(NOTEBOOK3_DIR / "validation_summary_table.csv", index=False)

    pd.DataFrame({"missing_uploaded_raw_sample": missing_raw}).to_csv(NOTEBOOK3_DIR / "missing_uploaded_samples.csv", index=False)
    return {
        "bacterial_count": len(bacterial),
        "raw_count": len(raw_samples),
        "missing_raw": missing_raw,
        "test_scores": test_dist,
        "summary": validation_summary,
        "metric_aucs": metric_aucs,
    }


def write_notes(context: dict, validation: dict) -> None:
    def as_markdown(df: pd.DataFrame) -> str:
        text_df = df.copy()
        for col in text_df.columns:
            if pd.api.types.is_float_dtype(text_df[col]):
                text_df[col] = text_df[col].map(lambda v: "" if pd.isna(v) else f"{v:.6g}")
            else:
                text_df[col] = text_df[col].fillna("").astype(str)
        headers = list(text_df.columns)
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in text_df.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
        return "\n".join(lines)

    thresholds = as_markdown(context["thresholds"])
    summary = as_markdown(validation["summary"])
    aurocs = as_markdown(validation["metric_aucs"])
    missing = ", ".join(validation["missing_raw"]) if validation["missing_raw"] else "None"
    notes = f"""# Corrected FDA Raman Validation Notes

## What Was Fixed

- Preprocessing now starts from the existing structured tabular datasets:
  `src/combined_raman_spectra.xlsx` plus the explicit non-bacterial D2 control
  in `src/combined_raman_spectra (Experimental).xlsx`. The previous generated
  pipeline mixed raw-file loading and normalization-first behavior; this pass
  applies the documented sequence: Savitzky-Golay smoothing, ALS baseline
  correction, negative clipping, and L2 normalization.
- Each preprocessed spectrum is fitted with a cubic B-spline (`splrep`, `k=3`,
  `s=1e-4`) before functional evaluation on the common 400-3500 cm-1 grid.
  This is the FDA step: it treats spectra as functions and makes the spline
  mean distinct from the discrete interpolated arithmetic mean.
- The Wasserstein curve is reframed as a shared dominant-peak locator. Low-reg
  Sinkhorn trials are recorded in
  `corrected_pipeline/wasserstein_sinkhorn_regularization_trials.csv`; the
  stable fallback used for peak identification is `{context['wasserstein_method']}`.
- Notebook 2-style NMF, wavelet, variance weights, and thresholds were rerun
  against the corrected functional control matrix.
- Notebook 3-style validation was run against the 9 bacterial spectra in the
  structured experimental workbook and the uploaded raw unadulterated samples.

## Region Definitions and Citations

- Fingerprint region: this implementation uses 400-1800 cm-1 because the
  reference peak table includes milk-relevant protein, lipid, and carbohydrate
  bands throughout that span. Literature boundaries vary: biomedical Raman work
  commonly uses 400-1800 cm-1, while other summaries use roughly 600-1800 or
  800-1800 cm-1.
- CH-stretch region: this implementation uses 2800-3000 cm-1, within the
  broader high-wavenumber region often described as 2800-3800 cm-1. The upper
  bound is narrowed to the instrument range and the supplied milk table, where
  fatty-acid peaks at 2865 and 2902 cm-1 are central.
- Sources consulted:
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC2715834/
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC10052158/
  - https://physicsopenlab.org/2022/01/11/raman-spectroscopy-of-organic-and-inorganic-molecules/
  - https://pubs.acs.org/doi/10.1021/acs.analchem.5c05031

## Corrected Thresholds

{thresholds}

## Validation Summary

{summary}

## Bacterial-vs-Control AUROC by Metric

{aurocs}

## Uploaded Raw Sample Limitation

The user described six unadulterated raw milk samples, but only five files were
present in Downloads: Samples 2-6. Missing file(s): {missing}. Because the
uploaded raw set is all labelled unadulterated and contains no adulterated
counter-class, AUROC is not computed for that set; raw scores are reported.
"""
    (OUT_ROOT / "VALIDATION_README.md").write_text(notes, encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    context = run_corrected_pipeline()
    validation = evaluate_validation_sets(context)
    write_notes(context, validation)
    print("Corrected pipeline outputs:", CORRECTED_DIR)
    print("Notebook 3 validation outputs:", NOTEBOOK3_DIR)
    print("Validation notes:", OUT_ROOT / "VALIDATION_README.md")
    print("Thresholds:")
    print(context["thresholds"].to_string(index=False))
    print("Validation summary:")
    print(validation["summary"].to_string(index=False))
    print("Metric AUROCs:")
    print(validation["metric_aucs"].to_string(index=False))
    if validation["missing_raw"]:
        print("Missing uploaded raw sample files:", validation["missing_raw"])


if __name__ == "__main__":
    main()
