import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def calculate_psi(expected, actual, bins=10):
    """
    Calculate Population Stability Index (PSI) between two numeric arrays.
    """
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()

    if expected.empty or actual.empty:
        return np.nan

    # Create breakpoints from expected distribution
    breakpoints = np.linspace(0, 100, bins + 1)
    expected_bins = np.percentile(expected, breakpoints)

    # Remove duplicate bin edges
    expected_bins = np.unique(expected_bins)

    if len(expected_bins) < 2:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=expected_bins)
    actual_counts, _ = np.histogram(actual, bins=expected_bins)

    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)

    # Avoid division by zero / log(0)
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi


def ks_drift_test(reference, current, alpha=0.05):
    """
    Run Kolmogorov-Smirnov test between reference and current numeric feature.
    """
    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    if reference.empty or current.empty:
        return {
            "ks_stat": np.nan,
            "p_value": np.nan,
            "drift_flag": False
        }

    ks_stat, p_value = ks_2samp(reference, current)

    return {
        "ks_stat": ks_stat,
        "p_value": p_value,
        "drift_flag": p_value < alpha
    }


def detect_drift_for_batch(reference_df, batch_df, numeric_cols, alpha=0.05, psi_threshold=0.2):
    """
    Detect drift for all numeric columns in one batch.
    Returns detailed per-feature drift results.
    """
    results = []

    for col in numeric_cols:
        ref_values = reference_df[col]
        batch_values = batch_df[col]

        ks_result = ks_drift_test(ref_values, batch_values, alpha=alpha)
        psi_value = calculate_psi(ref_values, batch_values)

        results.append({
            "feature": col,
            "ks_stat": ks_result["ks_stat"],
            "p_value": ks_result["p_value"],
            "ks_drift": ks_result["drift_flag"],
            "psi": psi_value,
            "psi_drift": psi_value > psi_threshold if not pd.isna(psi_value) else False
        })

    return pd.DataFrame(results)


def summarize_batch_drift(feature_drift_df, batch_id):
    """
    Create a high-level drift summary for one batch.
    """
    ks_drift_count = feature_drift_df["ks_drift"].sum()
    psi_drift_count = feature_drift_df["psi_drift"].sum()

    summary = {
        "batch_id": batch_id,
        "total_features_checked": len(feature_drift_df),
        "ks_drifted_features": int(ks_drift_count),
        "psi_drifted_features": int(psi_drift_count),
        "avg_psi": feature_drift_df["psi"].mean(),
        "max_psi": feature_drift_df["psi"].max()
    }

    return summary