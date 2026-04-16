import pandas as pd


def get_top_drifting_features(feature_drift_df, top_n=10):
    top_results = []

    for batch_id in sorted(feature_drift_df["batch_id"].unique()):
        batch_df = feature_drift_df[feature_drift_df["batch_id"] == batch_id].copy()

        top_batch = (
            batch_df.sort_values("psi", ascending=False)
            .head(top_n)[["batch_id", "feature", "psi", "ks_drift", "psi_drift"]]
        )

        top_results.append(top_batch)

    return pd.concat(top_results, ignore_index=True)


def get_global_drift_ranking(feature_drift_df):
    ranking = (
        feature_drift_df.groupby("feature")
        .agg(
            avg_psi=("psi", "mean"),
            max_psi=("psi", "max"),
            ks_drift_count=("ks_drift", "sum"),
            psi_drift_count=("psi_drift", "sum")
        )
        .reset_index()
        .sort_values(["avg_psi", "max_psi"], ascending=False)
    )

    return ranking