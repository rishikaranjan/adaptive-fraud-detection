import pandas as pd

from src.config import (
    TARGET_COL,
    TRAIN_RATIO,
    VALID_RATIO,
    TEST_RATIO,
    N_BATCHES,
    RESULTS_DIR
)
from src.data_loader import load_data, basic_cleaning
from src.preprocess import get_column_types, build_preprocessor
from src.simulator import chronological_split
from src.train import train_pipeline, evaluate_model, save_pipeline
from src.evaluate import evaluate_batches
from src.drift import detect_drift_for_batch, summarize_batch_drift
from src.drift_analysis import get_top_drifting_features, get_global_drift_ranking
from src.retrain import retrain_and_evaluate
from sklearn.metrics import average_precision_score



def main():
    print("Starting fraud detection pipeline...")

    df = load_data()
    df = basic_cleaning(df)

    # Drop useless ID/time columns (IMPORTANT)
    DROP_COLS = ["TransactionID", "TransactionDT"]
    df = df.drop(columns=DROP_COLS, errors="ignore")


    print(f"Dataset shape: {df.shape}")
    print(f"Fraud rate: {df[TARGET_COL].mean():.6f}")

    train_df, valid_df, test_df = chronological_split(
        df,
        train_ratio=TRAIN_RATIO,
        valid_ratio=VALID_RATIO,
        test_ratio=TEST_RATIO
    )

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    X_valid = valid_df.drop(columns=[TARGET_COL])
    y_valid = valid_df[TARGET_COL]

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    numeric_cols, categorical_cols = get_column_types(train_df, TARGET_COL)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    pipeline = train_pipeline(X_train, y_train, preprocessor)

    evaluate_model(pipeline, X_valid, y_valid, dataset_name="Validation")
    evaluate_model(pipeline, X_test, y_test, dataset_name="Test")

    batch_results = evaluate_batches(pipeline, X_test, y_test, n_batches=N_BATCHES)

    print("\nBatch-wise evaluation:")
    print(batch_results)

    # Save batch metrics
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    batch_results.to_csv(RESULTS_DIR / "phase1_batch_results.csv", index=False)
    print(f"Saved batch results to: {RESULTS_DIR / 'phase1_batch_results.csv'}")

    # Drift detection per batch
    print("\nRunning drift detection...")
    batch_size = len(X_test) // N_BATCHES
    drift_summaries = []
    all_feature_drift_results = []

    for i in range(N_BATCHES):
        start = i * batch_size
        end = (i + 1) * batch_size if i < N_BATCHES - 1 else len(X_test)

        X_batch = X_test.iloc[start:end]

        feature_drift_df = detect_drift_for_batch(
            reference_df=X_train,
            batch_df=X_batch,
            numeric_cols=numeric_cols,
            alpha=0.05,
            psi_threshold=0.2
        )

        feature_drift_df["batch_id"] = i + 1
        all_feature_drift_results.append(feature_drift_df)

        summary = summarize_batch_drift(feature_drift_df, batch_id=i + 1)
        drift_summaries.append(summary)


    drift_summary_df = pd.DataFrame(drift_summaries)
    feature_drift_all_df = pd.concat(all_feature_drift_results, ignore_index=True)


    print("\nBatch-wise drift summary:")
    print(drift_summary_df)

    drift_summary_df.to_csv(RESULTS_DIR / "phase2_batch_drift_summary.csv", index=False)
    feature_drift_all_df.to_csv(RESULTS_DIR / "phase2_feature_drift_details.csv", index=False)

    print(f"Saved drift summary to: {RESULTS_DIR / 'phase2_batch_drift_summary.csv'}")
    print(f"Saved feature-level drift details to: {RESULTS_DIR / 'phase2_feature_drift_details.csv'}")


    # 🔥 Phase 2.5: Drift analysis summaries
    top_drift_df = get_top_drifting_features(feature_drift_all_df, top_n=10)
    global_drift_df = get_global_drift_ranking(feature_drift_all_df)

    # Save results
    top_drift_df.to_csv(RESULTS_DIR / "phase25_top_drift_per_batch.csv", index=False)
    global_drift_df.to_csv(RESULTS_DIR / "phase25_global_drift_ranking.csv", index=False)

    print(f"Saved top drift per batch to: {RESULTS_DIR / 'phase25_top_drift_per_batch.csv'}")
    
    
    print(f"Saved global drift ranking to: {RESULTS_DIR / 'phase25_global_drift_ranking.csv'}")


    # 🚀 Phase 3: Adaptive Retraining
    print("\nRunning adaptive retraining...")

    batch_size = len(X_test) // N_BATCHES
    current_pipeline = pipeline

    adaptive_results = []



    for i in range(N_BATCHES):
        start = i * batch_size
        end = (i + 1) * batch_size if i < N_BATCHES - 1 else len(X_test)

        X_batch = X_test.iloc[start:end]
        y_batch = y_test.iloc[start:end]

        # Evaluate current model
        y_proba_old = current_pipeline.predict_proba(X_batch)[:, 1]
        old_pr_auc = average_precision_score(y_batch, y_proba_old)

        # Drift detection
        feature_drift_df = detect_drift_for_batch(
            reference_df=X_train,
            batch_df=X_batch,
            numeric_cols=numeric_cols
        )

        avg_psi = feature_drift_df["psi"].mean()

        retrained = False
        new_pr_auc = old_pr_auc

        # 🔥 Trigger retraining
        if avg_psi > 0.1:
            retrained = True

            # 🚀 Combine old + new data (ADAPTIVE LEARNING)
            X_retrain = pd.concat([X_train, X_batch])
            y_retrain = pd.concat([y_train, y_batch])

            new_pipeline, new_pr_auc = retrain_and_evaluate(
                X_retrain, y_retrain, X_batch, y_batch, preprocessor
            )
            

            # Replace model if better
            if new_pr_auc > old_pr_auc:
                current_pipeline = new_pipeline

        adaptive_results.append({
            "batch_id": i + 1,
            "avg_psi": avg_psi,
            "old_pr_auc": old_pr_auc,
            "new_pr_auc": new_pr_auc,
            "retrained": retrained
        })

    adaptive_df = pd.DataFrame(adaptive_results)
    adaptive_df.to_csv(RESULTS_DIR / "phase3_adaptive_results.csv", index=False)

    print("\nAdaptive results:")
    print(adaptive_df)


    # Save final model
    save_pipeline(current_pipeline)






if __name__ == "__main__":
    main()