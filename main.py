from src.config import (
    TARGET_COL,
    TRAIN_RATIO,
    VALID_RATIO,
    TEST_RATIO,
    N_BATCHES
)
from src.data_loader import load_data, basic_cleaning
from src.preprocess import get_column_types, build_preprocessor
from src.simulator import chronological_split
from src.train import train_pipeline, evaluate_model, save_pipeline
from src.evaluate import evaluate_batches


def main():
    print("Starting fraud detection pipeline...")

    df = load_data()
    df = basic_cleaning(df)

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

    from pathlib import Path
    from src.config import RESULTS_DIR

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    batch_results.to_csv(RESULTS_DIR / "phase1_batch_results.csv", index=False)
    print(f"Saved batch results to: {RESULTS_DIR / 'phase1_batch_results.csv'}")

    save_pipeline(pipeline)


if __name__ == "__main__":
    main()