import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score

from src.config import MODELS_DIR, RANDOM_STATE


def train_pipeline(X_train, y_train, preprocessor):
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(pipeline, X_eval, y_eval, dataset_name="Validation"):
    y_pred = pipeline.predict(X_eval)
    y_proba = pipeline.predict_proba(X_eval)[:, 1]

    print(f"\n--- {dataset_name} Metrics ---")
    print(classification_report(y_eval, y_pred, digits=4))

    pr_auc = average_precision_score(y_eval, y_proba)
    roc_auc = roc_auc_score(y_eval, y_proba)

    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc
    }


def save_pipeline(pipeline, filename="baseline_pipeline.joblib"):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / filename
    joblib.dump(pipeline, output_path)
    print(f"Saved pipeline to: {output_path}")