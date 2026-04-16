import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score

from src.config import MODELS_DIR, RANDOM_STATE


def compute_scale_pos_weight(y):
    negative_count = (y == 0).sum()
    positive_count = (y == 1).sum()

    if positive_count == 0:
        return 1.0

    return negative_count / positive_count


def train_pipeline(X_train, y_train, preprocessor):
    scale_pos_weight = compute_scale_pos_weight(y_train)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1
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


def save_pipeline(pipeline, filename="baseline_xgb_pipeline.joblib"):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / filename
    joblib.dump(pipeline, output_path)
    print(f"Saved pipeline to: {output_path}")