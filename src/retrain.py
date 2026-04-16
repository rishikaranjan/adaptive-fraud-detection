from src.train import train_pipeline
from sklearn.metrics import average_precision_score


def retrain_and_evaluate(X_train, y_train, X_val, y_val, preprocessor):
    new_pipeline = train_pipeline(X_train, y_train, preprocessor)

    y_proba = new_pipeline.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, y_proba)

    return new_pipeline, pr_auc