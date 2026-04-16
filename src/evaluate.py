import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score


def evaluate_batch(pipeline, X_batch, y_batch, batch_id):
    y_pred = pipeline.predict(X_batch)
    y_proba = pipeline.predict_proba(X_batch)[:, 1]

    result = {
        "batch_id": batch_id,
        "precision": precision_score(y_batch, y_pred, zero_division=0),
        "recall": recall_score(y_batch, y_pred, zero_division=0),
        "f1": f1_score(y_batch, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_batch, y_proba),
        "roc_auc": roc_auc_score(y_batch, y_proba) if len(set(y_batch)) > 1 else None,
        "fraud_rate": y_batch.mean(),
        "batch_size": len(y_batch)
    }

    return result


def evaluate_batches(pipeline, X_test, y_test, n_batches=10):
    batch_size = len(X_test) // n_batches
    all_results = []

    for i in range(n_batches):
        start = i * batch_size
        end = (i + 1) * batch_size if i < n_batches - 1 else len(X_test)

        X_batch = X_test.iloc[start:end]
        y_batch = y_test.iloc[start:end]

        result = evaluate_batch(pipeline, X_batch, y_batch, batch_id=i + 1)
        all_results.append(result)

    return pd.DataFrame(all_results)