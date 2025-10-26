from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

def evaluate_predictions(y_true, y_score, threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int)

    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "threshold": threshold
    }
