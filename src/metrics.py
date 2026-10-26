from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix


def classification_metrics(y_true, y_pred, class_names):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "classification_report": classification_report(
            y_true, y_pred, labels=list(range(len(class_names))),
            target_names=class_names, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(class_names)))
        ).tolist(),
    }
