import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    classification_report,
)

from src.dataset import (
    ID_TO_SENTIMENT,
    ID_TO_TOXICITY,
    SENTIMENT_NAMES,
    TOXICITY_NAMES,
)


def compute_metrics(
    sentiment_preds: np.ndarray,
    sentiment_targets: np.ndarray,
    toxicity_preds: np.ndarray,
    toxicity_targets: np.ndarray,
    toxicity_probs: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Tính toán hệ thống thước đo toàn diện theo chuẩn Giai đoạn 6 của đặc tả:
    - Macro-F1 (Sentiment & Toxicity)
    - Per-class F1
    - Precision, Recall
    - PR-AUC cho từng lớp Toxicity (đặc biệt các lớp hiếm 1, 2, 3)
    - Confusion Matrix
    """
    metrics = {}

    # --- Sentiment Metrics ---
    metrics["sentiment_macro_f1"] = float(
        f1_score(sentiment_targets, sentiment_preds, average="macro", zero_division=0)
    )
    sentiment_per_class = f1_score(
        sentiment_targets, sentiment_preds, average=None, zero_division=0
    )
    for idx, score in enumerate(sentiment_per_class):
        name = SENTIMENT_NAMES.get(idx, f"Class {idx}")
        metrics[f"sentiment_f1_{name}"] = float(score)

    # --- Toxicity Metrics ---
    metrics["toxicity_macro_f1"] = float(
        f1_score(toxicity_targets, toxicity_preds, average="macro", zero_division=0)
    )
    toxicity_per_class = f1_score(
        toxicity_targets, toxicity_preds, average=None, zero_division=0
    )
    for idx, score in enumerate(toxicity_per_class):
        name = TOXICITY_NAMES.get(idx, f"Class {idx}")
        metrics[f"toxicity_f1_{name}"] = float(score)

    # --- Composite Score (Dùng cho Composite Early Stopping) ---
    metrics["composite_score"] = float(
        0.5 * metrics["sentiment_macro_f1"] + 0.5 * metrics["toxicity_macro_f1"]
    )

    # --- PR-AUC cho Toxicity (đánh giá phân tách lớp hiếm độc lập với ngưỡng) ---
    if toxicity_probs is not None:
        num_classes = toxicity_probs.shape[1]
        for c in range(num_classes):
            binary_target = (toxicity_targets == c).astype(int)
            if np.sum(binary_target) > 0:
                p, r, _ = precision_recall_curve(binary_target, toxicity_probs[:, c])
                pr_auc_val = float(auc(r, p))
                name = TOXICITY_NAMES.get(c, f"Class {c}")
                metrics[f"toxicity_pr_auc_{name}"] = pr_auc_val

    # --- Confusion Matrices ---
    metrics["sentiment_confusion_matrix"] = confusion_matrix(
        sentiment_targets, sentiment_preds
    ).tolist()
    metrics["toxicity_confusion_matrix"] = confusion_matrix(
        toxicity_targets, toxicity_preds
    ).tolist()

    return metrics


def print_evaluation_summary(metrics: Dict[str, Any], title: str = "EVALUATION REPORT"):
    """In bảng tổng kết kết quả đánh giá trực quan."""
    print("=" * 65)
    print(f" {title} ")
    print("=" * 65)
    print(f"Composite Macro-F1 Score: {metrics['composite_score']:.4f}")
    print("-" * 65)
    print(f"Sentiment Macro-F1:       {metrics['sentiment_macro_f1']:.4f}")
    for k, v in metrics.items():
        if k.startswith("sentiment_f1_"):
            cls_name = k.replace("sentiment_f1_", "")
            print(f"  • {cls_name:<20}: {v:.4f}")

    print("-" * 65)
    print(f"Toxicity Macro-F1:        {metrics['toxicity_macro_f1']:.4f}")
    for k, v in metrics.items():
        if k.startswith("toxicity_f1_"):
            cls_name = k.replace("toxicity_f1_", "")
            pr_key = f"toxicity_pr_auc_{cls_name}"
            pr_str = f" | PR-AUC: {metrics[pr_key]:.4f}" if pr_key in metrics else ""
            print(f"  • {cls_name:<20}: {v:.4f}{pr_str}")

    print("=" * 65)
