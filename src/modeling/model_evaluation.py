"""
model_evaluation.py
Comprehensive evaluation metrics for imbalanced binary classification,
including threshold optimization and cost-sensitive business analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve, auc
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates a trained classification model with imbalance-aware metrics."""

    def __init__(self, model, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Args:
            model: A fitted, scikit-learn-compatible classifier (or pipeline).
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target (true labels).
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)
        self.y_proba = model.predict_proba(X_test)[:, 1]

    def get_standard_metrics(self) -> dict:
        """
        Calculate Accuracy, Precision, Recall, and F1-score at the default threshold.

        Returns:
            dict: The four standard classification metrics.
        """
        metrics = {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'precision': precision_score(self.y_test, self.y_pred),
            'recall': recall_score(self.y_test, self.y_pred),
            'f1_score': f1_score(self.y_test, self.y_pred)
        }
        logger.info(f"Standard metrics: {metrics}")
        return metrics

    def get_confusion_matrix(self) -> dict:
        """
        Get the confusion matrix broken down into named components.

        Returns:
            dict: true_negatives, false_positives, false_negatives, true_positives.
        """
        tn, fp, fn, tp = confusion_matrix(self.y_test, self.y_pred).ravel()
        result = {
            'true_negatives': int(tn), 'false_positives': int(fp),
            'false_negatives': int(fn), 'true_positives': int(tp)
        }
        logger.info(f"Confusion matrix: {result}")
        return result

    def get_pr_auc(self) -> float:
        """
        Calculate Precision-Recall AUC - more informative than ROC-AUC
        for imbalanced classification problems.

        Returns:
            float: The PR-AUC score.
        """
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_proba)
        pr_auc = auc(recall, precision)
        logger.info(f"PR-AUC: {pr_auc:.4f}")
        return pr_auc

    def find_optimal_threshold(self, metric: str = 'f1') -> dict:
        """
        Search across thresholds (0.1 to 0.9) to find the one that
        maximizes the given metric, instead of using the default 0.5.

        Args:
            metric (str): Metric to optimize. Currently supports 'f1'.

        Returns:
            dict: best_threshold, best_score, and score at default threshold (0.5).
        """
        thresholds = np.arange(0.1, 0.9, 0.01)
        scores = []

        for t in thresholds:
            y_pred_t = (self.y_proba >= t).astype(int)
            scores.append(f1_score(self.y_test, y_pred_t))

        best_idx = int(np.argmax(scores))
        default_score = f1_score(self.y_test, (self.y_proba >= 0.5).astype(int))

        result = {
            'best_threshold': round(float(thresholds[best_idx]), 2),
            'best_score': scores[best_idx],
            'default_threshold_score': default_score
        }
        logger.info(f"Optimal threshold: {result['best_threshold']} "
                    f"(F1={result['best_score']:.4f} vs default F1={default_score:.4f})")
        return result

    def calculate_business_cost(self, fn_cost: float, fp_cost: float) -> dict:
        """
        Translate confusion matrix errors into a total projected business cost.

        Args:
            fn_cost (float): Cost of a missed churner (e.g. annual revenue lost).
            fp_cost (float): Cost of a false alarm (e.g. wasted retention campaign).

        Returns:
            dict: Breakdown of costs and the total.
        """
        cm = self.get_confusion_matrix()
        fn_total = cm['false_negatives'] * fn_cost
        fp_total = cm['false_positives'] * fp_cost
        total_cost = fn_total + fp_total

        result = {
            'false_negative_cost': fn_total,
            'false_positive_cost': fp_total,
            'total_cost': total_cost
        }
        logger.info(f"Business cost: ${total_cost:,.2f} "
                    f"(FN cost: ${fn_total:,.2f}, FP cost: ${fp_total:,.2f})")
        return result

    def full_report(self, fn_cost: float = None, fp_cost: float = None) -> dict:
        """
        Generate a complete evaluation report combining all metrics above.

        Args:
            fn_cost (float, optional): If provided (with fp_cost), includes
                business cost analysis in the report.
            fp_cost (float, optional): See fn_cost.

        Returns:
            dict: All evaluation results combined.
        """
        report = {
            **self.get_standard_metrics(),
            'confusion_matrix': self.get_confusion_matrix(),
            'pr_auc': self.get_pr_auc(),
            'optimal_threshold': self.find_optimal_threshold()
        }

        if fn_cost is not None and fp_cost is not None:
            report['business_cost'] = self.calculate_business_cost(fn_cost, fp_cost)

        return report