"""
model_training.py
Handles model training with support for simple holdout training
and stratified cross-validation strategies.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains a model using either a simple holdout fit or cross-validation."""

    def __init__(self, model, scale_features: bool = True):
        """
        Args:
            model: A scikit-learn-compatible classifier (e.g. from ModelFactory).
            scale_features (bool): Whether to include StandardScaler in the
                training pipeline. Defaults to True.
        """
        self.model = model

        if scale_features:
            self.pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
        else:
            self.pipeline = Pipeline([
                ('classifier', model)
            ])

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train the model on the full training set (simple holdout approach).

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.

        Returns:
            The fitted pipeline.
        """
        self.pipeline.fit(X_train, y_train)
        logger.info("Model training complete (holdout method).")
        return self.pipeline

    def train_with_cross_validation(self, X_train: pd.DataFrame, y_train: pd.Series,
                                     n_splits: int = 5, scoring: str = 'f1'):
        """
        Train and evaluate the model using stratified k-fold cross-validation.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            n_splits (int): Number of folds. Defaults to 5.
            scoring (str): Metric to optimize/report. Defaults to 'f1'
                (better suited than accuracy for imbalanced data).

        Returns:
            dict: Contains individual fold scores, mean, and std deviation.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(self.pipeline, X_train, y_train, cv=skf, scoring=scoring)

        result = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }

        logger.info(f"Cross-validation ({n_splits}-fold, {scoring}): "
                    f"mean={result['mean']:.4f}, std={result['std']:.4f}")

        # Fit on full training data after CV, so the pipeline is ready to use
        self.pipeline.fit(X_train, y_train)

        return result