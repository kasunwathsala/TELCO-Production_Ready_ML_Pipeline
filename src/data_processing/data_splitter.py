"""
data_splitter.py
Splits data into train/test sets with stratification support
to preserve class balance across splits.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSplitter:
    """Splits features and target into training and test sets."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42, stratify: bool = True):
        """
        Args:
            test_size (float): Proportion of data to reserve for testing. Defaults to 0.2.
            random_state (int): Seed for reproducibility. Defaults to 42.
            stratify (bool): Whether to preserve target class ratio in both splits.
                Defaults to True - important for imbalanced classification problems.
        """
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

    def split(self, X: pd.DataFrame, y: pd.Series):
        """
        Split features and target into train/test sets.

        Args:
            X (pd.DataFrame): Feature columns.
            y (pd.Series): Target column.

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        stratify_param = y if self.stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )

        logger.info(f"Split complete. Train: {X_train.shape}, Test: {X_test.shape}")
        logger.info(f"Train target ratio: {y_train.mean():.3f}, Test target ratio: {y_test.mean():.3f}")

        return X_train, X_test, y_train, y_test