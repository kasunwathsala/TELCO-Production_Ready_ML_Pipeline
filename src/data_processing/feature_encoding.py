"""
feature_encoding.py
Encodes categorical columns using Label Encoding (binary columns)
and One-Hot Encoding (multi-category columns).
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEncoder:
    """Encodes categorical features using appropriate strategies per column type."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.label_encoders = {}  # keep encoders in case we need to inverse_transform later

    def label_encode(self, columns: list) -> pd.DataFrame:
        """
        Apply Label Encoding to binary/ordinal columns.

        Args:
            columns (list): List of column names to label encode.

        Returns:
            pd.DataFrame: Dataframe with the specified columns encoded.
        """
        for col in columns:
            if col not in self.df.columns:
                logger.warning(f"Column '{col}' not found - skipping.")
                continue

            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le  # save for potential later use

        logger.info(f"Label encoded {len(columns)} columns: {columns}")
        return self.df

    def one_hot_encode(self, columns: list, drop_first: bool = True) -> pd.DataFrame:
        """
        Apply One-Hot Encoding to multi-category, unordered columns.

        Args:
            columns (list): List of column names to one-hot encode.
            drop_first (bool): Whether to drop the first category to avoid
                the dummy variable trap. Defaults to True.

        Returns:
            pd.DataFrame: Dataframe with the specified columns one-hot encoded.
        """
        valid_columns = [c for c in columns if c in self.df.columns]
        missing = set(columns) - set(valid_columns)
        if missing:
            logger.warning(f"Columns not found - skipping: {missing}")

        shape_before = self.df.shape
        self.df = pd.get_dummies(self.df, columns=valid_columns, drop_first=drop_first)
        logger.info(f"One-hot encoded {len(valid_columns)} columns. "
                    f"Shape: {shape_before} -> {self.df.shape}")
        return self.df