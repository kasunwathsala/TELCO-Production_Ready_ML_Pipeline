"""
outlier_detection.py
Detects outliers in numerical columns using IQR and Z-score methods.
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutlierDetector:
    """Detects outliers in numerical columns using IQR or Z-score methods."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def detect_iqr(self, column: str, multiplier: float = 1.5) -> dict:
        """
        Detect outliers in a column using the IQR method.

        Args:
            column (str): Column to check for outliers.
            multiplier (float): IQR multiplier for bounds. Defaults to 1.5 (standard).

        Returns:
            dict: Contains outlier count, lower bound, and upper bound.
        """
        if column not in self.df.columns:
            raise KeyError(f"Column '{column}' not found.")

        q1 = self.df[column].quantile(0.25)
        q3 = self.df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]

        result = {
            'column': column,
            'outlier_count': len(outliers),
            'lower_bound': round(lower_bound, 2),
            'upper_bound': round(upper_bound, 2)
        }
        logger.info(f"IQR method - {column}: {result['outlier_count']} outliers found "
                    f"(valid range: {result['lower_bound']} to {result['upper_bound']})")
        return result

    def detect_zscore(self, column: str, threshold: float = 3.0) -> dict:
        """
        Detect outliers in a column using the Z-score method.

        Args:
            column (str): Column to check for outliers.
            threshold (float): Z-score threshold. Defaults to 3.0 (standard).

        Returns:
            dict: Contains outlier count and threshold used.
        """
        if column not in self.df.columns:
            raise KeyError(f"Column '{column}' not found.")

        z_scores = np.abs(stats.zscore(self.df[column]))
        outlier_count = int((z_scores > threshold).sum())

        result = {'column': column, 'outlier_count': outlier_count, 'threshold': threshold}
        logger.info(f"Z-score method - {column}: {outlier_count} outliers found (|Z| > {threshold})")
        return result

    def detect_all(self, columns: list, method: str = 'iqr') -> list:
        """Run outlier detection across multiple columns using the specified method."""
        results = []
        for col in columns:
            if method == 'iqr':
                results.append(self.detect_iqr(col))
            elif method == 'zscore':
                results.append(self.detect_zscore(col))
            else:
                raise ValueError("method must be 'iqr' or 'zscore'")
        return results