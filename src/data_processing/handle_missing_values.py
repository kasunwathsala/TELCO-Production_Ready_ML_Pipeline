"""
handle_missing_values.py
Detects and handles missing values in the dataset, including
hidden missing values disguised as empty strings.
"""

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingValueHandler:
    """Detects and fixes missing values, including hidden ones (empty strings)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def fix_total_charges(self, column: str = 'TotalCharges') -> pd.DataFrame:
        """
        Convert a numeric-looking text column to actual numbers,
        filling any resulting missing values with 0.

        New customers (tenure=0) often have empty TotalCharges since
        they haven't been billed yet - filling with 0 reflects that
        business logic rather than dropping the rows.

        Args:
            column (str): Name of the column to fix. Defaults to 'TotalCharges'.

        Returns:
            pd.DataFrame: The dataframe with the column cleaned.
        """
        if column not in self.df.columns:
            logger.error(f"Column '{column}' not found in dataframe.")
            raise KeyError(f"Column '{column}' not found.")

        # Convert to numeric; invalid parsing (e.g. empty strings) becomes NaN
        self.df[column] = pd.to_numeric(self.df[column], errors='coerce')

        missing_count = self.df[column].isnull().sum()
        if missing_count > 0:
            logger.info(f"Found {missing_count} hidden missing values in '{column}'. Filling with 0.")
            self.df[column] = self.df[column].fillna(0)
        else:
            logger.info(f"No missing values found in '{column}'.")

        return self.df

    def check_missing_summary(self) -> pd.Series:
        """Return a summary of missing values across all columns."""
        summary = self.df.isnull().sum()
        return summary[summary > 0]