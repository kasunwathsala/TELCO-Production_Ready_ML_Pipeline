"""
data_ingestion.py
Handles loading of raw customer data from CSV files.
"""

import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestion:
    """Handles loading and initial validation of the Telco churn dataset."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """
        Load data from the CSV file.

        Returns:
            pd.DataFrame: The loaded dataset.

        Raises:
            FileNotFoundError: If the file doesn't exist at the given path.
        """
        try:
            df = pd.read_csv(self.file_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found at: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise