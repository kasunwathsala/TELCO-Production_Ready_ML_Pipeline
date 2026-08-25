"""
model_inference.py
Handles predictions on new (unseen) customer data using a trained model,
supporting both single-customer and batch predictions.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelInference:
    """Generates churn predictions for new customer data using a trained model."""

    def __init__(self, trained_model, expected_columns: list):
        """
        Args:
            trained_model: A fitted, scikit-learn-compatible model or pipeline.
            expected_columns (list): The exact feature columns (and order) the
                model was trained on - used to validate incoming data.
        """
        self.model = trained_model
        self.expected_columns = expected_columns

    def _validate_input(self, X: pd.DataFrame) -> None:
        """
        Check that incoming data has the expected columns before predicting.

        Raises:
            ValueError: If required columns are missing.
        """
        missing_cols = set(self.expected_columns) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Input data is missing required columns: {missing_cols}")

    def predict_single(self, customer_data: dict, threshold: float = 0.5) -> dict:
        """
        Predict churn for a single customer.

        Args:
            customer_data (dict): Feature values for one customer,
                e.g. {'tenure': 5, 'MonthlyCharges': 70.5, ...}.
            threshold (float): Probability cutoff for classifying as churn.
                Defaults to 0.5; use a tuned value (e.g. 0.32) for better recall.

        Returns:
            dict: churn_prediction (Yes/No), churn_probability, threshold_used.
        """
        X = pd.DataFrame([customer_data])
        self._validate_input(X)
        X = X[self.expected_columns]  # ensure correct column order

        probability = self.model.predict_proba(X)[0, 1]
        prediction = 'Yes' if probability >= threshold else 'No'

        result = {
            'churn_prediction': prediction,
            'churn_probability': round(float(probability), 4),
            'threshold_used': threshold
        }
        logger.info(f"Single prediction: {result}")
        return result

    def predict_batch(self, X: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        Predict churn for multiple customers at once.

        Args:
            X (pd.DataFrame): Feature data for multiple customers.
            threshold (float): Probability cutoff for classifying as churn.

        Returns:
            pd.DataFrame: Original data with added 'churn_probability' and
                'churn_prediction' columns.
        """
        self._validate_input(X)
        X_ordered = X[self.expected_columns]

        probabilities = self.model.predict_proba(X_ordered)[:, 1]
        predictions = np.where(probabilities >= threshold, 'Yes', 'No')

        result_df = X.copy()
        result_df['churn_probability'] = probabilities.round(4)
        result_df['churn_prediction'] = predictions

        logger.info(f"Batch prediction complete for {len(X)} customers. "
                    f"Predicted churn: {(predictions == 'Yes').sum()}")
        return result_df