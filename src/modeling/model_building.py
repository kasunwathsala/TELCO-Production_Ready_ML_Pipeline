"""
model_building.py
Model factory for creating different classification models
(baselines and ensemble methods) with a consistent interface.
"""

import logging
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory class for building classification models by name."""

    # Registry of available models - add new models here only
    _MODEL_REGISTRY = {
        'logistic_regression': LogisticRegression,
        'decision_tree': DecisionTreeClassifier,
        'random_forest': RandomForestClassifier,
        'xgboost': XGBClassifier,
        'catboost': CatBoostClassifier,
    }

    @staticmethod
    def create_model(model_name: str, random_state: int = 42, **kwargs):
        """
        Create a model instance by name.

        Args:
            model_name (str): One of 'logistic_regression', 'decision_tree',
                'random_forest', 'xgboost', 'catboost'.
            random_state (int): Seed for reproducibility. Defaults to 42.
            **kwargs: Additional hyperparameters passed to the model constructor
                (e.g. n_estimators=100, max_depth=10).

        Returns:
            A scikit-learn-compatible classifier instance.

        Raises:
            ValueError: If model_name is not recognized.
        """
        model_name = model_name.lower()

        if model_name not in ModelFactory._MODEL_REGISTRY:
            available = list(ModelFactory._MODEL_REGISTRY.keys())
            raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")

        model_class = ModelFactory._MODEL_REGISTRY[model_name]

        # CatBoost uses 'verbose' instead of relying on sklearn defaults
        if model_name == 'catboost':
            kwargs.setdefault('verbose', 0)

        model = model_class(random_state=random_state, **kwargs)
        logger.info(f"Created '{model_name}' model with params: {kwargs}")
        return model

    @staticmethod
    def get_available_models() -> list:
        """Return the list of model names supported by the factory."""
        return list(ModelFactory._MODEL_REGISTRY.keys())