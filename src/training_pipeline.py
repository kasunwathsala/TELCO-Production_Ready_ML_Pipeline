"""
training_pipeline.py
Orchestrates the complete end-to-end training pipeline:
data ingestion -> cleaning -> feature engineering -> encoding ->
scaling -> splitting -> model training -> evaluation.
"""

import logging
from src.data_processing.data_ingestion import DataIngestion
from src.data_processing.handle_missing_values import MissingValueHandler
from src.data_processing.feature_encoding import FeatureEncoder
from src.data_processing.feature_scaling import FeatureScaler
from src.data_processing.data_splitter import DataSplitter
from src.modeling.model_building import ModelFactory
from src.modeling.model_training import ModelTrainer
from src.modeling.model_evaluation import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """End-to-end pipeline: raw data in, evaluated trained model out."""

    def __init__(self, config: dict):
        """
        Args:
            config (dict): Pipeline configuration. Expected keys:
                - file_path (str): Path to the raw CSV.
                - binary_columns (list): Columns for label encoding.
                - onehot_columns (list): Columns for one-hot encoding.
                - numeric_columns (list): Columns to scale.
                - target_column (str): Name of the target column.
                - model_name (str): Model to train (e.g. 'xgboost').
                - model_params (dict): Hyperparameters for the model.
        """
        self.config = config
        self.trained_pipeline = None
        self.evaluation_report = None

    def run(self) -> dict:
        """
        Execute the full pipeline end-to-end.

        Returns:
            dict: The evaluation report for the trained model.
        """
        logger.info("=== Starting Training Pipeline ===")

        # 1. Data Ingestion
        ingestion = DataIngestion(self.config['file_path'])
        df = ingestion.load_data()

        # 2. Handle Missing Values
        missing_handler = MissingValueHandler(df)
        df = missing_handler.fix_total_charges()

        # 3. Encoding
        encoder = FeatureEncoder(df)
        df = encoder.label_encode(self.config['binary_columns'])
        df = encoder.one_hot_encode(self.config['onehot_columns'])

        # 4. Split features/target
        target_col = self.config['target_column']
        X = df.drop(columns=[target_col, 'customerID'], errors='ignore')
        y = df[target_col]

        # 5. Train/Test Split
        splitter = DataSplitter(test_size=0.2, random_state=42, stratify=True)
        X_train, X_test, y_train, y_test = splitter.split(X, y)

        # 6. Scaling (fit on train, transform test - no leakage)
        numeric_cols = self.config['numeric_columns']
        scaler = FeatureScaler(X_train, method='standard')
        X_train = scaler.fit_transform(numeric_cols)
        scaler.df = X_test.copy()
        X_test = scaler.transform(numeric_cols)

        # 7. Model Building
        model = ModelFactory.create_model(
            self.config['model_name'],
            **self.config.get('model_params', {})
        )

        # 8. Training (with cross-validation on train set)
        trainer = ModelTrainer(model, scale_features=False)  # already scaled above
        cv_results = trainer.train_with_cross_validation(X_train, y_train)
        self.trained_pipeline = trainer.pipeline

        # 9. Evaluation on held-out test set
        evaluator = ModelEvaluator(self.trained_pipeline, X_test, y_test)
        self.evaluation_report = evaluator.full_report(
            fn_cost=self.config.get('fn_cost'),
            fp_cost=self.config.get('fp_cost')
        )
        self.evaluation_report['cross_validation'] = cv_results

        logger.info("=== Training Pipeline Complete ===")
        return self.evaluation_report