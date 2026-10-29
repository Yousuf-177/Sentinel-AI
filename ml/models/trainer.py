"""
SENTINEL-AI — Model Trainer

Handles training, evaluation, and artifact saving for the CNN-LSTM model.
"""

import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
# pyrefly: ignore [missing-import]
from tensorflow.keras.utils import to_categorical

from ml.models.cnn_lstm import build_cnn_lstm_model, get_training_callbacks

logger = logging.getLogger(__name__)


class Trainer:
    """
    Manages model training, evaluation, and artifact persistence.
    
    Usage:
        trainer = Trainer(num_features=42, num_classes=6)
        trainer.train(X_train, y_train, X_val, y_val)
        metrics = trainer.evaluate(X_test, y_test)
        trainer.save_model("ml/models/artifacts/sentinel_cnn_lstm.h5")
    """

    def __init__(self, num_features: int, num_classes: int = 6, config_path: str = "ml/config/model_config.yaml"):
        self.num_features = num_features
        self.num_classes = num_classes
        self.config_path = config_path
        self.model = None
        self.history = None
        self.class_names = ["Benign", "DoS/DDoS", "PortScan", "BruteForce", "WebAttack", "Infiltration"]

    def build(self):
        """Build the CNN-LSTM model."""
        self.model = build_cnn_lstm_model(
            input_shape=(self.num_features,),
            num_classes=self.num_classes,
            config_path=self.config_path,
        )
        return self.model

    def compute_class_weights(self, y_train: np.ndarray) -> dict:
        """
        Compute class weights to handle imbalanced data.
        """
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, weights))
        logger.info(f"Class weights: {class_weight_dict}")
        return class_weight_dict

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 256,
        use_class_weights: bool = True,
    ):
        """
        Train the model with optional class weighting.
        """
        if self.model is None:
            self.build()

        # Convert to one-hot
        y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
        y_val_cat = to_categorical(y_val, num_classes=self.num_classes)

        # Class weights
        class_weights = self.compute_class_weights(y_train) if use_class_weights else None

        # Callbacks
        cbs = get_training_callbacks(config_path=self.config_path)

        logger.info(f"Training for {epochs} epochs, batch_size={batch_size}")
        self.history = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=cbs,
            verbose=1,
        )

        return self.history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the model on the test set.
        
        Returns:
            dict: Comprehensive metrics including per-class precision/recall/F1.
        """
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
            "macro_precision": precision_score(y_test, y_pred, average="macro"),
            "macro_recall": recall_score(y_test, y_pred, average="macro"),
            "classification_report": classification_report(
                y_test, y_pred, target_names=self.class_names, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        # Log summary
        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"  Accuracy:        {metrics['accuracy']:.4f}")
        logger.info(f"  Macro F1:        {metrics['macro_f1']:.4f}")
        logger.info(f"  Weighted F1:     {metrics['weighted_f1']:.4f}")
        logger.info(f"  Macro Precision: {metrics['macro_precision']:.4f}")
        logger.info(f"  Macro Recall:    {metrics['macro_recall']:.4f}")
        logger.info("-" * 60)
        logger.info(classification_report(y_test, y_pred, target_names=self.class_names))
        logger.info("=" * 60)

        return metrics

    def save_model(self, path: str):
        """Save the trained model to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}")

    def load_model(self, path: str):
        """Load a trained model from disk."""
        from tensorflow import keras
        self.model = keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")
