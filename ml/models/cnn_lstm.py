"""
SENTINEL-AI — CNN-LSTM Model Architecture

Defines the hybrid 1D-CNN + LSTM model for network intrusion classification.
"""

import logging
from pathlib import Path

import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

logger = logging.getLogger(__name__)


def build_cnn_lstm_model(
    input_shape: tuple,
    num_classes: int = 6,
    config_path: str = "ml/config/model_config.yaml",
) -> keras.Model:
    """
    Build the CNN-LSTM hybrid model for network intrusion detection.
    
    Architecture:
    1. Input layer (feature vector)
    2. Reshape for 1D-CNN (treats features as a 1D sequence)
    3. 1D-CNN blocks: extract spatial/correlational patterns from features
    4. LSTM blocks: capture temporal/sequential patterns
    5. Dense classifier head: multi-class softmax output
    
    Args:
        input_shape: Shape of input data (num_features,)
        num_classes: Number of classification classes (default: 6)
        config_path: Path to model configuration YAML
    
    Returns:
        Compiled Keras model
    """
    # Load config
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        arch = config["model"]["architecture"]
        train_cfg = config["training"]
    else:
        logger.warning(f"Config not found at {config_path}, using defaults")
        arch = {
            "cnn_filters": [64, 128],
            "cnn_kernel_size": 3,
            "cnn_activation": "relu",
            "cnn_pool_size": 2,
            "cnn_dropout": 0.3,
            "lstm_units": [128, 64],
            "lstm_dropout": 0.3,
            "lstm_recurrent_dropout": 0.2,
            "dense_units": [128, 64],
            "dense_dropout": 0.4,
            "dense_activation": "relu",
            "output_activation": "softmax",
        }
        train_cfg = {"learning_rate": 0.001, "loss": "categorical_crossentropy"}

    # Build model
    inputs = keras.Input(shape=input_shape, name="flow_features")

    # Reshape to (features, 1) for 1D-CNN
    x = layers.Reshape((input_shape[0], 1))(inputs)

    # === 1D-CNN Block ===
    # Extracts spatial patterns (feature correlations)
    for i, filters in enumerate(arch["cnn_filters"]):
        x = layers.Conv1D(
            filters=filters,
            kernel_size=arch["cnn_kernel_size"],
            activation=arch["cnn_activation"],
            padding="same",
            name=f"conv1d_{i+1}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_cnn_{i+1}")(x)
        if i < len(arch["cnn_filters"]) - 1:
            x = layers.MaxPooling1D(pool_size=arch["cnn_pool_size"], name=f"maxpool_{i+1}")(x)
        x = layers.Dropout(arch["cnn_dropout"], name=f"dropout_cnn_{i+1}")(x)

    # === LSTM Block ===
    # Captures temporal/sequential patterns
    for i, units in enumerate(arch["lstm_units"]):
        return_sequences = i < len(arch["lstm_units"]) - 1
        x = layers.LSTM(
            units=units,
            return_sequences=return_sequences,
            dropout=arch["lstm_dropout"],
            recurrent_dropout=arch["lstm_recurrent_dropout"],
            name=f"lstm_{i+1}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_lstm_{i+1}")(x)

    # === Dense Classifier Head ===
    for i, units in enumerate(arch["dense_units"]):
        x = layers.Dense(units, activation=arch["dense_activation"], name=f"dense_{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn_dense_{i+1}")(x)
        x = layers.Dropout(arch["dense_dropout"], name=f"dropout_dense_{i+1}")(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation=arch["output_activation"], name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="sentinel_cnn_lstm")

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=train_cfg["learning_rate"]),
        loss=train_cfg["loss"],
        metrics=["accuracy"],
    )

    logger.info(f"Model built: {model.count_params():,} parameters")
    model.summary(print_fn=logger.info)

    return model


def get_training_callbacks(
    log_dir: str = "ml/logs/",
    config_path: str = "ml/config/model_config.yaml",
) -> list:
    """
    Get standard training callbacks (EarlyStopping, ReduceLR, TensorBoard, ModelCheckpoint).
    """
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        es_cfg = config["training"]["early_stopping"]
        lr_cfg = config["training"]["reduce_lr"]
    else:
        es_cfg = {"monitor": "val_loss", "patience": 7, "restore_best_weights": True}
        lr_cfg = {"monitor": "val_loss", "factor": 0.5, "patience": 3, "min_lr": 1e-5}

    return [
        callbacks.EarlyStopping(
            monitor=es_cfg["monitor"],
            patience=es_cfg["patience"],
            restore_best_weights=es_cfg["restore_best_weights"],
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor=lr_cfg["monitor"],
            factor=lr_cfg["factor"],
            patience=lr_cfg["patience"],
            min_lr=lr_cfg["min_lr"],
            verbose=1,
        ),
        callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        callbacks.ModelCheckpoint(
            filepath="ml/models/checkpoints/best_model.h5",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]
