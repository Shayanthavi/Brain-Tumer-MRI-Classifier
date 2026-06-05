"""
Hyperparameter Tuning with Optuna
=================================
Script: src/tune.py

Tunes hyperparameters such as learning rate, dense units, and dropout rates
using Optuna. 
"""

import os
import argparse
import optuna
import tensorflow as tf
from tensorflow import keras

# Import project specific modules
from src.models import get_model, MODEL_BUILDERS
from src.train import load_datasets, _compute_class_weights, _make_callbacks

# Silence some TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def objective(trial, model_name):
    # 1. Sample hyperparameters
    dense1_units = trial.suggest_categorical("dense1_units", [256, 512, 1024])
    dense2_units = trial.suggest_categorical("dense2_units", [128, 256, 512])
    dropout1 = trial.suggest_float("dropout1", 0.2, 0.6, step=0.1)
    dropout2 = trial.suggest_float("dropout2", 0.2, 0.6, step=0.1)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    
    # 2. Load Data
    train_ds, val_ds, test_ds = load_datasets(model_name)
    class_weights = _compute_class_weights(train_ds)

    # 3. Build Model with sampled hyperparameters
    model = get_model(
        model_name, 
        fine_tune=False,
        dense1_units=dense1_units,
        dense2_units=dense2_units,
        dropout1=dropout1,
        dropout2=dropout2
    )

    # 4. Compile Model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    # Optuna Pruning Callback
    try:
        from optuna.integration import TFKerasPruningCallback
        callbacks = [TFKerasPruningCallback(trial, "val_accuracy")]
    except ImportError:
        callbacks = []
    
    # Add early stopping to prevent wasting time on bad trials
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    )

    # 5. Train Model (limit epochs for tuning speed)
    epochs = 10 if model_name != "custom_cnn" else 20
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # 6. Return best validation accuracy
    val_acc = max(history.history["val_accuracy"])
    return val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning")
    parser.add_argument(
        "--model",
        type=str,
        default="custom_cnn",
        choices=list(MODEL_BUILDERS.keys()),
        help="Model to tune (default: custom_cnn)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=10,
        help="Number of trials to run",
    )
    args = parser.parse_args()

    print(f"\n[INFO] Starting Optuna tuning for {args.model}...")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args.model), n_trials=args.n_trials)

    print("\n[INFO] Tuning complete!")
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best val_accuracy: {study.best_trial.value:.4f}")
    print("  Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
