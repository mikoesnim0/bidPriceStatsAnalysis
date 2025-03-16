"""
Model evaluation and visualization module for BidPrice prediction.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, median_absolute_error
)

# Try importing AutoGluon, but handle if not installed
try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False

from src.config import (
    TEST_DATA_FILE, MODELS_DIR, RESULTS_DIR, TARGET_COLUMN
)
from src.data_processing import load_data
from src.utils import setup_logger, save_json, timer, generate_timestamp

logger = setup_logger(__name__)

def load_model(model_path: str) -> Any:
    """
    Load a trained model.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {model_path}")
    
    # Check if it's an AutoGluon model
    if os.path.exists(os.path.join(model_path, "predictor.pkl")):
        if not AUTOGLUON_AVAILABLE:
            raise ImportError("AutoGluon is not installed but required to load this model")
        return TabularPredictor.load(model_path)
    
    # Otherwise assume it's a pickle file
    import pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

def predict(
    model: Any,
    X: pd.DataFrame,
    model_type: str = "autogluon"
) -> np.ndarray:
    """
    Generate predictions from model.
    
    Args:
        model: Trained model
        X: Features
        model_type: Type of model ("lgbm" or "autogluon")
        
    Returns:
        Predicted values
    """
    logger.info(f"Generating predictions with {model_type} model")
    
    if model_type == "autogluon":
        if not AUTOGLUON_AVAILABLE:
            raise ImportError("AutoGluon is not installed")
        return model.predict(X).values
    elif model_type == "lgbm":
        return model.predict(X)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def calculate_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    logger.info("Calculating evaluation metrics")
    
    # Make sure inputs are numpy arrays
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    
    # Calculate metrics
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred))
    }
    
    logger.info(f"Metrics: {metrics}")
    return metrics

def plot_actual_vs_predicted(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot actual vs predicted values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save the plot
    """
    logger.info("Generating actual vs predicted plot")
    
    # Make sure inputs are numpy arrays
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.grid(True)
    
    # Add metrics to the plot
    metrics = calculate_metrics(y_true, y_pred)
    plt.annotate(
        f"RMSE: {metrics['rmse']:.4f}\n"
        f"MAE: {metrics['mae']:.4f}\n"
        f"R²: {metrics['r2']:.4f}",
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
    )
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    plt.close()

def plot_residuals(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot residuals.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save the plot
    """
    logger.info("Generating residuals plot")
    
    # Make sure inputs are numpy arrays
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 8))
    
    # Residuals vs predicted
    plt.subplot(2, 1, 1)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    plt.grid(True)
    
    # Residuals distribution
    plt.subplot(2, 1, 2)
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.title('Residuals Distribution')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    plt.close()

def evaluate_model(
    model: Any,
    test_data: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    model_type: str = "autogluon"
) -> Dict[str, Any]:
    """
    Evaluate model on test data.
    
    Args:
        model: Trained model
        test_data: Test dataset
        target_col: Target column name
        model_type: Type of model ("lgbm" or "autogluon")
        
    Returns:
        Evaluation results
    """
    logger.info(f"Evaluating {model_type} model")
    
    # Separate features and target
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]
    
    # Generate predictions
    y_pred = predict(model, X_test, model_type)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # Generate timestamp for results
    timestamp = generate_timestamp()
    
    # Save results
    results_dir = os.path.join(RESULTS_DIR, f"evaluation_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics as JSON
    save_json(metrics, os.path.join(results_dir, "metrics.json"))
    
    # Create visualizations
    plot_actual_vs_predicted(
        y_test, y_pred,
        save_path=os.path.join(results_dir, "actual_vs_predicted.png")
    )
    
    plot_residuals(
        y_test, y_pred,
        save_path=os.path.join(results_dir, "residuals.png")
    )
    
    return {
        "metrics": metrics,
        "results_dir": results_dir,
        "timestamp": timestamp
    }

def main():
    """
    Main function to run the evaluation pipeline.
    """
    start_time = timer()
    logger.info("Starting model evaluation pipeline")
    
    # Load test data
    test_data = load_data(TEST_DATA_FILE)
    
    # Find the latest model
    model_dirs = [d for d in os.listdir(MODELS_DIR) 
                 if os.path.isdir(os.path.join(MODELS_DIR, d))]
    
    if not model_dirs:
        logger.error("No models found in models directory")
        return
    
    # Sort by timestamp
    latest_model_dir = sorted(model_dirs)[-1]
    model_path = os.path.join(MODELS_DIR, latest_model_dir)
    
    # Load model info
    with open(os.path.join(model_path, "model_info.json"), 'r') as f:
        model_info = json.load(f)
    
    model_type = model_info.get("model_type", "autogluon")
    
    # Load model
    model = load_model(model_path)
    
    # Evaluate model
    evaluation_results = evaluate_model(
        model, test_data, TARGET_COLUMN, model_type
    )
    
    logger.info(f"Evaluation completed in {timer(start_time)}")
    logger.info(f"Results saved to {evaluation_results['results_dir']}")

if __name__ == "__main__":
    main() 