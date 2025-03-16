"""
Model training module for BidPrice prediction.
"""
import os
import pickle
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import cross_val_score, KFold
import lightgbm as lgbm
from datetime import datetime

# Try importing AutoGluon, but handle if not installed
try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False

from src.config import (
    TRAIN_DATA_FILE, MODELS_DIR, RESULTS_DIR, MODEL_PARAMS,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TEXT_FEATURES,
    TARGET_COLUMN, RANDOM_SEED, CV_FOLDS
)
from src.data_processing import load_data, prepare_features
from src.utils import setup_logger, save_pickle, save_json, timer, generate_timestamp

logger = setup_logger(__name__)

def train_lgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: Optional[pd.DataFrame] = None,
    y_valid: Optional[pd.Series] = None,
    params: Dict[str, Any] = None
) -> Tuple[lgbm.Booster, Dict[str, Any]]:
    """
    Train a LightGBM model.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_valid: Validation features
        y_valid: Validation target
        params: Model parameters
        
    Returns:
        Trained model and training results
    """
    logger.info("Training LightGBM model")
    
    # Get default parameters if not provided
    if params is None:
        params = MODEL_PARAMS["lgbm"]
    
    start_time = timer()
    
    # Create LightGBM datasets
    lgb_train = lgbm.Dataset(X_train, y_train)
    
    if X_valid is not None and y_valid is not None:
        lgb_valid = lgbm.Dataset(X_valid, y_valid, reference=lgb_train)
        valid_sets = [lgb_train, lgb_valid]
        valid_names = ["train", "valid"]
        logger.info("Using provided validation set")
    else:
        valid_sets = [lgb_train]
        valid_names = ["train"]
        logger.info("No validation set provided, using training set only")
    
    # Train model
    model = lgbm.train(
        params,
        lgb_train,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=[
            lgbm.log_evaluation(period=100)
        ]
    )
    
    # Collect training results
    results = {
        "train_time": timer(start_time),
        "best_iteration": model.best_iteration,
        "feature_importance": {
            name: score for name, score in zip(
                model.feature_name(),
                model.feature_importance("gain")
            )
        }
    }
    
    if X_valid is not None and y_valid is not None:
        results["valid_metric"] = model.best_score["valid"][params["metric"]]
    
    logger.info(f"LightGBM training completed in {results['train_time']}")
    logger.info(f"Best iteration: {results['best_iteration']}")
    
    return model, results

def train_autogluon_model(
    train_data: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    params: Dict[str, Any] = None
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    """
    Train an AutoGluon model.
    
    Args:
        train_data: Training data including target
        target_col: Target column name
        params: Model parameters
        
    Returns:
        Trained predictor and training results
    """
    if not AUTOGLUON_AVAILABLE:
        error_msg = "AutoGluon is not installed. Install with: pip install autogluon.tabular"
        logger.error(error_msg)
        raise ImportError(error_msg)
    
    logger.info("Training AutoGluon model")
    
    # Get default parameters if not provided
    if params is None:
        params = MODEL_PARAMS["autogluon"]
    
    start_time = timer()
    
    # Create output directory for AutoGluon
    timestamp = generate_timestamp()
    save_path = os.path.join(MODELS_DIR, f"autogluon_{timestamp}")
    
    # Train model
    predictor = TabularPredictor(
        label=target_col,
        path=save_path
    )
    
    predictor.fit(
        train_data=train_data,
        time_limit=params["time_limit"],
        presets=params["presets"],
        num_stack_levels=params["num_stack_levels"],
        num_bag_folds=params["num_bag_folds"]
    )
    
    # Collect training results
    results = {
        "train_time": timer(start_time),
        "leaderboard": predictor.leaderboard().to_dict(),
        "feature_importance": predictor.feature_importance(train_data).to_dict() if hasattr(predictor, "feature_importance") else {}
    }
    
    logger.info(f"AutoGluon training completed in {results['train_time']}")
    
    return predictor, results

def evaluate_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "lgbm",
    params: Dict[str, Any] = None,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_SEED
) -> Dict[str, Any]:
    """
    Evaluate model using cross-validation.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        model_type: Type of model to evaluate ("lgbm" or "autogluon")
        params: Model parameters
        cv_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility
        
    Returns:
        Cross-validation results
    """
    logger.info(f"Evaluating {model_type} model with {cv_folds}-fold cross-validation")
    
    start_time = timer()
    
    if model_type == "lgbm":
        # Use default params if not provided
        if params is None:
            params = MODEL_PARAMS["lgbm"]
        
        # Define cross-validation strategy
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        # Initialize lists to store metrics
        fold_scores = []
        fold_models = []
        
        # Perform cross-validation
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
            logger.info(f"Training fold {fold+1}/{cv_folds}")
            
            # Split data for this fold
            X_fold_train = X.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]
            X_fold_valid = X.iloc[valid_idx]
            y_fold_valid = y.iloc[valid_idx]
            
            # Train model
            model, _ = train_lgbm_model(
                X_fold_train, y_fold_train,
                X_fold_valid, y_fold_valid,
                params
            )
            
            # Make predictions on validation set
            y_pred = model.predict(X_fold_valid)
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((y_fold_valid - y_pred) ** 2))
            fold_scores.append(rmse)
            fold_models.append(model)
            
            logger.info(f"Fold {fold+1} RMSE: {rmse:.6f}")
        
        # Calculate average score
        avg_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        logger.info(f"Cross-validation RMSE: {avg_score:.6f} ± {std_score:.6f}")
        
        # Return results
        results = {
            "model_type": model_type,
            "cv_folds": cv_folds,
            "cv_time": timer(start_time),
            "fold_scores": fold_scores,
            "mean_score": avg_score,
            "std_score": std_score
        }
        
        return results
    
    elif model_type == "autogluon":
        if not AUTOGLUON_AVAILABLE:
            error_msg = "AutoGluon is not installed. Install with: pip install autogluon.tabular"
            logger.error(error_msg)
            raise ImportError(error_msg)
        
        # Use default params if not provided
        if params is None:
            params = MODEL_PARAMS["autogluon"]
        
        # For AutoGluon, we'll use its built-in k-fold capabilities
        full_data = X.copy()
        full_data[TARGET_COLUMN] = y
        
        timestamp = generate_timestamp()
        save_path = os.path.join(MODELS_DIR, f"autogluon_cv_{timestamp}")
        
        predictor = TabularPredictor(
            label=TARGET_COLUMN,
            path=save_path
        )
        
        # Perform k-fold validation
        scores = predictor.fit_cv(
            train_data=full_data,
            k_fold=cv_folds,
            random_state=random_state,
            time_limit=params["time_limit"],
            presets=params["presets"],
            hyperparameters=None  # Use default hyperparameters
        )
        
        logger.info(f"AutoGluon CV scores: {scores}")
        
        # Return results
        results = {
            "model_type": model_type,
            "cv_folds": cv_folds,
            "cv_time": timer(start_time),
            "scores": scores
        }
        
        return results
    
    else:
        error_msg = f"Unknown model type: {model_type}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def save_model(
    model: Any,
    model_type: str,
    results: Dict[str, Any]
) -> str:
    """
    Save model and its results.
    
    Args:
        model: Trained model
        model_type: Type of model ("lgbm" or "autogluon")
        results: Model results
        
    Returns:
        Path to the saved model directory
    """
    logger.info(f"Saving {model_type} model and results")
    
    # Create timestamp for model version
    timestamp = generate_timestamp()
    model_dir = os.path.join(MODELS_DIR, f"{model_type}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, f"{model_type}_model.pkl")
    
    if model_type == "lgbm":
        # LightGBM models can be saved directly with pickle
        save_pickle(model, model_path)
    elif model_type == "autogluon":
        # AutoGluon models should already be saved to disk during training
        # Just save a reference to the model path
        results["model_path"] = model.path
    else:
        error_msg = f"Unknown model type: {model_type}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Save results
    results_path = os.path.join(model_dir, f"{model_type}_results.json")
    save_json(results, results_path)
    
    logger.info(f"Model and results saved to {model_dir}")
    
    return model_dir

def train_single_target_model(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    target_col: str,
    use_gpu: bool = True,
    selected_models: Optional[List[str]] = None,
    preset: str = "medium_quality_faster_train"
) -> str:
    """
    Train a model for a single target column using AutoGluon.
    
    Args:
        X_train: Training features DataFrame
        Y_train: Training targets DataFrame
        target_col: Target column name to train for
        use_gpu: Whether to use GPU for training
        selected_models: List of specific models to use, or None for all
        preset: AutoGluon quality preset
    
    Returns:
        Path to the saved model
    """
    logger.info(f"Training model for target: {target_col}")
    
    # Extract the specific target column from Y_train
    if target_col in Y_train.columns:
        y_train = Y_train[target_col]
    else:
        raise ValueError(f"Target column '{target_col}' not found in training data")
    
    # Combine features and target into one DataFrame for AutoGluon
    train_data = X_train.copy()
    train_data[target_col] = y_train
    
    # Set up AutoGluon parameters
    params = MODEL_PARAMS["autogluon"].copy()
    
    # Update parameters based on function arguments
    params["presets"] = preset
    
    # Set up model output directory
    model_path = os.path.join(MODELS_DIR, target_col)
    os.makedirs(model_path, exist_ok=True)
    
    # Configure AutoGluon parameters
    ag_params = {
        "path": model_path,
        "label": target_col,
        "problem_type": "regression"
    }
    
    # Add GPU setting
    ag_args = {
        "num_gpus": 1 if use_gpu else 0
    }
    
    # Add model selection if provided
    if selected_models:
        ag_args["excluded_model_types"] = None  # Reset exclusions
        ag_args["included_model_types"] = selected_models
    
    # Train the model
    try:
        predictor = TabularPredictor(**ag_params)
        
        predictor.fit(
            train_data=train_data,
            time_limit=params["time_limit"],
            presets=params["presets"],
            num_stack_levels=params["num_stack_levels"],
            num_bag_folds=params["num_bag_folds"],
            **ag_args
        )
        
        # Save model info
        model_info = {
            "target_column": target_col,
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "preset": preset,
            "use_gpu": use_gpu,
            "selected_models": selected_models,
            "feature_count": X_train.shape[1],
            "sample_count": X_train.shape[0]
        }
        
        # Save model info as JSON
        with open(os.path.join(model_path, "model_info.json"), "w") as f:
            json.dump(model_info, f, indent=4)
        
        logger.info(f"Model for {target_col} trained and saved to {model_path}")
        
        return model_path
        
    except Exception as e:
        logger.error(f"Error training model for {target_col}: {str(e)}")
        raise

def main():
    """Main function to train models."""
    try:
        # Load training data
        logger.info("Loading training data")
        train_data = load_data(TRAIN_DATA_FILE)
        
        # Separate features and target
        X = train_data.drop(columns=[TARGET_COLUMN])
        y = train_data[TARGET_COLUMN]
        
        # Prepare features
        X_prepared, _ = prepare_features(X)
        
        # Cross-validation with LightGBM
        logger.info("Performing cross-validation with LightGBM")
        cv_results = evaluate_cv(X_prepared, y, model_type="lgbm")
        
        # Train final LightGBM model on all data
        logger.info("Training final LightGBM model on all data")
        final_model, train_results = train_lgbm_model(X_prepared, y)
        
        # Combine results
        final_results = {
            "training_results": train_results,
            "cv_results": cv_results
        }
        
        # Save model and results
        model_dir = save_model(final_model, "lgbm", final_results)
        logger.info(f"Model saved to {model_dir}")
        
        # Try training AutoGluon model if available
        if AUTOGLUON_AVAILABLE:
            logger.info("AutoGluon is available, training model")
            ag_model, ag_results = train_autogluon_model(train_data)
            save_model(ag_model, "autogluon", ag_results)
        else:
            logger.info("AutoGluon is not available, skipping")
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 