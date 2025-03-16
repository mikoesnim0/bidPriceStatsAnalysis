"""
Prediction module for BidPrice prediction.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

# Try importing AutoGluon, but handle if not installed
try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False

from src.config import (
    MODELS_DIR, TARGET_COLUMN
)
from src.utils import setup_logger, timer
from src.evaluate import load_model

logger = setup_logger(__name__)

def get_latest_model_path() -> str:
    """
    Get the path to the latest trained model.
    
    Returns:
        Path to the latest model directory
    """
    logger.info("Finding latest model")
    
    # Find all model directories
    model_dirs = [d for d in os.listdir(MODELS_DIR) 
                 if os.path.isdir(os.path.join(MODELS_DIR, d))]
    
    if not model_dirs:
        raise FileNotFoundError("No models found in models directory")
    
    # Sort by timestamp
    latest_model_dir = sorted(model_dirs)[-1]
    model_path = os.path.join(MODELS_DIR, latest_model_dir)
    
    logger.info(f"Latest model found: {latest_model_dir}")
    return model_path

def prepare_prediction_data(
    df: pd.DataFrame,
    required_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Prepare data for prediction.
    
    Args:
        df: Input DataFrame
        required_features: List of required feature columns
        
    Returns:
        Prepared DataFrame
    """
    logger.info("Preparing data for prediction")
    
    # Make a copy of the input DataFrame
    df_copy = df.copy()
    
    if required_features:
        # Check if all required features are present
        missing_features = [feat for feat in required_features if feat not in df_copy.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Keep only required features
        df_copy = df_copy[required_features]
    
    # For AutoGluon, if target column exists, drop it
    if TARGET_COLUMN in df_copy.columns:
        df_copy = df_copy.drop(columns=[TARGET_COLUMN])
    
    # Log data info
    logger.info(f"Prediction data shape: {df_copy.shape}")
    
    return df_copy

def predict_with_model(
    model: Any,
    df: pd.DataFrame,
    model_type: str = "autogluon",
    required_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Make predictions with a trained model.
    
    Args:
        model: Trained model
        df: Input data
        model_type: Type of model ("lgbm" or "autogluon")
        required_features: List of required feature columns
        
    Returns:
        DataFrame with predictions
    """
    logger.info(f"Making predictions with {model_type} model")
    
    # Prepare data for prediction
    prediction_data = prepare_prediction_data(df, required_features)
    
    # Make predictions
    if model_type == "autogluon":
        if not AUTOGLUON_AVAILABLE:
            raise ImportError("AutoGluon is not installed")
        predictions = model.predict(prediction_data)
    elif model_type == "lgbm":
        predictions = model.predict(prediction_data)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Add predictions to a copy of the original DataFrame
    result_df = df.copy()
    result_df['predicted_price'] = predictions
    
    logger.info(f"Generated {len(predictions)} predictions")
    return result_df

def predict(
    data: pd.DataFrame,
    model_path: Optional[str] = None,
    model_type: str = "autogluon"
) -> pd.DataFrame:
    """
    Main prediction function.
    
    Args:
        data: Input data
        model_path: Path to the model
        model_type: Type of model ("lgbm" or "autogluon")
        
    Returns:
        DataFrame with predictions
    """
    start_time = timer()
    logger.info("Starting prediction process")
    
    # Get model path if not provided
    if not model_path:
        model_path = get_latest_model_path()
    
    # Load model info
    model_info_path = os.path.join(model_path, "model_info.json")
    if os.path.exists(model_info_path):
        import json
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        # Update model type if specified in model info
        if 'model_type' in model_info:
            model_type = model_info['model_type']
        
        # Get required features if available
        required_features = model_info.get('features', None)
    else:
        required_features = None
    
    # Load model
    model = load_model(model_path)
    
    # Make predictions
    result_df = predict_with_model(
        model, data, model_type, required_features
    )
    
    logger.info(f"Prediction completed in {timer(start_time)}")
    return result_df

def main():
    """
    Main function to run the prediction pipeline.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Make predictions with trained model')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('output_file', help='Path to output CSV file')
    parser.add_argument('--model_path', help='Path to model directory')
    parser.add_argument('--model_type', default='autogluon', 
                        choices=['autogluon', 'lgbm'],
                        help='Type of model to use')
    
    args = parser.parse_args()
    
    # Load input data
    data = pd.read_csv(args.input_file)
    
    # Make predictions
    result_df = predict(
        data, 
        model_path=args.model_path,
        model_type=args.model_type
    )
    
    # Save predictions
    result_df.to_csv(args.output_file, index=False)
    logger.info(f"Predictions saved to {args.output_file}")

if __name__ == "__main__":
    main() 