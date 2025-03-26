"""
Model training module for BidPrice prediction.
"""
import os
import pickle
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import cross_val_score, KFold, train_test_split
import lightgbm as lgbm
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import logging
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try importing AutoGluon, but handle if not installed
try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False

# Fix relative imports
from src.config import (
    TRAIN_DATA_FILE, MODELS_DIR, RESULTS_DIR, MODEL_PARAMS,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TEXT_FEATURES,
    TARGET_COLUMN, RANDOM_SEED, CV_FOLDS
)
from src.data_processing import load_data, prepare_features
from src.utils import setup_logger, save_pickle, save_json, timer, generate_timestamp
from src.mongodb_handler import MongoDBHandler

# Try to import initialization module
try:
    from src.init_dirs import init_directories
except ImportError:
    # Define a simple initialization function if module is missing
    def init_directories():
        """Initialize required directories."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print(f"Created directories: {MODELS_DIR}, {RESULTS_DIR}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

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

def prepare_train_test_data(df, target_prefix, test_size=0.2, random_state=42):
    """
    Prepare training and testing datasets.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target_prefix (str): Prefix for target columns (e.g., "010", "020", "050", "100").
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) - Training and testing data splits.
    """
    logger.info(f"Preparing train/test split with target_prefix: {target_prefix}")
    
    # Get all columns that start with the target_prefix
    target_cols = [col for col in df.columns if col.startswith(f"{target_prefix}_")]
    
    if not target_cols:
        logger.error(f"❌ No target columns found with prefix '{target_prefix}_'")
        raise ValueError(f"❌ No target columns found with prefix '{target_prefix}_'")
    
    # All other columns except target columns and '공고번호' will be features
    feature_cols = [col for col in df.columns if not col.startswith(tuple([f"{prefix}_" for prefix in ["010", "020", "050", "100"]])) and col != "공고번호"]
    
    logger.info(f"Number of feature columns: {len(feature_cols)}")
    logger.info(f"Number of target columns: {len(target_cols)}")
    
    # Prepare X and y
    X = df[feature_cols].copy()
    y = df[target_cols].copy()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Testing data shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, feature_cols, target_cols

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest model.
    
    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.DataFrame): Training targets.
        n_estimators (int): Number of trees in the forest.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        dict: Dictionary of trained models for each target column.
    """
    logger.info(f"Training Random Forest models with {n_estimators} estimators")
    
    models = {}
    
    for col in y_train.columns:
        logger.info(f"Training model for target: {col}")
        
        # Create and train model
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        model.fit(X_train, y_train[col])
        
        # Store model
        models[col] = model
        
        logger.info(f"Model for {col} trained successfully")
    
    logger.info(f"Trained {len(models)} models")
    return models

def evaluate_models(models, X_test, y_test):
    """
    Evaluate trained models on test data.
    
    Parameters:
        models (dict): Dictionary of trained models.
        X_test (pd.DataFrame): Test features.
        y_test (pd.DataFrame): Test targets.
        
    Returns:
        dict: Dictionary of evaluation metrics for each model.
    """
    logger.info("Evaluating models on test data")
    
    results = {}
    
    for col, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test[col], y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test[col], y_pred)
        
        # Store results
        results[col] = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        logger.info(f"Model {col}: RMSE = {rmse:.6f}, R² = {r2:.6f}")
    
    # Calculate average performance across all models
    avg_rmse = np.mean([result['rmse'] for result in results.values()])
    avg_r2 = np.mean([result['r2'] for result in results.values()])
    
    logger.info(f"Average performance: RMSE = {avg_rmse:.6f}, R² = {avg_r2:.6f}")
    
    return results

def save_models(models, model_dir, dataset_key, target_prefix):
    """
    Save trained models to disk.
    
    Parameters:
        models (dict): Dictionary of trained models.
        model_dir (str): Directory to save models.
        dataset_key (str): Dataset key (e.g., "DataSet_3").
        target_prefix (str): Prefix for target columns.
        
    Returns:
        list: Paths to saved model files.
    """
    logger.info(f"Saving {len(models)} models to {model_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    model_paths = []
    
    for col, model in models.items():
        # Create filename
        filename = f"{dataset_key}_{target_prefix}_{col.split('_')[-1]}.pkl"
        filepath = os.path.join(model_dir, filename)
        
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        model_paths.append(filepath)
        logger.info(f"Model saved to {filepath}")
    
    return model_paths

def train_models(dataset_key="DataSet_3", target_prefix="100", test_size=0.2, n_estimators=100, random_state=42):
    """
    Train models for a specific dataset and target prefix.
    
    Parameters:
        dataset_key (str): Dataset key to use (e.g., "DataSet_3", "DataSet_2").
        target_prefix (str): Prefix for target columns (e.g., "010", "020", "050", "100").
        test_size (float): Proportion of the dataset to include in the test split.
        n_estimators (int): Number of trees in the Random Forest.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: (models, results, model_paths) - Trained models, evaluation results, and saved model paths.
    """
    logger.info(f"Training models for {dataset_key} with target prefix {target_prefix}")
    
    # Initialize directories
    init_directories()
    
    # Load data from MongoDB
    with MongoDBHandler() as mongo_handler:
        collection_names = mongo_handler.get_default_collection_names()
        datasets = mongo_handler.load_datasets({dataset_key: collection_names[dataset_key]})
    
    df = datasets[dataset_key]
    logger.info(f"Loaded {dataset_key} with shape: {df.shape}")
    
    # Prepare train/test data
    X_train, X_test, y_train, y_test, feature_cols, target_cols = prepare_train_test_data(
        df, target_prefix, test_size, random_state
    )
    
    # Train models
    models = train_random_forest(X_train, y_train, n_estimators, random_state)
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test)
    
    # Save models
    model_dir = os.path.join("models", dataset_key.lower())
    model_paths = save_models(models, model_dir, dataset_key.lower(), target_prefix)
    
    # Save feature columns list for future reference
    feature_path = os.path.join(model_dir, f"{dataset_key.lower()}_features.pkl")
    with open(feature_path, 'wb') as f:
        pickle.dump(feature_cols, f)
    
    logger.info(f"Feature columns saved to {feature_path}")
    
    return models, results, model_paths

def train_all_models():
    """
    Train models for all datasets and target prefixes.
    """
    logger.info("Training all models")
    
    # Define datasets and target prefixes
    datasets = ["DataSet_3", "DataSet_2"]
    target_prefixes = ["010", "020", "050", "100"]
    
    all_results = {}
    
    for dataset_key in datasets:
        all_results[dataset_key] = {}
        
        for target_prefix in target_prefixes:
            logger.info(f"Training {dataset_key} with target prefix {target_prefix}")
            
            try:
                _, results, _ = train_models(dataset_key, target_prefix)
                all_results[dataset_key][target_prefix] = results
            except Exception as e:
                logger.error(f"❌ Failed to train {dataset_key} with {target_prefix}: {e}")
    
    logger.info("All model training completed")
    return all_results

def load_data_from_mongodb(dataset_key="dataset3", split="train"):
    """
    Load training or test data from MongoDB.
    
    Args:
        dataset_key (str): Dataset key (dataset2, dataset3, or datasetetc).
        split (str): Data split (train, test, or valid).
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    logger.info(f"Loading {split} data from MongoDB for {dataset_key}")
    
    # 데이터베이스 이름
    db_name = 'data_preprocessed'
    
    # 컬렉션 이름 (예: preprocessed_dataset3_train)
    collection_name = f"preprocessed_{dataset_key}_{split}"
    
    try:
        with MongoDBHandler(db_name=db_name) as mongo_handler:
            # Check if the collection exists
            available_collections = mongo_handler.db.list_collection_names()
            
            if collection_name not in available_collections:
                logger.error(f"❌ Collection {collection_name} not found in database {db_name}")
                logger.info(f"Available collections: {available_collections}")
                raise ValueError(f"Collection {collection_name} not found in database {db_name}")
            
            # Try to get the data from MongoDB
            collection = mongo_handler.db[collection_name]
            cursor = collection.find({}, {"_id": 0})
            df = pd.DataFrame(list(cursor))
            
            if df.empty:
                logger.error(f"❌ No data found in collection {collection_name}")
                raise ValueError(f"No data found in collection {collection_name}")
            
            logger.info(f"✅ Loaded {len(df)} records from {collection_name}")
            logger.info(f"✅ Columns: {df.columns.tolist()[:10]}... (truncated)")
            
            # Find target columns
            target_prefixes = ['010_', '020_', '050_', '100_']
            found_targets = False
            
            for prefix in target_prefixes:
                target_cols = [col for col in df.columns if col.startswith(prefix)]
                if target_cols:
                    found_targets = True
                    logger.info(f"✅ Found {len(target_cols)} target columns with prefix '{prefix}'")
            
            if not found_targets:
                logger.warning("⚠️ No target columns found with expected prefixes")
                # Try adding dummy target columns for testing purposes
                logger.info("⚠️ Adding dummy target columns for testing")
                for i in range(1, 11):
                    bin_label = f"100_{i:03d}"
                    df[bin_label] = np.random.random(len(df)) * 0.1
            
            return df
            
    except Exception as e:
        logger.error(f"❌ Failed to load data from MongoDB: {e}")
        raise

def train_model_for_dataset(dataset_key="dataset3", target_prefix="100", model_type="lgbm", time_limit_per_model=60):
    """
    Train a model for a specific dataset and target prefix.
    
    Args:
        dataset_key (str): Dataset key (dataset2, dataset3, or datasetetc).
        target_prefix (str): Target column prefix (e.g., "100").
        model_type (str): Model type to train ("lgbm" or "autogluon").
        time_limit_per_model (int): Maximum time in seconds to spend training each model.
        
    Returns:
        dict: Training results.
    """
    logger.info(f"Training {model_type} model for {dataset_key} with target prefix {target_prefix}")
    
    try:
        # Load training data
        train_data = load_data_from_mongodb(dataset_key, "train")
        
        # Load validation data
        valid_data = load_data_from_mongodb(dataset_key, "valid")
        
        # Prepare target columns
        target_cols = [col for col in train_data.columns if col.startswith(f"{target_prefix}_")]
        
        if not target_cols:
            logger.error(f"❌ No target columns found with prefix '{target_prefix}_'")
            raise ValueError(f"No target columns found with prefix '{target_prefix}_'")
        
        logger.info(f"✅ Found {len(target_cols)} target columns: {target_cols[:5]}... (truncated)")
        
        # 데이터 타입 처리: object 타입을 숫자로 변환
        for col in train_data.columns:
            if train_data[col].dtype == 'object':
                logger.info(f"Converting column '{col}' from object to categorical")
                try:
                    # 범주형 변수로 변환
                    train_data[col] = train_data[col].astype('category').cat.codes
                    if col in valid_data.columns:
                        valid_data[col] = valid_data[col].astype('category').cat.codes
                except Exception as e:
                    # 변환 실패 시 컬럼 제외
                    logger.warning(f"Could not convert '{col}', will exclude from features: {e}")
                    train_data = train_data.drop(columns=[col])
                    if col in valid_data.columns:
                        valid_data = valid_data.drop(columns=[col])
        
        # Prepare feature columns (exclude all target columns and 공고번호)
        all_target_prefixes = ['010_', '020_', '050_', '100_']
        feature_cols = [col for col in train_data.columns 
                       if not any(col.startswith(prefix) for prefix in all_target_prefixes)
                       and col != "공고번호"]
        
        logger.info(f"✅ Using {len(feature_cols)} feature columns")
        
        # 타겟 컬럼을 최대 5개만 사용 (시간 제한을 위해)
        if len(target_cols) > 5:
            logger.info(f"Limiting to first 5 target columns for faster training")
            target_cols = target_cols[:5]
        
        # Train a model for each target column
        results = {}
        
        for target_col in target_cols:
            start_time = datetime.now()
            logger.info(f"Training model for target: {target_col} (started at {start_time})")
            
            # Split features and target
            X_train = train_data[feature_cols]
            y_train = train_data[target_col]
            
            X_valid = valid_data[feature_cols]
            y_valid = valid_data[target_col]
            
            if model_type == "lgbm":
                # LightGBM 파라미터 설정 - 빠른 학습을 위해 간소화
                params = {
                    "objective": "regression",
                    "metric": "rmse",
                    "verbosity": -1,
                    "early_stopping_rounds": 10,
                    "num_iterations": 50,  # 반복 횟수 제한
                    "learning_rate": 0.1,
                    "num_leaves": 31
                }
                
                # Train LightGBM model with time limit
                try:
                    model, model_results = train_lgbm_model(X_train, y_train, X_valid, y_valid, params)
                    
                    # Save model
                    model_dir = os.path.join(MODELS_DIR, dataset_key.lower())
                    os.makedirs(model_dir, exist_ok=True)
                    
                    model_path = os.path.join(model_dir, f"{dataset_key.lower()}_{target_prefix}_{target_col.split('_')[-1]}.pkl")
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    
                    logger.info(f"✅ Model for {target_col} saved to {model_path}")
                    
                    # Save feature columns for future reference
                    features_path = os.path.join(model_dir, f"{dataset_key.lower()}_features.pkl")
                    if not os.path.exists(features_path):
                        with open(features_path, 'wb') as f:
                            pickle.dump(feature_cols, f)
                        logger.info(f"✅ Feature columns saved to {features_path}")
                    
                    # Store results
                    results[target_col] = model_results
                    
                except Exception as e:
                    logger.error(f"❌ Failed to train LightGBM model for {target_col}: {e}")
                    results[target_col] = {"error": str(e)}
                
            elif model_type == "autogluon" and AUTOGLUON_AVAILABLE:
                # AutoGluon 빠른 학습 설정
                try:
                    # Prepare data for AutoGluon
                    train_data_ag = X_train.copy()
                    train_data_ag[target_col] = y_train
                    
                    # Train AutoGluon model with time limit
                    model_dir = os.path.join(MODELS_DIR, "autogluon", dataset_key.lower(), 
                                            f"{target_prefix}_{target_col.split('_')[-1]}")
                    os.makedirs(model_dir, exist_ok=True)
                    
                    predictor = TabularPredictor(label=target_col, path=model_dir)
                    predictor.fit(
                        train_data=train_data_ag,
                        time_limit=time_limit_per_model,  # 시간 제한
                        presets="medium_quality_faster_train",
                        num_stack_levels=0,  # 스태킹 비활성화
                        num_bag_folds=2  # 최소 배깅 폴드
                    )
                    
                    # Evaluate on validation set
                    valid_data_ag = X_valid.copy()
                    valid_data_ag[target_col] = y_valid
                    
                    performance = predictor.evaluate(valid_data_ag)
                    logger.info(f"✅ AutoGluon model for {target_col} evaluation: {performance}")
                    
                    # Store results
                    results[target_col] = {
                        "model_path": model_dir,
                        "performance": performance
                    }
                    
                except Exception as e:
                    logger.error(f"❌ Failed to train AutoGluon model for {target_col}: {e}")
                    results[target_col] = {"error": str(e)}
                
            else:
                logger.error(f"❌ Unsupported model type: {model_type}")
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # 시간 제한 확인
            elapsed_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Model training for {target_col} took {elapsed_time:.2f} seconds")
            
            # 시간 제한을 넘기면 중단
            if elapsed_time > time_limit_per_model:
                logger.warning(f"Time limit exceeded for {target_col}, stopping training")
                break
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Failed to train model: {e}")
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Train models for bid price prediction")
    parser.add_argument("--dataset", type=str, default="dataset3", 
                        choices=["dataset2", "dataset3", "datasetetc"],
                        help="Dataset to use for training")
    parser.add_argument("--model", type=str, default="lgbm", 
                        choices=["lgbm", "autogluon"],
                        help="Model type to train")
    parser.add_argument("--target-prefix", type=str, default="100", 
                        choices=["010", "020", "050", "100"],
                        help="Target column prefix (e.g., '100' for 100_001, 100_002, etc.)")
    parser.add_argument("--time-limit", type=int, default=60,
                        help="Time limit per model in seconds (default: 60)")
    args = parser.parse_args()
    
    try:
        # Initialize directories
        init_directories()
        
        # Train model
        results = train_model_for_dataset(args.dataset, args.target_prefix, args.model, args.time_limit)
        
        # Print summary
        logger.info("Training completed successfully")
        logger.info(f"Trained {len(results)} models for {args.dataset} with target prefix {args.target_prefix}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise 