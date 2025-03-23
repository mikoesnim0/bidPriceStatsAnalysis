"""
Prediction module for BidPrice prediction.
"""
import os
import logging
import pandas as pd
import numpy as np
import pickle
import json
from typing import Dict, Any, Optional, List

# Import our modules
from mongodb_handler import MongoDBHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def load_model(model_path):
    """
    Load a trained model from disk.
    
    Parameters:
        model_path (str): Path to the model pickle file.
        
    Returns:
        object: The loaded model.
    """
    logger.info(f"Loading model from {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model from {model_path}: {e}")
        raise

def load_feature_columns(features_path):
    """
    Load feature columns from disk.
    
    Parameters:
        features_path (str): Path to the feature columns pickle file.
        
    Returns:
        list: List of feature column names.
    """
    
    try:
        with open(features_path, 'rb') as f:
            feature_cols = pickle.load(f)
        
        return feature_cols
    except Exception as e:
        logger.error(f"❌ Failed to load feature columns from {features_path}: {e}")
        raise

def predict_probabilities(input_data, dataset_key, target_prefix):
    """
    Make predictions using trained models.
    
    Parameters:
        input_data (pd.DataFrame): Input data for prediction.
        dataset_key (str): Dataset key (e.g., "DataSet_3", "DataSet_2").
        target_prefix (str): Prefix for target columns (e.g., "010", "020", "050", "100").
        
    Returns:
        pd.DataFrame: Predictions for each bin.
    """
    logger.info(f"Making predictions with {dataset_key} models and target prefix {target_prefix}")
    
    # Define model directory
    model_dir = os.path.join("models", dataset_key.lower())
    
    # Load feature columns
    features_path = os.path.join(model_dir, f"{dataset_key.lower()}_features.pkl")
    feature_cols = load_feature_columns(features_path)
    
    # Check if all required feature columns are in input data
    missing_features = [col for col in feature_cols if col not in input_data.columns]
    if missing_features:
        logger.error(f"❌ Missing features in input data: {missing_features}")
        raise ValueError(f"❌ Input data is missing required features: {missing_features}")
    
    # Filter input data to include only required features
    X = input_data[feature_cols].copy()
    
    # Get list of model files
    model_files = [f for f in os.listdir(model_dir) if f.startswith(f"{dataset_key.lower()}_{target_prefix}_") and f.endswith(".pkl")]
    
    if not model_files:
        logger.error(f"❌ No models found for {dataset_key} with target prefix {target_prefix}")
        raise FileNotFoundError(f"❌ No models found for {dataset_key} with target prefix {target_prefix}")
    
    # Initialize results DataFrame with input data index
    results = pd.DataFrame(index=X.index)
    
    # Load each model and make predictions
    for model_file in model_files:
        # Extract bin label from filename
        bin_label = model_file.split("_")[-1].replace(".pkl", "")
        
        # Load model
        model_path = os.path.join(model_dir, model_file)
        model = load_model(model_path)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Add predictions to results
        results[f"{target_prefix}_{bin_label}"] = predictions
    
    logger.info(f"Made predictions for {len(results)} samples across {len(model_files)} bins")
    
    return results

def load_and_predict(notice_id, dataset_key="DataSet_3", target_prefix="100", db_name=None, collection_name=None):
    """
    Load a specific notice from MongoDB and make predictions.
    
    Parameters:
        notice_id (str): The notice ID to predict for.
        dataset_key (str): Dataset key to use for prediction models.
        target_prefix (str): Prefix for target columns.
        db_name (str, optional): Specific database name to use.
        collection_name (str, optional): Specific collection name to use.
        
    Returns:
        dict: Predictions and metadata.
    """
    logger.info(f"Making predictions for notice ID: {notice_id}")
    
    # Connect to MongoDB and load the data
    with MongoDBHandler() as mongo_handler:
        if db_name and collection_name:
            # Use specified database and collection
            collection = mongo_handler.client[db_name][collection_name]
        else:
            # Use default collection based on dataset_key
            collection_names = mongo_handler.get_default_collection_names()
            collection = mongo_handler.db[collection_names[dataset_key]]
        
        # Find the notice
        notice_data = collection.find_one({"공고번호": notice_id}, {"_id": 0})
        
        if not notice_data:
            logger.error(f"❌ Notice with ID {notice_id} not found in {collection.name}")
            raise ValueError(f"❌ Notice with ID {notice_id} not found in {collection.name}")
    
    # Convert to DataFrame (single row)
    input_df = pd.DataFrame([notice_data])
    
    # Make predictions
    predictions = predict_probabilities(input_df, dataset_key, target_prefix)
    
    # Combine with input data for context
    result = {
        "notice_id": notice_id,
        "dataset_key": dataset_key,
        "target_prefix": target_prefix,
        "db_name": db_name or mongo_handler.db.name,
        "collection_name": collection_name or collection.name,
        "predictions": predictions.to_xdict(orient="records")[0],
        "metadata": {k: v for k, v in notice_data.items() if k != "공고번호"}
    }
    
    logger.info(f"Prediction complete for notice ID: {notice_id}")
    
    return result

def save_prediction(prediction, output_dir="results"):
    """
    Save prediction results to a JSON file.
    
    Parameters:
        prediction (dict): Prediction results.
        output_dir (str): Directory to save results.
        
    Returns:
        str: Path to the saved file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename
    notice_id = prediction["notice_id"]
    dataset_key = prediction["dataset_key"]
    target_prefix = prediction["target_prefix"]
    filename = f"prediction_{notice_id}_{dataset_key}_{target_prefix}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save prediction
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(prediction, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Prediction saved to {filepath}")
    
    return filepath

def get_all_notice_ids(db_name=None, collection_name=None):
    """
    Get all notice IDs from the specified database and collection.
    
    Parameters:
        db_name (str, optional): Database name to use.
        collection_name (str, optional): Collection name to use.
        
    Returns:
        list: List of notice IDs.
    """
    with MongoDBHandler() as mongo_handler:
        if db_name and collection_name:
            collection = mongo_handler.client[db_name][collection_name]
        else:
            collection_names = mongo_handler.get_default_collection_names()
            collection = mongo_handler.db[collection_names["DataSet_3"]]
        
        notice_ids = collection.distinct("공고번호")
        logger.info(f"Found {len(notice_ids)} notice IDs in {collection.name}")
        return notice_ids

def batch_predict(notice_ids, dataset_key="DataSet_3", target_prefix="100", db_name=None, collection_name=None):
    """
    Make predictions for multiple notices.
    
    Parameters:
        notice_ids (list): List of notice IDs to predict for.
        dataset_key (str): Dataset key to use for prediction models.
        target_prefix (str): Prefix for target columns.
        db_name (str, optional): Specific database name to use.
        collection_name (str, optional): Specific collection name to use.
        
    Returns:
        list: List of prediction results.
    """
    logger.info(f"Batch predicting for {len(notice_ids)} notices")
    logger.info(f"Using database: {db_name or 'default'}, collection: {collection_name or 'default'}")
    
    results = []
    
    for notice_id in notice_ids:
        try:
            prediction = load_and_predict(notice_id, dataset_key, target_prefix, db_name, collection_name)
            save_prediction(prediction)
            results.append(prediction)
        except Exception as e:
            logger.error(f"❌ Failed to predict for notice ID {notice_id}: {e}")
    
    logger.info(f"Batch prediction completed for {len(results)} out of {len(notice_ids)} notices")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict bid price distribution")
    parser.add_argument("--notice_id", type=str, help="Notice ID to predict for")
    parser.add_argument("--dataset", type=str, default="DataSet_3", help="Dataset key to use")
    parser.add_argument("--target_prefix", type=str, default="100", help="Target prefix to use")
    parser.add_argument("--batch_file", type=str, help="Path to file containing notice IDs for batch prediction")
    parser.add_argument("--db_name", type=str, help="Specific database name to use")
    parser.add_argument("--collection_name", type=str, help="Specific collection name to use")
    parser.add_argument("--all_notices", action="store_true", help="Predict for all notices in the database")
    
    args = parser.parse_args()
    
    try:
        if args.all_notices:
            notice_ids = get_all_notice_ids(args.db_name, args.collection_name)
            batch_predict(notice_ids, args.dataset, args.target_prefix, args.db_name, args.collection_name)
        
        elif args.batch_file:
            with open(args.batch_file, 'r') as f:
                notice_ids = [line.strip() for line in f if line.strip()]
            
            batch_predict(notice_ids, args.dataset, args.target_prefix, args.db_name, args.collection_name)
        
        elif args.notice_id:
            prediction = load_and_predict(args.notice_id, args.dataset, args.target_prefix, args.db_name, args.collection_name)
            saved_path = save_prediction(prediction)
            print(f"Prediction saved to {saved_path}")
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise