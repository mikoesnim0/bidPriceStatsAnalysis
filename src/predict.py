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
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from db_config.mongodb_handler import MongoDBHandler
from src.config import get_model_path

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

def predict_probabilities(df, dataset_key="dataset2", target_prefix="100"):
    """
    Make predictions for each probability bin.
    
    Parameters:
        df (pd.DataFrame): Input data
        dataset_key (str): Dataset key for models to use
        target_prefix (str): Target column prefix
        
    Returns:
        pd.DataFrame: Predictions for each bin
    """
    logger.info(f"Making predictions with models from dataset: {dataset_key}, prefix: {target_prefix}")
    
    # 결과 저장할 DataFrame
    results = pd.DataFrame(index=df.index)
    
    try:
        # AutoGluon 사용 가능 여부 확인
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError:
            logger.error("AutoGluon is not installed. Install it with: pip install autogluon.tabular")
            return results
        
        # 모델 디렉토리 경로
        from src.config import get_model_path
        model_dir = get_model_path(dataset_key, target_prefix)
        
        if not os.path.exists(model_dir):
            logger.warning(f"Model directory not found: {model_dir}")
            return results
        
        # 모델 디렉토리 확인
        bin_dirs = []
        try:
            bin_dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d)) and d.startswith(f"{target_prefix}_")]
        except Exception as e:
            logger.error(f"Error listing model directory {model_dir}: {str(e)}")
            return results
        
        if not bin_dirs:
            logger.warning(f"No model bins found in {model_dir}")
            return results
        
        logger.info(f"Found {len(bin_dirs)} model bins: {bin_dirs}")
        
        # 각 bin에 대해 추론
        for bin_dir_name in bin_dirs:
            try:
                bin_path = os.path.join(model_dir, bin_dir_name)
                
                # bin ID 추출
                bin_id = bin_dir_name.split("_")[-1]
                col_name = f"{target_prefix}_{bin_id}"
                
                # 모델 로드
                logger.info(f"Loading model from {bin_path}")
                predictor = TabularPredictor.load(bin_path)
                
                # 예측 수행
                try:
                    predictions = predictor.predict(df)
                    results[col_name] = predictions
                    logger.info(f"Successfully predicted {col_name}, shape: {predictions.shape}")
                except Exception as pred_err:
                    logger.error(f"Error during prediction for {col_name}: {str(pred_err)}")
                    # NaN으로 채우기
                    results[col_name] = float('nan')
            
            except Exception as e:
                logger.error(f"Error loading/using model {bin_dir_name}: {str(e)}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error in predict_probabilities: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return results

def load_and_predict(notice_id, dataset_key="dataset2", target_prefix="100", db_name="data_preprocessed", collection_name=None):
    """
    Load a specific notice from MongoDB and make predictions.
    
    Parameters:
        notice_id (str): The notice ID to predict for.
        dataset_key (str): Dataset key to use for prediction models (e.g., "dataset2", "2", "DataSet_2").
        target_prefix (str): Prefix for target columns (e.g., "010", "020", "050", "100").
        db_name (str, optional): Specific database name to use.
        collection_name (str, optional): Specific collection name to use.
        
    Returns:
        dict: Predictions and metadata.
    """
    logger.info(f"Making predictions for notice ID: {notice_id}")
    
    try:
        mongo_handler = MongoDBHandler()
        db = mongo_handler.client[db_name]

        available_collections = db.list_collection_names()
        logger.info(f"Available collections: {available_collections}")

        # 컬렉션 결정
        if collection_name:
            # 지정된 컬렉션 사용
            if collection_name not in available_collections:
                raise ValueError(f"Specified collection '{collection_name}' not found in database")
            collection = db[collection_name]
        else:
            # dataset_key에 따른 컬렉션 형식 준비 (test 컬렉션 우선)
            collection_patterns = [
                f"preprocessed_dataset{dataset_key}_test",
                f"preprocessed_dataset{dataset_key}_train",
                f"preprocessed_dataset_{dataset_key}_test",
                f"preprocessed_dataset_{dataset_key}_train",
                f"preprocessed_{dataset_key}_test",
                f"preprocessed_{dataset_key}_train",
                f"preprocessed_test",
                f"preprocessed_train"
            ]

            # 사용 가능한 첫 번째 컬렉션 선택
            selected_collection = next((c for c in collection_patterns if c in available_collections), None)

            if not selected_collection:
                # 대체 방법: 모든 컬렉션에서 해당 공고번호 검색
                for coll_name in available_collections:
                    sample = db[coll_name].find_one({"공고번호": notice_id})
                    if sample:
                        selected_collection = coll_name
                        logger.info(f"Found notice in collection: {selected_collection}")
                        break

            if not selected_collection:
                raise ValueError(f"Could not find appropriate collection for dataset_key: {dataset_key}")

            collection = db[selected_collection]
            logger.info(f"Using collection: {selected_collection}")

        # 공고 데이터 검색
        notice_data = collection.find_one({"공고번호": notice_id}, {"_id": 0})

        if not notice_data:
            # 다른 모든 컬렉션에서도 검색
            for coll_name in available_collections:
                if coll_name == collection.name:
                    continue

                notice_data = db[coll_name].find_one({"공고번호": notice_id}, {"_id": 0})
                if notice_data:
                    logger.info(f"Found notice in different collection: {coll_name}")
                    break

            if not notice_data:
                raise ValueError(f"Notice with ID {notice_id} not found in any collection")

        # DataFrame으로 변환 (단일 행)
        input_df = pd.DataFrame([notice_data])

        # 모델 디렉토리 확인
        model_base_dir = get_model_path(dataset_key, target_prefix)
        if not os.path.exists(model_base_dir):
            logger.warning(f"Model directory not found: {model_base_dir}")

            # 다른 dataset_key 시도
            alternative_keys = ["dataset2", "dataset3", "datasetetc", "2", "3", "etc"]
            for alt_key in alternative_keys:
                if alt_key == dataset_key:
                    continue

                alt_path = get_model_path(alt_key, target_prefix)
                if os.path.exists(alt_path):
                    logger.info(f"Using alternative model path: {alt_path}")
                    dataset_key = alt_key
                    model_base_dir = alt_path
                    break

        # 예측 수행
        predictions = predict_probabilities(input_df, dataset_key, target_prefix)

        # 입력 데이터와 함께 결과 반환
        result = {
            "notice_id": notice_id,
            "dataset_key": dataset_key,
            "target_prefix": target_prefix,
            "db_name": db_name,
            "collection_name": collection.name,
            "predictions": predictions.to_dict(orient="records")[0] if not predictions.empty else {},
            "metadata": {k: v for k, v in notice_data.items() if k != "공고번호"}
        }

        logger.info(f"Prediction complete for notice ID: {notice_id}")

        return result
    
    except Exception as e:
        logger.error(f"Error in load_and_predict: {str(e)}")
        # 추적 정보 로깅
        import traceback
        logger.error(traceback.format_exc())
        
        # 오류 정보 포함한 결과 반환
        return {
            "notice_id": notice_id,
            "dataset_key": dataset_key,
            "target_prefix": target_prefix,
            "error": str(e),
            "success": False
        }

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