"""
Test script for prediction functionality.
"""
import os
import pandas as pd
import pickle
import sys
import traceback
from pathlib import Path

# 상위 디렉토리 path에 추가
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.mongodb_handler import MongoDBHandler
from src.config import get_model_path

def get_test_data(n_samples=5):
    """
    Get test data from MongoDB preprocessed collection.
    
    Parameters:
        n_samples (int): Number of samples to get
        
    Returns:
        pd.DataFrame: Test data
    """
    with MongoDBHandler(db_name='data_preprocessed') as mongo_handler:
        # 사용 가능한 컬렉션 확인
        collections = mongo_handler.db.list_collection_names()
        print(f"사용 가능한 컬렉션: {collections}")
        
        # 컬렉션 선택 (test 데이터 우선)
        test_collections = [c for c in collections if 'test' in c]
        if not test_collections:
            print("테스트 컬렉션을 찾을 수 없습니다. 임의 컬렉션을 사용합니다.")
            collection_name = collections[0] if collections else None
        else:
            collection_name = test_collections[0]
        
        if not collection_name:
            raise ValueError("데이터 컬렉션을 찾을 수 없습니다.")
        
        print(f"선택된 컬렉션: {collection_name}")
        collection = mongo_handler.db[collection_name]
        
        # Get first n samples
        cursor = collection.find({}, {"_id": 0}).limit(n_samples)
        data = list(cursor)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        print(f"Retrieved {len(df)} samples from {collection_name}")
        print("\nColumns in the data:")
        print(df.columns.tolist())
        
        if not data:
            print("데이터를 찾을 수 없습니다.")
            return None
        
        print("\nFirst row of data:")
        print(df.iloc[0])
        
        print("\nData types of columns:")
        print(df.dtypes)
        
        return df

def load_autogluon_model(dataset_key, target_prefix, bin_label):
    """
    Load AutoGluon model
    
    Parameters:
        dataset_key (str): Dataset key (e.g. "dataset2", "DataSet_3")
        target_prefix (str): Target prefix (e.g. "010")
        bin_label (str): Bin label (e.g. "001")
        
    Returns:
        object: Loaded model
    """
    try:
        import autogluon.tabular as ag
        
        # 모델 경로 얻기
        model_path = get_model_path(dataset_key, target_prefix, bin_label)
        
        print(f"모델 경로: {model_path}")
        
        # 경로가 존재하는지 확인
        if not os.path.exists(model_path):
            print(f"경로가 존재하지 않음: {model_path}")
            return None
        
        # Load the predictor
        predictor = ag.TabularPredictor.load(str(model_path))
        print(f"모델 로드 성공: {target_prefix}_{bin_label}")
        
        return predictor
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        traceback.print_exc()
        return None

def find_available_models(dataset_keys=None, target_prefixes=None):
    """
    사용 가능한 모델 찾기
    
    Parameters:
        dataset_keys (list): 검색할 데이터셋 키 목록
        target_prefixes (list): 검색할 타겟 접두사 목록
        
    Returns:
        dict: 사용 가능한 모델 목록
    """
    if dataset_keys is None:
        dataset_keys = ["dataset2", "dataset3", "datasetetc", "2", "3", "etc"]
    
    if target_prefixes is None:
        target_prefixes = ["010", "020", "050", "100"]
    
    available_models = {}
    
    for dataset_key in dataset_keys:
        for target_prefix in target_prefixes:
            try:
                # 모델 디렉토리 경로 가져오기
                model_dir = get_model_path(dataset_key, target_prefix)
                
                # 디렉토리가 존재하는지 확인
                if not os.path.exists(model_dir):
                    continue
                
                # bin 디렉토리 목록 가져오기
                bin_dirs = [d for d in os.listdir(model_dir) if d.startswith(f"{target_prefix}_")]
                
                if bin_dirs:
                    if dataset_key not in available_models:
                        available_models[dataset_key] = {}
                    
                    available_models[dataset_key][target_prefix] = [d.split("_")[1] for d in bin_dirs]
            except Exception as e:
                print(f"모델 검색 중 오류 발생 ({dataset_key}, {target_prefix}): {e}")
    
    return available_models

def predict_with_autogluon(df, dataset_key="datasetetc", target_prefix="010"):
    """
    Make predictions using AutoGluon models
    
    Parameters:
        df (pd.DataFrame): Input data
        dataset_key (str): Dataset key
        target_prefix (str): Target prefix (e.g. "010", "020")
        
    Returns:
        pd.DataFrame: Predictions
    """
    # 사용 가능한 모델 찾기
    available_models = find_available_models([dataset_key], [target_prefix])
    
    if not available_models or dataset_key not in available_models or target_prefix not in available_models[dataset_key]:
        print(f"사용 가능한 모델을 찾을 수 없습니다: {dataset_key}/{target_prefix}")
        print("다른 데이터셋 키와 타겟 접두사 조합을 시도합니다...")
        
        # 다른 조합 시도
        available_models = find_available_models()
        
        if not available_models:
            print("사용 가능한 모델이 없습니다.")
            return pd.DataFrame()
        
        # 첫 번째 사용 가능한 모델 선택
        dataset_key = list(available_models.keys())[0]
        target_prefix = list(available_models[dataset_key].keys())[0]
        print(f"사용할 모델: {dataset_key}/{target_prefix}")
    
    # 사용 가능한 bin 모델
    bin_labels = available_models[dataset_key][target_prefix]
    
    # Create results DataFrame
    results = pd.DataFrame(index=df.index)
    
    for bin_label in bin_labels:
        try:
            # Load the model
            model = load_autogluon_model(dataset_key, target_prefix, bin_label)
            
            if model is None:
                print(f"모델을 로드할 수 없음: {dataset_key}/{target_prefix}/{bin_label}")
                continue
            
            # Make predictions
            predictions = model.predict(df)
            
            # Add to results
            results[f"{target_prefix}_{bin_label}"] = predictions
            
            print(f"Successfully predicted bin {bin_label} with {target_prefix}")
            
        except Exception as e:
            print(f"Error predicting bin {bin_label}: {e}")
            traceback.print_exc()
    
    return results

def main():
    try:
        # Get test data
        test_data = get_test_data(5)
        
        if test_data is None or test_data.empty:
            print("테스트 데이터를 가져올 수 없습니다.")
            return
        
        # 모든 사용 가능한 모델 찾기
        available_models = find_available_models()
        
        if not available_models:
            print("사용 가능한 모델이 없습니다. 먼저 모델을 학습해야 합니다.")
            return
        
        print("\n사용 가능한 모델:")
        for dk in available_models:
            for tp in available_models[dk]:
                print(f"- {dk}/{tp}: {len(available_models[dk][tp])}개 bin")
        
        # 첫 번째 사용 가능한 모델로 예측
        first_dk = list(available_models.keys())[0]
        first_tp = list(available_models[first_dk].keys())[0]
        
        print(f"\n{first_dk}/{first_tp} 모델로 예측 시도...")
        
        # Try prediction with AutoGluon model
        predictions = predict_with_autogluon(test_data, dataset_key=first_dk, target_prefix=first_tp)
        
        if predictions.empty:
            print("예측에 실패했습니다.")
            return
        
        print("\nPrediction successful!")
        print("\nPrediction columns:")
        print(predictions.columns.tolist())
        print("\nFirst row of predictions:")
        print(predictions.iloc[0])
        
    except Exception as e:
        print(f"\nError during prediction: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 