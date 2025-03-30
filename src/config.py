"""
Configuration settings for the BidPrice prediction model.
"""
import os
from pathlib import Path

# Project directories
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Data settings
RAW_DATA_FILE = DATA_DIR / "raw_data.csv"
PROCESSED_DATA_FILE = DATA_DIR / "processed_data.csv"
TRAIN_DATA_FILE = DATA_DIR / "train_data.csv"
TEST_DATA_FILE = DATA_DIR / "test_data.csv"

# 모델 경로 helper 함수
def get_model_path(dataset_key="dataset2", target_prefix="010", bin_id=None):
    """
    모델 경로를 표준화된 방식으로 반환합니다.
    
    Args:
        dataset_key (str): 데이터셋 키 (예: "dataset2", "DataSet_2", "2" 등)
        target_prefix (str): 타겟 접두사 (예: "010", "020", "050", "100")
        bin_id (str, optional): 세부 bin ID (예: "001"). None이면 타겟 접두사 레벨까지 경로 반환.
        
    Returns:
        Path: 표준화된 모델 경로
    """
    # dataset_key 표준화
    dataset_key = str(dataset_key).lower()
    
    # "dataset" 접두사 처리
    if dataset_key.startswith("dataset"):
        dataset_key = dataset_key.replace("dataset", "dataset")
    elif dataset_key.startswith("dataset_"):
        dataset_key = dataset_key.replace("dataset_", "dataset")
    # 숫자만 있는 경우
    elif dataset_key.isdigit():
        dataset_key = f"dataset{dataset_key}"
    # DataSet_ 형식 처리
    elif dataset_key.lower().startswith("dataset_"):
        dataset_key = f"dataset{dataset_key.split('_')[-1]}"
        
    # etc 케이스 처리
    if "etc" in dataset_key:
        dataset_key = "datasetetc"
        
    base_path = MODELS_DIR / "autogluon" / dataset_key / target_prefix
    if bin_id:
        # bin_id가 문자열이 아니면 문자열로 변환
        if not isinstance(bin_id, str):
            bin_id = str(bin_id)
        # models/autogluon/dataset2/010/010_001 형식
        model_path = base_path / f"{target_prefix}_{bin_id}"
        return model_path
    return base_path

# Model parameters
MODEL_PARAMS = {
    "lgbm": {
        "num_boost_round": 10000,
        "early_stopping_rounds": 100,
        "learning_rate": 0.01,
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "max_depth": -1
    },
    "autogluon": {
        "time_limit": 120,  # 2분으로 설정 (테스트용, 실제는 더 길게)
        "eval_metric": "root_mean_squared_error",
        "presets": "medium_quality_faster_train",  # 빠른 학습을 위해 프리셋 변경
        "num_stack_levels": 0,  # 스택 레벨 비활성화 (빠른 학습을 위해)
        "num_bag_folds": 2      # 백 폴드 감소 (빠른 학습을 위해)
    }
}

# Training settings
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature settings
NUMERIC_FEATURES = []  # List of numeric feature column names
CATEGORICAL_FEATURES = []  # List of categorical feature column names
TEXT_FEATURES = []  # List of text feature column names
TARGET_COLUMN = "bid_price"  # Target column name

# Metrics for evaluation
METRICS = ["rmse", "r2_score", "mae"]

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = ROOT_DIR / "logs" / "bidprice.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True) 