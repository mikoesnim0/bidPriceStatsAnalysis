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
        "time_limit": 10000,  # 1 hour
        "eval_metric": "root_mean_squared_error",
        "presets": "best_quality",
        "num_stack_levels": 2,
        "num_bag_folds": 5
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

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = ROOT_DIR / "logs" / "bidprice.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True) 