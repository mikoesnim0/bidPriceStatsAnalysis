"""
Data loading and preprocessing module for BidPrice prediction.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import os
from typing import Tuple, Dict, Any, List, Optional

from src.config import (
    RAW_DATA_FILE, PROCESSED_DATA_FILE, TRAIN_DATA_FILE, TEST_DATA_FILE,
    DATA_DIR, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TEXT_FEATURES, TARGET_COLUMN,
    RANDOM_SEED, TEST_SIZE
)
from src.utils import setup_logger

logger = setup_logger(__name__)

def load_data(file_path: str = None) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Loaded DataFrame
    """
    if file_path is None:
        # 사용자가 파일 경로를 지정하지 않은 경우, 기본 파일 경로 사용
        file_path = os.path.join(DATA_DIR, "DR_Modified_3_output.csv")
        if not os.path.exists(file_path):
            file_path = RAW_DATA_FILE

    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw data.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    logger.info("Preprocessing data")
    
    # Make a copy of the DataFrame to avoid modifying the original
    processed_df = df.copy()
    
    # Handle missing values
    for col in processed_df.columns:
        missing_count = processed_df[col].isna().sum()
        if missing_count > 0:
            logger.info(f"Column {col} has {missing_count} missing values")
            
            if col in NUMERIC_FEATURES:
                # Fill numeric features with median
                processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                logger.info(f"Filled missing values in {col} with median")
            elif col in CATEGORICAL_FEATURES:
                # Fill categorical features with mode
                processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
                logger.info(f"Filled missing values in {col} with mode")
            elif col in TEXT_FEATURES:
                # Fill text features with empty string
                processed_df[col] = processed_df[col].fillna("")
                logger.info(f"Filled missing values in {col} with empty string")
    
    # Handle categorical features
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in processed_df.columns:
            le = LabelEncoder()
            processed_df[col] = le.fit_transform(processed_df[col])
            label_encoders[col] = le
            logger.info(f"Label encoded column {col}")
    
    # Basic text feature processing
    for col in TEXT_FEATURES:
        if col in processed_df.columns:
            # Create text length feature
            processed_df[f"{col}_length"] = processed_df[col].str.len()
            logger.info(f"Created text length feature for {col}")
    
    # Log preprocessing completion
    logger.info(f"Preprocessing complete. Output shape: {processed_df.shape}")
    
    return processed_df

def split_train_test(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info(f"Splitting data into train and test sets with test_size={test_size}")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        file_path: Path to save the CSV file
    """
    logger.info(f"Saving data to {file_path}")
    try:
        df.to_csv(file_path, index=False)
        logger.info(f"Data saved successfully")
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        raise

def prepare_features(
    df: pd.DataFrame,
    numeric_features: List[str] = NUMERIC_FEATURES,
    categorical_features: List[str] = CATEGORICAL_FEATURES,
    text_features: List[str] = TEXT_FEATURES,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = False
) -> Tuple[pd.DataFrame, Optional[StandardScaler]]:
    """
    Prepare features for model training or prediction.
    
    Args:
        df: Input DataFrame
        numeric_features: List of numeric feature column names
        categorical_features: List of categorical feature column names
        text_features: List of text feature column names
        scaler: Optional pre-fitted scaler for numeric features
        fit_scaler: Whether to fit the scaler on this data
        
    Returns:
        Transformed DataFrame and fitted scaler (if fit_scaler is True)
    """
    logger.info("Preparing features for model")
    
    # Validate features
    all_features = numeric_features + categorical_features + text_features
    missing_cols = [col for col in all_features if col not in df.columns]
    
    if missing_cols:
        logger.warning(f"Missing columns in data: {missing_cols}")
    
    # Scale numeric features
    if numeric_features:
        numeric_df = df[numeric_features].copy()
        
        if scaler is None and fit_scaler:
            scaler = StandardScaler()
            numeric_scaled = scaler.fit_transform(numeric_df)
            logger.info("Fitted and applied StandardScaler to numeric features")
        elif scaler is not None:
            numeric_scaled = scaler.transform(numeric_df)
            logger.info("Applied pre-fitted StandardScaler to numeric features")
        else:
            numeric_scaled = numeric_df.values
            
        for i, col in enumerate(numeric_features):
            df[col] = numeric_scaled[:, i]
    
    return (df, scaler) if fit_scaler else (df, None)

def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load preprocessed training and testing data.
    
    Returns:
        train_X, test_X, train_Y, test_Y DataFrames
    """
    logger.info("Loading preprocessed data")
    
    try:
        # 필요한 파일 경로
        train_file = os.path.join(DATA_DIR, "train_data.csv")
        test_file = os.path.join(DATA_DIR, "test_data.csv")
        train_targets_file = os.path.join(DATA_DIR, "train_targets.csv")
        test_targets_file = os.path.join(DATA_DIR, "test_targets.csv")
        
        # 파일 존재 확인
        required_files = [train_file, test_file, train_targets_file, test_targets_file]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            logger.warning(f"Missing preprocessed data files: {missing_files}")
            raise FileNotFoundError(f"Missing preprocessed data files: {missing_files}")
        
        # 데이터 로드
        logger.info(f"Loading train features from {train_file}")
        train_X = pd.read_csv(train_file)
        
        logger.info(f"Loading test features from {test_file}")
        test_X = pd.read_csv(test_file)
        
        logger.info(f"Loading train targets from {train_targets_file}")
        train_Y = pd.read_csv(train_targets_file)
        
        logger.info(f"Loading test targets from {test_targets_file}")
        test_Y = pd.read_csv(test_targets_file)
        
        # 데이터셋 형태 확인
        logger.info(f"Train features shape: {train_X.shape}")
        logger.info(f"Test features shape: {test_X.shape}")
        logger.info(f"Train targets shape: {train_Y.shape}")
        logger.info(f"Test targets shape: {test_Y.shape}")
        
        # 타겟 컬럼 확인
        logger.info(f"Target columns: {train_Y.columns.tolist()}")
        
        return train_X, test_X, train_Y, test_Y
    
    except Exception as e:
        logger.error(f"Error loading preprocessed data: {str(e)}")
        raise

def split_and_save_data(X_train, Y_train, X_test=None, Y_test=None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    이미 분할된 데이터를 디스크에 저장합니다.
    혹은 단일 데이터셋이 제공된 경우 분할 후 저장합니다.
    
    Args:
        X_train: 학습 특성 DataFrame
        Y_train: 학습 타겟 DataFrame
        X_test: 테스트 특성 DataFrame (없으면 X_train에서 분할)
        Y_test: 테스트 타겟 DataFrame (없으면 Y_train에서 분할)
        
    Returns:
        train_X, test_X, train_Y, test_Y - 분할된 DataFrames
    """
    logger.info("데이터 저장 중")
    
    # 데이터가 아직 분할되지 않은 경우 분할 수행
    if X_test is None or Y_test is None:
        logger.info("데이터 분할 중 (테스트 데이터가 제공되지 않음)")
        # 학습/테스트 분할 (stratify 사용 안 함 - 여러 타겟 컬럼이 있을 수 있음)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_train, Y_train, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )
    
    logger.info(f"Train features shape: {X_train.shape}")
    logger.info(f"Test features shape: {X_test.shape}")
    logger.info(f"Train targets shape: {Y_train.shape}")
    logger.info(f"Test targets shape: {Y_test.shape}")
    
    # 디렉토리 생성
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 파일 저장
    train_file = os.path.join(DATA_DIR, "train_data.csv")
    test_file = os.path.join(DATA_DIR, "test_data.csv")
    train_targets_file = os.path.join(DATA_DIR, "train_targets.csv")
    test_targets_file = os.path.join(DATA_DIR, "test_targets.csv")
    
    # 특성 데이터 저장
    logger.info(f"Saving train features to {train_file}")
    X_train.to_csv(train_file, index=False)
    
    logger.info(f"Saving test features to {test_file}")
    X_test.to_csv(test_file, index=False)
    
    # 타겟 데이터 저장
    logger.info(f"Saving train targets to {train_targets_file}")
    Y_train.to_csv(train_targets_file, index=False)
    
    logger.info(f"Saving test targets to {test_targets_file}")
    Y_test.to_csv(test_targets_file, index=False)
    
    logger.info("Data successfully split and saved")
    
    return X_train, X_test, Y_train, Y_test

def main():
    """Main function to demonstrate data processing pipeline."""
    try:
        # Load raw data
        # raw_data = load_data(RAW_DATA_FILE)
        
        # Preprocess data
        # processed_data = preprocess_data(raw_data)
        processed_data = pd.read_csv(PROCESSED_DATA_FILE)

        # Save processed data
        # save_data(processed_data, PROCESSED_DATA_FILE)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = split_train_test(processed_data)
        
        # Prepare features
        X_train_prepared, scaler = prepare_features(X_train, fit_scaler=True)
        X_test_prepared, _ = prepare_features(X_test, scaler=scaler, fit_scaler=False)
        
        # Save train and test data
        train_data = X_train_prepared.copy()
        train_data[TARGET_COLUMN] = y_train.values
        save_data(train_data, TRAIN_DATA_FILE)
        
        test_data = X_test_prepared.copy()
        test_data[TARGET_COLUMN] = y_test.values
        save_data(test_data, TEST_DATA_FILE)
        
        logger.info("Data processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in data processing pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 