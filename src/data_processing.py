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
    
    If preprocessed data files don't exist, attempt to load from DR_Modified_3_output.csv 
    and perform preprocessing.
    
    Returns:
        X_train, X_test, y_train, y_test DataFrames
    """
    logger.info("Loading preprocessed data")
    
    # Default processed data path
    dr_modified_path = os.path.join(DATA_DIR, "DR_Modified_3_output.csv")
    
    try:
        # Try to load train and test data
        if os.path.exists(TRAIN_DATA_FILE) and os.path.exists(TEST_DATA_FILE):
            logger.info(f"Loading train data from {TRAIN_DATA_FILE}")
            train_data = pd.read_csv(TRAIN_DATA_FILE)
            
            logger.info(f"Loading test data from {TEST_DATA_FILE}")
            test_data = pd.read_csv(TEST_DATA_FILE)
            
            # Extract features and targets
            if TARGET_COLUMN in train_data.columns and TARGET_COLUMN in test_data.columns:
                X_train = train_data.drop(columns=[TARGET_COLUMN])
                y_train = train_data[TARGET_COLUMN]
                
                X_test = test_data.drop(columns=[TARGET_COLUMN])
                y_test = test_data[TARGET_COLUMN]
                
                logger.info(f"Successfully loaded preprocessed data. "
                           f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
                
                return X_train, X_test, y_train, y_test
            else:
                logger.warning(f"Target column '{TARGET_COLUMN}' not found in data files")
        
        # If train/test files don't exist or are invalid, try loading DR_Modified_3_output.csv
        if os.path.exists(dr_modified_path):
            logger.info(f"Train/test split files not found or invalid. Loading from {dr_modified_path}")
            data = pd.read_csv(dr_modified_path)
            
            # Check if this is already preprocessed data
            logger.info(f"Loaded data with shape {data.shape}")
            
            # Split into features and target(s)
            # Assuming all columns that start with numeric values (e.g., 020_001) are targets
            target_columns = [col for col in data.columns if any(c.isdigit() for c in col[:3])]
            
            if not target_columns:
                # If no target columns found, assume the default target column
                if TARGET_COLUMN in data.columns:
                    target_columns = [TARGET_COLUMN]
                else:
                    raise ValueError(f"No target columns found in {dr_modified_path}")
            
            logger.info(f"Identified {len(target_columns)} target columns: {target_columns[:5]}...")
            
            # *** 중요: NaN 값이 있는 행 제거 ***
            # AutoGluon은 타겟 변수에 NaN 값이 있으면 학습 실패함
            original_size = len(data)
            
            # 모든 타겟 컬럼에 대해 NaN이 있는 행 확인
            nan_rows = data[target_columns].isna().any(axis=1)
            nan_count = nan_rows.sum()
            
            if nan_count > 0:
                logger.warning(f"Found {nan_count} rows with NaN values in target columns. Removing these rows.")
                # NaN 값이 있는 행 제거
                data = data[~nan_rows]
                logger.info(f"Data shape after removing NaN rows: {data.shape} (removed {original_size - len(data)} rows)")
            
            # 처리 후 NaN 확인
            remaining_nans = data[target_columns].isna().any().sum()
            if remaining_nans > 0:
                logger.error(f"There are still {remaining_nans} target columns with NaN values after processing")
                raise ValueError("Failed to remove all NaN values from target columns")
            else:
                logger.info("All rows with NaN values in target columns have been successfully removed")
            
            # Create a single target DataFrame with all target columns
            y_data = data[target_columns]
            X_data = data.drop(columns=target_columns)
            
            # Split into train and test
            X_train, X_test, y_train, y_test = train_test_split(
                X_data, y_data, test_size=TEST_SIZE, random_state=RANDOM_SEED
            )
            
            logger.info(f"Split data into train and test sets. "
                       f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
            
            # Save the split data for future use
            try:
                train_data = pd.concat([X_train, y_train], axis=1)
                test_data = pd.concat([X_test, y_test], axis=1)
                
                save_data(train_data, TRAIN_DATA_FILE)
                save_data(test_data, TEST_DATA_FILE)
                
                logger.info(f"Saved train and test data for future use")
            except Exception as e:
                logger.warning(f"Could not save train/test data: {str(e)}")
            
            return X_train, X_test, y_train, y_test
        
        # If all else fails, raise an error
        raise FileNotFoundError(f"No preprocessed data files found. Please run data processing first.")
    
    except Exception as e:
        logger.error(f"Error loading preprocessed data: {str(e)}")
        raise

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