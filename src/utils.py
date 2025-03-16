"""
Utility functions for the BidPrice prediction model.
"""
import os
import logging
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, Optional, Union

from src.config import LOG_LEVEL, LOG_FILE

def setup_logger(name: str, level: str = LOG_LEVEL) -> logging.Logger:
    """
    Set up a logger with specified name and level.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    # Get the logging level from string
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Create handlers
    file_handler = logging.FileHandler(LOG_FILE)
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Set formatter for handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

def save_pickle(obj: Any, file_path: str) -> None:
    """
    Save object as pickle file.
    
    Args:
        obj: Object to save
        file_path: Path to save the pickle file
    """
    logger = setup_logger(__name__)
    logger.info(f"Saving pickle file to {file_path}")
    
    try:
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Successfully saved pickle file to {file_path}")
    except Exception as e:
        logger.error(f"Error saving pickle file: {str(e)}")
        raise

def load_pickle(file_path: str) -> Any:
    """
    Load object from pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Loaded object
    """
    logger = setup_logger(__name__)
    logger.info(f"Loading pickle file from {file_path}")
    
    try:
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Successfully loaded pickle file from {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Error loading pickle file: {str(e)}")
        raise

def save_json(obj: Dict[str, Any], file_path: str) -> None:
    """
    Save object as JSON file.
    
    Args:
        obj: Object to save
        file_path: Path to save the JSON file
    """
    logger = setup_logger(__name__)
    logger.info(f"Saving JSON file to {file_path}")
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, cls=NumpyEncoder, indent=4)
        logger.info(f"Successfully saved JSON file to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON file: {str(e)}")
        raise

def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load object from JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded object
    """
    logger = setup_logger(__name__)
    logger.info(f"Loading JSON file from {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        logger.info(f"Successfully loaded JSON file from {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Error loading JSON file: {str(e)}")
        raise

def timer(start_time: Optional[float] = None) -> Union[float, str]:
    """
    Timer utility for measuring elapsed time.
    
    Args:
        start_time: Start time in seconds (if None, current time will be returned)
        
    Returns:
        If start_time is None, returns current time
        If start_time is provided, returns formatted elapsed time string
    """
    from time import time
    
    if start_time is None:
        return time()
    
    elapsed_time = time() - start_time
    
    if elapsed_time < 60:
        return f"{elapsed_time:.2f} seconds"
    elif elapsed_time < 3600:
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        return f"{minutes} minutes {seconds:.2f} seconds"
    else:
        hours = int(elapsed_time // 3600)
        remaining = elapsed_time % 3600
        minutes = int(remaining // 60)
        seconds = remaining % 60
        return f"{hours} hours {minutes} minutes {seconds:.2f} seconds"

def generate_timestamp() -> str:
    """
    Generate a timestamp string for file naming.
    
    Returns:
        Timestamp string in format "YYYYMMDD_HHMMSS"
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_memory_usage(df: pd.DataFrame) -> str:
    """
    Get memory usage of DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Memory usage string
    """
    memory_usage = df.memory_usage(deep=True).sum()
    
    if memory_usage < 1024:
        return f"{memory_usage} bytes"
    elif memory_usage < 1024 ** 2:
        return f"{memory_usage / 1024:.2f} KB"
    elif memory_usage < 1024 ** 3:
        return f"{memory_usage / (1024 ** 2):.2f} MB"
    else:
        return f"{memory_usage / (1024 ** 3):.2f} GB"

def handle_exceptions(func):
    """
    Decorator for handling exceptions in functions.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    logger = setup_logger(__name__)
    
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    
    return wrapper 