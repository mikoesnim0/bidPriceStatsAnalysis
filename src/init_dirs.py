"""
Directory initialization for the BidPrice project.
"""
import os
from src.config import MODELS_DIR, RESULTS_DIR

def init_directories():
    """
    Initialize all required directories for the project.
    Creates models and results directories if they don't exist.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Create subdirectories for different dataset models
    for dataset in ['dataset_3', 'dataset_2']:
        dataset_dir = os.path.join(MODELS_DIR, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
    
    # Create autogluon directory
    autogluon_dir = os.path.join(MODELS_DIR, 'autogluon')
    os.makedirs(autogluon_dir, exist_ok=True)
    
    print(f"Created directory structure for models and results.")

if __name__ == "__main__":
    init_directories() 