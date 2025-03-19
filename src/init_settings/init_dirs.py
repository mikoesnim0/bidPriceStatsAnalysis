import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def init_directories():
    """
    Initialize the necessary directories for the project.
    """
    logger.info("Initializing project directories...")
    
    # Define required directories
    directories = [
        "data",
        "data/raw",
        "data/processed",
        "data/interim",
        "models",
        "results",
        "intervals"  # For storing interval files
    ]
    
    # Create directories if they don't exist
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"✅ Directory '{directory}' checked/created")
    
    logger.info("✅ All directories initialized successfully")

if __name__ == "__main__":
    init_directories() 