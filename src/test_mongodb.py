import logging
import pandas as pd
from dotenv import load_dotenv
from statsModelsPredict.src.db_config.mongodb_handler import MongoDBHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_mongodb_connection():
    """
    Test connection to MongoDB.
    """
    logger.info("Testing MongoDB connection...")
    
    try:
        with MongoDBHandler() as mongo_handler:
            # Check connection with a simple ping command instead of listing databases
            # This avoids the need for admin privileges
            result = mongo_handler.db.command('ping')
            logger.info(f"✅ Connected to MongoDB database: {mongo_handler.db_name}")
            logger.info(f"✅ Ping result: {result}")
            
            try:
                # Try to list collections in the current database
                # This requires read privileges on the database
                collections = mongo_handler.db.list_collection_names()
                logger.info(f"✅ Collections in '{mongo_handler.db_name}': {collections}")
            except Exception as coll_e:
                # If listing collections fails, it might be due to permissions
                logger.warning(f"⚠️ Could not list collections: {coll_e}")
                logger.info("⚠️ This is likely a permissions issue. We'll continue with basic operations.")
            
            # Try to access a specific collection without listing all collections
            try:
                # Check if we can access the collection we'll be using
                collection_name = f"{mongo_handler.collection_prefix}_3"
                collection = mongo_handler.db[collection_name]
                # Try a simple count operation
                count = collection.count_documents({})
                logger.info(f"✅ Collection '{collection_name}' accessible, contains {count} documents")
            except Exception as coll_e:
                logger.warning(f"⚠️ Could not access collection: {coll_e}")
        
        logger.info("MongoDB connection test completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"❌ MongoDB connection test failed: {e}")
        return False

def create_test_data():
    """
    Create test data for MongoDB upload.
    """
    logger.info("Creating test data...")
    
    # Create simple test dataframes
    data_3 = {
        "공고번호": ["TEST001", "TEST002", "TEST003"],
        "norm_log_기초금액": [0.1, 0.2, 0.3],
        "UMAP_공고제목_1": [0.01, 0.02, 0.03]
    }
    
    data_2 = {
        "공고번호": ["TEST004", "TEST005"],
        "norm_log_기초금액": [0.4, 0.5],
        "UMAP_공고제목_1": [0.04, 0.05]
    }
    
    data_etc = {
        "공고번호": ["TEST006"],
        "norm_log_기초금액": [0.6],
        "UMAP_공고제목_1": [0.06]
    }
    
    # Create DataFrames
    df_3 = pd.DataFrame(data_3)
    df_2 = pd.DataFrame(data_2)
    df_etc = pd.DataFrame(data_etc)
    
    # Create dataset dictionary
    dataset_dict = {
        "DataSet_3": df_3,
        "DataSet_2": df_2,
        "DataSet_etc": df_etc
    }
    
    logger.info("Test data created successfully")
    return dataset_dict

def test_upload_data():
    """
    Test uploading data to MongoDB.
    """
    logger.info("Testing data upload to MongoDB...")
    
    try:
        # Create test data
        dataset_dict = create_test_data()
        
        # Upload to MongoDB
        with MongoDBHandler() as mongo_handler:
            collection_names = mongo_handler.save_datasets(dataset_dict)
            logger.info(f"✅ Data uploaded to collections: {collection_names}")
        
        logger.info("Data upload test completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"❌ Data upload test failed: {e}")
        return False

def test_retrieve_data():
    """
    Test retrieving data from MongoDB.
    """
    logger.info("Testing data retrieval from MongoDB...")
    
    try:
        with MongoDBHandler() as mongo_handler:
            # Get default collection names
            collection_names = mongo_handler.get_default_collection_names()
            logger.info(f"Default collection names: {collection_names}")
            
            # Retrieve data
            dataset_dict = mongo_handler.load_datasets(collection_names)
            
            # Check the data
            for key, df in dataset_dict.items():
                logger.info(f"✅ Retrieved {len(df)} records from {key}")
                if not df.empty:
                    logger.info(f"Sample data from {key}:\n{df.head(2)}")
        
        logger.info("Data retrieval test completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"❌ Data retrieval test failed: {e}")
        return False

if __name__ == "__main__":
    # Test MongoDB connection
    if test_mongodb_connection():
        # If connection successful, test upload
        if test_upload_data():
            # If upload successful, test retrieval
            test_retrieve_data() 