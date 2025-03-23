"""
Test script for prediction functionality.
"""
import os
import pandas as pd
import pickle
from mongodb_handler import MongoDBHandler

def get_test_data(n_samples=5):
    """
    Get test data from MongoDB preprocessed collection.
    
    Parameters:
        n_samples (int): Number of samples to get
        
    Returns:
        pd.DataFrame: Test data
    """
    with MongoDBHandler() as mongo_handler:
        # Connect to data_preprocessed database
        db = mongo_handler.client['data_preprocessed']
        collection = db['preprocessed_dataset3_test']
        
        # Get first n samples
        cursor = collection.find({}, {"_id": 0}).limit(n_samples)
        data = list(cursor)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        print(f"Retrieved {len(df)} samples from preprocessed_dataset3_test")
        print("\nColumns in the data:")
        print(df.columns.tolist())
        
        print("\nFirst row of data:")
        print(df.iloc[0])
        
        print("\nData types of columns:")
        print(df.dtypes)
        
        return df

def load_autogluon_model(model_dir, bin_label):
    """
    Load AutoGluon model
    
    Parameters:
        model_dir (str): Directory containing the model
        bin_label (str): Bin label (e.g. "001")
        
    Returns:
        object: Loaded model
    """
    import autogluon.tabular as ag
    
    # Path to the specific model bin
    model_path = os.path.join("/data/dev/bidPriceStatsAnalysis/models/autogluon/datasetetc/010", f"010_{bin_label}")
    
    # Load the predictor
    predictor = ag.TabularPredictor.load(model_path)
    
    return predictor

def predict_with_autogluon(df, target_prefix="010"):
    """
    Make predictions using AutoGluon models
    
    Parameters:
        df (pd.DataFrame): Input data
        target_prefix (str): Target prefix (e.g. "010", "020")
        
    Returns:
        pd.DataFrame: Predictions
    """
    # Get list of available bin models
    model_dir = os.path.join("models", "autogluon", "datasetetc", target_prefix)
    bin_dirs = [d for d in os.listdir(model_dir) if d.startswith(f"{target_prefix}_")]
    
    # Create results DataFrame
    results = pd.DataFrame(index=df.index)
    
    for bin_dir in bin_dirs:
        # Extract bin label (e.g. "001" from "010_001")
        bin_label = bin_dir.split("_")[1]
        
        try:
            # Load the model
            model = load_autogluon_model(model_dir, bin_label)
            
            # Make predictions
            predictions = model.predict(df)
            
            # Add to results
            results[f"{target_prefix}_{bin_label}"] = predictions
            
            print(f"Successfully predicted bin {bin_label} with {target_prefix}")
            
        except Exception as e:
            print(f"Error predicting bin {bin_label}: {e}")
    
    return results

def main():
    # Get test data
    test_data = get_test_data(5)
    
    try:
        # Try prediction with AutoGluon model
        predictions = predict_with_autogluon(test_data, target_prefix="010")
        
        print("\nPrediction successful!")
        print("\nPrediction columns:")
        print(predictions.columns.tolist())
        print("\nFirst row of predictions:")
        print(predictions.iloc[0])
        
    except Exception as e:
        print(f"\nError during prediction: {e}")
        raise

if __name__ == "__main__":
    main() 