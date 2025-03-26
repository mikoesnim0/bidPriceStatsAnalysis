"""
Simple MongoDB connection test.
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mongodb_handler import MongoDBHandler

def main():
    # 데이터베이스와 컬렉션 이름 체크
    db_name = 'data_preprocessed'
    collections_to_check = [
        'preprocessed_dataset3_train',
        'preprocessed_dataset3_test',
        'preprocessed_dataset2_train',
        'preprocessed_dataset2_test'
    ]
    
    with MongoDBHandler(db_name=db_name) as mongo:
        print(f'Connected to {mongo.db_name}')
        
        all_collections = mongo.db.list_collection_names()
        print(f'All collections in {db_name}: {all_collections}')
        print("\n")
        
        # 각 컬렉션 확인
        for collection_name in collections_to_check:
            if collection_name in all_collections:
                count = mongo.db[collection_name].count_documents({})
                print(f'{collection_name}: {count} documents')
                
                if count > 0:
                    first_doc = mongo.db[collection_name].find_one({})
                    print(f'  First document keys: {list(first_doc.keys())}')
                    
                    # Check for target columns
                    target_cols = [k for k in first_doc.keys() if k.startswith('100_')]
                    print(f'  Target columns (100_): {target_cols}')
                    
                    # 다른 접두사 확인 (010_, 020_, 050_)
                    for prefix in ['010_', '020_', '050_']:
                        prefix_cols = [k for k in first_doc.keys() if k.startswith(prefix)]
                        if prefix_cols:
                            print(f'  Target columns ({prefix}): {prefix_cols}')
            else:
                print(f'{collection_name}: Not found in database')
        
        print("\n")
        
        # 비어있지 않은 컬렉션 찾기
        print("Non-empty collections:")
        for collection_name in all_collections:
            count = mongo.db[collection_name].count_documents({})
            if count > 0:
                print(f'  {collection_name}: {count} documents')

if __name__ == "__main__":
    main() 