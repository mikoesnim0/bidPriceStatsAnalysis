import os
import logging
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import numpy as np
import urllib.parse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# MongoDB connection details
# URI 형식: mongodb://[username:password@]host:port/database[?options]
# URI에 DB 이름이 포함된 경우와 MONGO_DB 환경변수가 다를 경우 주의
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
MONGO_DB = os.getenv('MONGO_DB', 'gfcon')
MONGO_COLLECTION_PREFIX = os.getenv('MONGO_COLLECTION_PREFIX', 'preprocessed')

# 백업 인증 정보 (URI에 포함되지 않은 경우 사용)
MONGODB_USER = os.getenv('MONGODB_USER')
MONGODB_PASSWORD = os.getenv('MONGODB_PASSWORD')
MONGODB_AUTH_SOURCE = os.getenv('MONGODB_AUTH_SOURCE', 'admin')

class MongoDBHandler:
    """
    A class to handle MongoDB operations for bid price analysis project.
    """
    
    def __init__(self, uri=None, db_name=None, collection_prefix=None, username=None, password=None, auth_source=None):
        """
        Initialize the MongoDB handler.
        
        Parameters:
            uri (str, optional): MongoDB connection URI. Defaults to env variable.
            db_name (str, optional): Database name. Defaults to env variable.
            collection_prefix (str, optional): Collection name prefix. Defaults to env variable.
            username (str, optional): MongoDB username if not in URI. Defaults to env variable.
            password (str, optional): MongoDB password if not in URI. Defaults to env variable.
            auth_source (str, optional): Authentication source. Defaults to 'admin'.
        """
        self.uri = uri or MONGO_URI
        self.db_name = db_name or MONGO_DB
        self.collection_prefix = collection_prefix or MONGO_COLLECTION_PREFIX
        self.username = username or MONGODB_USER
        self.password = password or MONGODB_PASSWORD
        self.auth_source = auth_source or MONGODB_AUTH_SOURCE
        self.client = None
        self.db = None
        
        # URI에 인증 정보가 없고, 별도로 제공된 경우 URI 생성
        if self.username and self.password and '@' not in self.uri:
            # URL 인코딩하여 특수문자 처리
            encoded_password = urllib.parse.quote_plus(self.password)
            parts = self.uri.split('://')
            if len(parts) == 2:
                protocol, address = parts
                if '/' in address:  # 호스트 부분과 DB 부분이 있는 경우
                    host_part, db_part = address.split('/', 1)
                    self.uri = f"{protocol}://{self.username}:{encoded_password}@{host_part}/{db_part}"
                    if '?' not in self.uri:
                        self.uri += f"?authSource={self.auth_source}"
                    elif 'authSource=' not in self.uri:
                        self.uri += f"&authSource={self.auth_source}"
                else:  # 호스트 부분만 있는 경우
                    self.uri = f"{protocol}://{self.username}:{encoded_password}@{address}/{self.db_name}?authSource={self.auth_source}"
        
        logger.info(f"🔗 MongoDB URI configured (sanitized): {self._sanitize_uri(self.uri)}")
    
    def _sanitize_uri(self, uri):
        """비밀번호를 가리고 URI 반환"""
        if '@' in uri:
            parts = uri.split('@')
            auth_part = parts[0].split('://')
            if len(auth_part) > 1 and ':' in auth_part[1]:
                # 비밀번호 부분 마스킹
                username_part, _ = auth_part[1].split(':', 1)
                return f"{auth_part[0]}://{username_part}:****@{parts[1]}"
        return uri
    
    def connect(self):
        """
        Connect to MongoDB.
        
        Returns:
            self: For method chaining.
        """
        try:
            # 연결 시 타임아웃 설정 추가
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            
            # 실제 연결 테스트 (이때 인증 실패가 발생할 수 있음)
            self.client.server_info()
            
            # 연결 성공 후 DB 객체 획득
            self.db = self.client[self.db_name]
            
            logger.info(f"✅ Connected to MongoDB database: {self.db_name}")
            return self
        except Exception as e:
            # 보다 구체적인 오류 메시지 제공
            if "Authentication failed" in str(e):
                logger.error(f"❌ MongoDB 인증 실패: {e}")
                logger.error("❌ .env 파일의 사용자명/비밀번호를 확인하세요.")
            elif "ServerSelectionTimeoutError" in str(e.__class__):
                logger.error(f"❌ MongoDB 서버 연결 실패: {e}")
                logger.error("❌ MongoDB가 실행 중인지, 서버 주소가 올바른지 확인하세요.")
            else:
                logger.error(f"❌ MongoDB 연결 중 오류 발생: {e}")
            
            raise
    
    def close(self):
        """
        Close the MongoDB connection.
        """
        if self.client:
            self.client.close()
            logger.info("✅ MongoDB connection closed")
    
    def save_datasets(self, dataset_dict, use_single_collection=False):
        """
        Save datasets to MongoDB.
        
        Parameters:
            dataset_dict (dict): Dictionary of datasets to save.
            use_single_collection (bool): Whether to use a single collection for all datasets.
            
        Returns:
            dict: Collection names where datasets were saved.
        """
        if self.db is None:
            logger.error("❌ Not connected to MongoDB")
            raise ConnectionError("❌ Not connected to MongoDB")
        
        collection_names = {}
        
        try:
            if use_single_collection:
                # 단일 컬렉션 접근법: preprocessed 컬렉션 하나에 모든 데이터셋 저장
                collection_name = self.collection_prefix
                collection = self.db[collection_name]
                
                for key, df in dataset_dict.items():
                    # 데이터프레임 전처리 (MongoDB 호환성)
                    df = self._prepare_for_mongodb(df)
                    
                    # 각 레코드에 데이터셋 식별자 추가
                    records = df.to_dict('records')
                    for record in records:
                        record['dataset_type'] = key
                    
                    # 배치 처리로 삽입
                    BATCH_SIZE = 1000
                    for i in range(0, len(records), BATCH_SIZE):
                        batch = records[i:i+BATCH_SIZE]
                        try:
                            # 이미 존재하는 문서는 업데이트, 없으면 삽입
                            for doc in batch:
                                collection.update_one(
                                    {'공고번호': doc['공고번호'], 'dataset_type': doc['dataset_type']},
                                    {'$set': doc},
                                    upsert=True
                                )
                        except Exception as e:
                            logger.error(f"❌ 배치 저장 중 오류 발생: {e}")
                            continue
                    
                    collection_names[key] = collection_name
                    logger.info(f"✅ {key} 데이터셋을 {collection_name}에 업데이트 완료")
            else:
                # 다중 컬렉션 접근법: 각 데이터셋을 별도 컬렉션에 저장
                for key, df in dataset_dict.items():
                    # 데이터프레임 전처리 (MongoDB 호환성)
                    df = self._prepare_for_mongodb(df)
                    
                    # Format collection name with an underscore instead of a prefix
                    # preprocessed_dataset_3 -> preprocessed_3
                    dataset_number = key.split('_')[-1].lower()
                    collection_name = f"{self.collection_prefix}_{dataset_number}"
                    collection = self.db[collection_name]
                    
                    # 레코드 변환
                    records = df.to_dict('records')
                    
                    # 배치 처리로 삽입
                    BATCH_SIZE = 1000
                    for i in range(0, len(records), BATCH_SIZE):
                        batch = records[i:i+BATCH_SIZE]
                        try:
                            # 기존 컬렉션 유지하며 업데이트
                            for doc in batch:
                                collection.update_one(
                                    {'공고번호': doc['공고번호']},
                                    {'$set': doc},
                                    upsert=True
                                )
                        except Exception as e:
                            logger.error(f"❌ 배치 저장 중 오류 발생: {e}")
                            continue
                    
                    collection_names[key] = collection_name
                    logger.info(f"✅ {len(records)} 레코드를 {collection_name}에 저장 완료")
            
            return collection_names
        except Exception as e:
            logger.error(f"❌ MongoDB 저장 중 오류 발생: {e}")
            raise
    
    def _prepare_for_mongodb(self, df):
        """MongoDB에 저장 가능한 형태로 DataFrame 전처리"""
        df = df.copy()
        
        # 컬럼명에서 점(.) 제거 (MongoDB 필드명으로 사용 불가)
        df.columns = [col.replace('.', '_') for col in df.columns]
        
        # NumPy 배열을 리스트로 변환
        for col in df.columns:
            # NumPy 배열 검사 및 변환
            if df[col].apply(lambda x: isinstance(x, np.ndarray)).any():
                df[col] = df[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        
        return df
    
    def load_datasets(self, collection_names):
        """
        Load datasets from MongoDB.
        
        Parameters:
            collection_names (dict): Dictionary mapping dataset keys to collection names.
            
        Returns:
            dict: Dictionary of datasets loaded from MongoDB.
        """
        if self.db is None:
            logger.error("❌ Not connected to MongoDB")
            raise ConnectionError("❌ Not connected to MongoDB")
        
        dataset_dict = {}
        
        try:
            for key, collection_name in collection_names.items():
                collection = self.db[collection_name]
                cursor = collection.find({}, {'_id': 0})  # Exclude MongoDB _id field
                df = pd.DataFrame(list(cursor))
                
                dataset_dict[key] = df
                logger.info(f"✅ Loaded {len(df)} records from {collection_name}")
            
            return dataset_dict
        except Exception as e:
            logger.error(f"❌ Failed to load data from MongoDB: {e}")
            raise
    
    def get_default_collection_names(self):
        """
        데이터셋 키에 따른 기본 컬렉션 이름 매핑을 반환합니다.
        
        Returns:
            dict: Default collection names.
        """
        # 데이터베이스 이름에 따라 형식 변경
        if self.db_name and 'data_preprocessed' in self.db_name.lower():
            # data_preprocessed 데이터베이스에서 사용하는 컬렉션 이름 형식
            return {
                "dataset2": "preprocessed_dataset2_train",
                "dataset3": "preprocessed_dataset3_train", 
                "datasetetc": "preprocessed_datasetetc_train",
                "DataSet_2": "preprocessed_dataset2_train",
                "DataSet_3": "preprocessed_dataset3_train",
                "DataSet_etc": "preprocessed_datasetetc_train",
                "2": "preprocessed_dataset2_train",
                "3": "preprocessed_dataset3_train",
                "etc": "preprocessed_datasetetc_train"
            }
        else:
            # 기존 gfcon 데이터베이스에서 사용하는 컬렉션 이름 형식
            return {
                "DataSet_3": f"{self.collection_prefix}_3",
                "DataSet_2": f"{self.collection_prefix}_2",
                "DataSet_etc": f"{self.collection_prefix}_etc",
                "dataset3": f"{self.collection_prefix}_3",
                "dataset2": f"{self.collection_prefix}_2",
                "datasetetc": f"{self.collection_prefix}_etc",
                "3": f"{self.collection_prefix}_3",
                "2": f"{self.collection_prefix}_2",
                "etc": f"{self.collection_prefix}_etc"
            }
    
    def __enter__(self):
        """
        Support for context manager protocol.
        """
        return self.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Support for context manager protocol.
        """
        self.close()