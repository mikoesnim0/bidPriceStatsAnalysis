import os
import logging
import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from data_loader import DataLoader
from preprocessing.cleaner import DataCleaner
from preprocessing.transformer import DataTransformer
from preprocessing.feature_eng import FeatureEngineer
from mongodb_handler import MongoDBHandler
import argparse
import json

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

class PreprocessingPipeline:
    """데이터 전처리 파이프라인"""
    
    def __init__(self, data_dir=None, config=None):
        """
        초기화 함수
        
        Parameters:
            data_dir (str): 데이터 디렉토리 경로
            config (dict): 전처리 설정 (파일별 전처리 방법 정의)
        """
        self.data_dir = data_dir or os.getenv('DATA_DIR', './data')
        self.config = config or self._default_config()
        
        # 구성 요소 초기화
        self.data_loader = DataLoader(data_dir=self.data_dir)
        self.cleaner = DataCleaner()
        self.transformer = DataTransformer()
        self.feature_engineer = FeatureEngineer()
        self.mongodb_handler = MongoDBHandler()
        
        logger.info("🚀 전처리 파이프라인 초기화 완료")
    
    def _default_config(self):
        """
        기본 전처리 설정 반환
        
        Returns:
            dict: 파일별 전처리 설정
        """
        return {
            # 파일 이름 패턴별 전처리 설정
            "file_patterns": {
                "bid_data_*.csv": {
                    "log_transform_columns": ["기초금액", "예정금액", "예가", "투찰가"],
                    "normalize_columns": ["기초금액", "예정금액", "예가", "투찰가", "norm_log_기초금액", "norm_log_예정금액"],
                    "categorical_columns": ["공고종류", "업종", "낙찰방법"],
                    "text_columns": ["공고제목", "공고내용"],
                    "key_column": "공고번호",
                    "date_columns": ["입찰일자", "개찰일시"],
                    "target_columns": ["거래적정성", "낙찰가격비율"],
                    "required_columns": ["공고번호", "기초금액", "예정금액"]
                },
                "notice_data_*.csv": {
                    "log_transform_columns": ["기초금액", "예정금액"],
                    "normalize_columns": ["기초금액", "예정금액", "norm_log_기초금액", "norm_log_예정금액"],
                    "categorical_columns": ["공고종류", "업종", "낙찰방법"],
                    "text_columns": ["공고제목", "공고내용"],
                    "key_column": "공고번호",
                    "date_columns": ["입찰일자", "개찰일시"],
                    "required_columns": ["공고번호", "기초금액"]
                }
            },
            # 데이터셋 유형별 설정 (파일명 기반으로 식별)
            "dataset_types": {
                "DataSet_3": {
                    "description": "3개 입찰건 이상 참여 업체 데이터",
                    "priority": 1
                },
                "DataSet_2": {
                    "description": "2개 입찰건 참여 업체 데이터",
                    "priority": 2
                },
                "DataSet_etc": {
                    "description": "기타 데이터",
                    "priority": 3
                }
            },
            # 고급 전처리 설정
            "advanced_features": {
                "enabled": True,     # 고급 전처리 사용 여부
                "pca_enabled": True,  # PCA 차원 축소 사용 여부
                "embedding_enabled": True,  # 텍스트 임베딩 사용 여부
                "feature_selection_enabled": True  # 특성 선택 사용 여부
            }
        }
    
    def run(self, file_pattern=None, save_to_mongodb=True, custom_preprocessing=None, advanced_features=None):
        """
        전처리 파이프라인 실행
        
        Parameters:
            file_pattern (str): 파일 패턴 (예: '*.csv')
            save_to_mongodb (bool): MongoDB에 저장 여부
            custom_preprocessing (callable): 추가 전처리 함수
            advanced_features (bool): 고급 전처리 기능 사용 여부(설정보다 우선함)
            
        Returns:
            dict: 전처리된 데이터셋 딕셔너리
        """
        start_time = time.time()
        logger.info("🚀 전처리 파이프라인 실행 시작")
        
        # 고급 전처리 설정 초기화
        use_advanced = advanced_features if advanced_features is not None else self.config.get('advanced_features', {}).get('enabled', False)
        logger.info(f"🔍 고급 전처리 사용: {'✅ 활성화' if use_advanced else '❌ 비활성화'}")
        
        # 1. 데이터 로드
        logger.info("📂 데이터 로드 단계")
        dataset_dict = self.data_loader.load_raw_data(file_pattern)
        
        if not dataset_dict:
            logger.error("❌ 데이터 로드 실패: 데이터셋이 비어 있습니다.")
            return {}
        
        # 데이터셋 정보 출력
        logger.info(f"✅ {len(dataset_dict)}개 데이터셋 로드 완료:")
        for name, df in dataset_dict.items():
            logger.info(f"   - {name}: {df.shape[0]} 행 x {df.shape[1]} 열")
        
        # 2. 데이터 정제
        logger.info("\n🧹 데이터 정제 단계")
        cleaned_dict = {}
        for name, df in dataset_dict.items():
            # 파일 유형 기반 설정 찾기
            preprocessing_config = self._get_preprocessing_config(name)
            if preprocessing_config:
                logger.info(f"🔍 '{name}' 데이터셋에 맞춤형 전처리 적용")
                # 필수 컬럼 확인
                if preprocessing_config.get("required_columns"):
                    self._validate_required_columns(df, preprocessing_config["required_columns"], name)
            
            # 기본 정제 적용
            cleaned_dict[name] = self.cleaner.clean_dataset(df, name)
        
        # 3. 데이터 변환 (파일 유형별 맞춤 설정 적용)
        logger.info("\n🔄 데이터 변환 단계")
        transformed_dict = {}
        for name, df in cleaned_dict.items():
            # 파일 유형 기반 설정 찾기
            preprocessing_config = self._get_preprocessing_config(name)
            
            # 맞춤형 로그 변환 및 정규화 적용
            if preprocessing_config:
                # 로그 변환 적용
                if preprocessing_config.get("log_transform_columns"):
                    df = self._apply_log_transform(df, preprocessing_config["log_transform_columns"], name)
                
                # 정규화 적용
                if preprocessing_config.get("normalize_columns"):
                    # 기본 변환 프로세스 진행
                    df = self.transformer.transform_dataset(df, name)
                    
                    # 날짜 컬럼 처리
                    if preprocessing_config.get("date_columns"):
                        df = self._process_date_columns(df, preprocessing_config["date_columns"])
                else:
                    # 기본 변환 프로세스 진행
                    df = self.transformer.transform_dataset(df, name)
            else:
                # 기본 변환 프로세스 진행
                df = self.transformer.transform_dataset(df, name)
            
            transformed_dict[name] = df
        
        # 4. 특성 엔지니어링 (파일 의존적)
        logger.info("\n🔧 특성 엔지니어링 단계")
        engineered_dict = {}
        for name, df in transformed_dict.items():
            # 파일 유형 기반 설정 찾기
            preprocessing_config = self._get_preprocessing_config(name)
            
            # 맞춤형 특성 엔지니어링 적용
            if preprocessing_config:
                # 텍스트 특성 추출
                if preprocessing_config.get("text_columns"):
                    logger.info(f"📝 '{name}' 데이터셋의 텍스트 특성 처리 중...")
                
                # 범주형 특성 처리
                if preprocessing_config.get("categorical_columns"):
                    logger.info(f"🏷️ '{name}' 데이터셋의 범주형 특성 처리 중...")
                
                # 타겟 컬럼 계산
                if preprocessing_config.get("target_columns"):
                    logger.info(f"🎯 '{name}' 데이터셋의 타겟 특성 계산 중...")
            
            # 고급 특성 엔지니어링 적용
            df = self.feature_engineer.engineer_features(df, name, use_advanced)
            
            # 사용자 정의 전처리 적용 (있는 경우)
            if custom_preprocessing and callable(custom_preprocessing):
                logger.info(f"🛠️ '{name}' 데이터셋에 사용자 정의 전처리 적용 중...")
                df = custom_preprocessing(df, name, preprocessing_config)
            
            engineered_dict[name] = df
        
        # 5. 입찰가 분석 특화 처리 (필요한 경우)
        logger.info("\n💰 입찰가 분석 특화 처리 단계")
        for name, df in engineered_dict.items():
            preprocessing_config = self._get_preprocessing_config(name)
            if preprocessing_config and "기초금액" in df.columns and "예정금액" in df.columns:
                # 가격비율 계산
                if "투찰가" in df.columns:
                    df["낙찰가격비율"] = df["투찰가"] / df["예정금액"]
                    logger.info(f"✅ '{name}' 데이터셋의 낙찰가격비율 계산 완료")
                
                # 기타 입찰가 관련 특성 생성
                self._create_bid_price_features(df, name)
            
            engineered_dict[name] = df
            
            # 데이터셋 크기 및 특성 정보 출력
            logger.info(f"✅ '{name}' 데이터셋 처리 완료: {df.shape[0]}행 x {df.shape[1]}열")
            # 상위 20개 컬럼 이름 출력 (중요 컬럼 확인용)
            col_sample = list(df.columns[:20])
            logger.info(f"   샘플 컬럼(20개): {col_sample}")
        
        # 6. MongoDB에 저장 (선택 사항)
        if save_to_mongodb:
            logger.info("\n💾 MongoDB에 저장 단계")
            try:
                with self.mongodb_handler as mongo:
                    collection_names = mongo.save_datasets(engineered_dict)
                    logger.info(f"✅ {len(collection_names)}개 데이터셋을 MongoDB에 저장 완료")
                    for key, coll in collection_names.items():
                        logger.info(f"   - {key} -> {coll}")
            except Exception as e:
                logger.error(f"❌ MongoDB 저장 중 오류 발생: {e}")
        
        # 처리 시간 기록
        elapsed_time = time.time() - start_time
        logger.info(f"\n✅ 전처리 파이프라인 실행 완료 (소요 시간: {elapsed_time:.2f}초)")
        
        return engineered_dict
    
    def _get_preprocessing_config(self, dataset_name):
        """
        데이터셋 이름에 따른 전처리 설정 반환
        
        Parameters:
            dataset_name (str): 데이터셋 이름
            
        Returns:
            dict: 전처리 설정
        """
        # 데이터셋 이름에서 파일 패턴 추출
        import re
        
        # 기본 설정
        default_config = None
        
        # 파일 패턴 매칭
        for pattern, config in self.config["file_patterns"].items():
            pattern_regex = pattern.replace("*", ".*").replace(".", "\.")
            if re.search(pattern_regex, dataset_name, re.IGNORECASE):
                return config
        
        # 데이터셋 유형별 설정
        for dataset_type, type_config in self.config["dataset_types"].items():
            if dataset_type in dataset_name:
                # 만약 파일 패턴이 없지만 데이터셋 유형은 일치하는 경우
                if default_config is None:
                    # 첫 번째 파일 패턴 설정을 기본값으로 사용
                    default_config = next(iter(self.config["file_patterns"].values()), {})
        
        return default_config
    
    def _validate_required_columns(self, df, required_columns, dataset_name):
        """
        필수 컬럼 존재 여부 확인
        
        Parameters:
            df (DataFrame): 데이터프레임
            required_columns (list): 필수 컬럼 목록
            dataset_name (str): 데이터셋 이름
            
        Raises:
            ValueError: 필수 컬럼이 없는 경우
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"❌ '{dataset_name}' 데이터셋에 필수 컬럼이 없습니다: {missing_columns}")
            raise ValueError(f"필수 컬럼 누락: {missing_columns}")
        
        logger.info(f"✅ '{dataset_name}' 데이터셋의 필수 컬럼 검증 완료")
    
    def _apply_log_transform(self, df, columns, dataset_name):
        """
        지정된 컬럼에 로그 변환 적용
        
        Parameters:
            df (DataFrame): 데이터프레임
            columns (list): 로그 변환할 컬럼 목록
            dataset_name (str): 데이터셋 이름
            
        Returns:
            DataFrame: 로그 변환이 적용된 데이터프레임
        """
        for col in columns:
            if col in df.columns:
                try:
                    # 음수 값 처리
                    if (df[col] <= 0).any():
                        min_val = abs(df[col].min()) + 1 if df[col].min() <= 0 else 0
                        logger.info(f"⚠️ '{col}' 컬럼에 0 이하 값이 있어 {min_val} 추가")
                        df[f"norm_log_{col}"] = np.log1p(df[col] + min_val)
                    else:
                        df[f"norm_log_{col}"] = np.log1p(df[col])
                    
                    logger.info(f"✅ '{dataset_name}' 데이터셋의 '{col}' 컬럼에 로그 변환 적용 -> 'norm_log_{col}'")
                except Exception as e:
                    logger.error(f"❌ '{col}' 컬럼 로그 변환 중 오류 발생: {e}")
        
        return df
    
    def _process_date_columns(self, df, date_columns):
        """
        날짜 컬럼 처리
        
        Parameters:
            df (DataFrame): 데이터프레임
            date_columns (list): 날짜 컬럼 목록
            
        Returns:
            DataFrame: 날짜 처리된 데이터프레임
        """
        for col in date_columns:
            if col in df.columns:
                try:
                    # 날짜 형식 변환
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    
                    # 연도, 월, 일 추출
                    df[f"{col}_year"] = df[col].dt.year
                    df[f"{col}_month"] = df[col].dt.month
                    df[f"{col}_day"] = df[col].dt.day
                    
                    # 요일 추출 (0: 월요일, 6: 일요일)
                    df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                    
                    # 분기 추출
                    df[f"{col}_quarter"] = df[col].dt.quarter
                    
                    logger.info(f"✅ '{col}' 컬럼 날짜 특성 추출 완료")
                except Exception as e:
                    logger.error(f"❌ '{col}' 컬럼 날짜 처리 중 오류 발생: {e}")
        
        return df
    
    def _create_bid_price_features(self, df, dataset_name):
        """
        입찰가 관련 특성 생성
        
        Parameters:
            df (DataFrame): 데이터프레임
            dataset_name (str): 데이터셋 이름
            
        Returns:
            DataFrame: 입찰가 특성이 추가된 데이터프레임
        """
        try:
            # 기초금액 대비 예정금액 비율
            if "기초금액" in df.columns and "예정금액" in df.columns:
                # 0으로 나누는 것 방지
                df["예정가비율"] = df["예정금액"] / df["기초금액"].replace(0, np.nan)
                df["예정가비율"] = df["예정가비율"].fillna(0)
                
                # 로그 변환된 값 사이의 비율도 계산
                if "norm_log_기초금액" in df.columns and "norm_log_예정금액" in df.columns:
                    df["log_예정가비율"] = df["norm_log_예정금액"] / df["norm_log_기초금액"].replace(0, np.nan)
                    df["log_예정가비율"] = df["log_예정가비율"].fillna(0)
            
            # 추가 가격 비율 특성들
            if "투찰가" in df.columns:
                if "기초금액" in df.columns:
                    df["투찰가_기초금액비율"] = df["투찰가"] / df["기초금액"].replace(0, np.nan)
                    df["투찰가_기초금액비율"] = df["투찰가_기초금액비율"].fillna(0)
                
                if "예정금액" in df.columns:
                    df["투찰가_예정금액비율"] = df["투찰가"] / df["예정금액"].replace(0, np.nan)
                    df["투찰가_예정금액비율"] = df["투찰가_예정금액비율"].fillna(0)
            
            logger.info(f"✅ '{dataset_name}' 데이터셋의 입찰가 관련 특성 생성 완료")
            
            return df
        except Exception as e:
            logger.error(f"❌ 입찰가 관련 특성 생성 중 오류 발생: {e}")
            return df

def get_args():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description='입찰가 분석용 데이터 전처리 파이프라인 실행')
    parser.add_argument('--file-pattern', type=str, default='*.csv',
                        help='처리할 파일 패턴 (예: *.csv)')
    parser.add_argument('--no-mongo', action='store_true',
                        help='MongoDB에 저장하지 않음')
    parser.add_argument('--advanced-features', action='store_true',
                        help='고급 전처리 기능 사용 (PCA, 임베딩, 특성 선택 등)')
    return parser.parse_args()

if __name__ == "__main__":
    # 명령줄 인수 파싱
    args = get_args()
    
    # 파이프라인 실행
    pipeline = PreprocessingPipeline()
    processed_data = pipeline.run(
        file_pattern=args.file_pattern,
        save_to_mongodb=not args.no_mongo,
        advanced_features=args.advanced_features
    )
    
    if processed_data:
        # 처리된 데이터셋 정보 출력
        logger.info("\n📊 처리된 데이터셋 통계:")
        for name, df in processed_data.items():
            logger.info(f"\n{name}:")
            logger.info(f"  - 형태: {df.shape[0]} 행 x {df.shape[1]} 열")
            logger.info(f"  - 컬럼: {', '.join(df.columns[:5])}... 외 {max(0, len(df.columns)-5)}개")
            logger.info(f"  - 메모리 사용량: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB") 