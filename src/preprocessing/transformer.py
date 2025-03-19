import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)

class DataTransformer:
    """데이터 변환을 위한 클래스"""
    
    def __init__(self):
        self.scalers = {}  # 스케일러 저장
    
    def transform_dataset(self, df, dataset_name):
        """
        데이터셋 변환 수행
        
        Parameters:
            df (DataFrame): 변환할 데이터프레임
            dataset_name (str): 데이터셋 이름
            
        Returns:
            DataFrame: 변환된 데이터프레임
        """
        logger.info(f"🔄 {dataset_name} 데이터셋 변환 시작")
        
        # 1. 로그 변환 (금액 컬럼 등)
        df = self._apply_log_transform(df)
        
        # 2. 스케일링 (정규화/표준화)
        df = self._apply_scaling(df, dataset_name)
        
        # 3. 인코딩 (범주형 변수)
        df = self._encode_categorical_features(df)
        
        logger.info(f"✅ {dataset_name} 데이터셋 변환 완료")
        return df
    
    def _apply_log_transform(self, df):
        """금액 관련 컬럼에 로그 변환 적용"""
        # 금액 관련 컬럼 식별
        amount_columns = []
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            if any(keyword in col for keyword in ['금액', 'price', 'amount', '비용']):
                # 0 이하 값이 있는지 확인 (로그 변환을 위해)
                if (df[col] <= 0).any():
                    # 0 이하 값이 있으면 모든 값에 작은 값을 더해 양수로 만듦
                    min_val = abs(df[col].min()) + 1 if df[col].min() <= 0 else 0
                    logger.info(f"⚠️ '{col}' 컬럼에 0 이하 값이 있어 {min_val} 추가")
                    df[f"norm_log_{col}"] = np.log1p(df[col] + min_val)
                else:
                    df[f"norm_log_{col}"] = np.log1p(df[col])
                
                amount_columns.append(col)
                logger.info(f"✅ '{col}' 컬럼에 로그 변환 적용 -> 'norm_log_{col}'")
        
        return df
    
    def _apply_scaling(self, df, dataset_name):
        """수치형 컬럼에 스케일링 적용"""
        # 스케일링할 컬럼 선택 (norm_log_ 접두사 포함)
        scaling_cols = [
            col for col in df.select_dtypes(include=['int64', 'float64']).columns
            if not any(keyword in col for keyword in ['id', 'ID', '번호', 'index'])
        ]
        
        if scaling_cols:
            # MinMaxScaler 적용 (0~1 범위로 정규화)
            scaler = MinMaxScaler()
            
            # 결측치가 있는 경우 대체
            df_scaled = df.copy()
            for col in scaling_cols:
                if df[col].isna().any():
                    df_scaled[col] = df[col].fillna(df[col].median())
            
            # 스케일링 적용
            df_scaled[scaling_cols] = scaler.fit_transform(df_scaled[scaling_cols])
            
            # 스케일러 저장 (나중에 테스트 데이터 변환에 사용)
            self.scalers[dataset_name] = {
                'scaler': scaler,
                'columns': scaling_cols
            }
            
            logger.info(f"✅ {len(scaling_cols)}개 수치형 컬럼에 MinMaxScaler 적용")
            
            return df_scaled
        
        return df
    
    def _encode_categorical_features(self, df):
        """범주형 변수 인코딩"""
        # 범주형 컬럼 (문자열 타입) 선택
        cat_columns = df.select_dtypes(include=['object']).columns
        
        if len(cat_columns) > 0:
            logger.info(f"🔄 {len(cat_columns)}개 범주형 컬럼 인코딩")
            
            # 원-핫 인코딩 적용 (범주 수가 적은 컬럼에만)
            for col in cat_columns:
                unique_values = df[col].nunique()
                
                # 범주 수가 10개 이하인 경우에만 원-핫 인코딩 적용
                if 2 <= unique_values <= 10:
                    # 원-핫 인코딩
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                    df = pd.concat([df, dummies], axis=1)
                    logger.info(f"✅ '{col}' 컬럼에 원-핫 인코딩 적용 (범주 수: {unique_values})")
        
        return df 