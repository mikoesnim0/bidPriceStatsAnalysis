import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    """데이터 정제를 위한 클래스"""
    
    def __init__(self):
        self.cleaning_stats = {}
    
    def clean_dataset(self, df, dataset_name):
        """
        데이터셋 정제 수행
        
        Parameters:
            df (DataFrame): 정제할 데이터프레임
            dataset_name (str): 데이터셋 이름
            
        Returns:
            DataFrame: 정제된 데이터프레임
        """
        logger.info(f"🧹 {dataset_name} 데이터셋 정제 시작")
        original_shape = df.shape
        
        # 정제 통계 초기화
        self.cleaning_stats[dataset_name] = {
            'original_rows': original_shape[0],
            'original_columns': original_shape[1],
            'missing_values_removed': 0,
            'duplicates_removed': 0
        }
        
        # 1. 결측치 처리
        df_cleaned = self._handle_missing_values(df, dataset_name)
        
        # 2. 중복 데이터 제거
        df_cleaned = self._remove_duplicates(df_cleaned, dataset_name)
        
        # 3. 이상치 처리
        df_cleaned = self._handle_outliers(df_cleaned, dataset_name)
        
        # 4. 데이터 타입 변환
        df_cleaned = self._convert_dtypes(df_cleaned)
        
        # 최종 데이터셋 크기 및 정제 통계 기록
        final_shape = df_cleaned.shape
        self.cleaning_stats[dataset_name]['final_rows'] = final_shape[0]
        self.cleaning_stats[dataset_name]['final_columns'] = final_shape[1]
        
        logger.info(f"✅ {dataset_name} 데이터셋 정제 완료")
        logger.info(f"   - 원본 크기: {original_shape[0]} 행 x {original_shape[1]} 열")
        logger.info(f"   - 정제 후 크기: {final_shape[0]} 행 x {final_shape[1]} 열")
        logger.info(f"   - 제거된 결측치 행: {self.cleaning_stats[dataset_name]['missing_values_removed']}")
        logger.info(f"   - 제거된 중복 행: {self.cleaning_stats[dataset_name]['duplicates_removed']}")
        
        return df_cleaned
    
    def _handle_missing_values(self, df, dataset_name):
        """결측치 처리"""
        # 결측치 확인
        missing_counts = df.isna().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            logger.info(f"⚠️ {dataset_name}에서 총 {total_missing}개의 결측치 발견")
            
            # 필수 컬럼에 결측치가 있는 행 제거
            # '공고번호'와 같은 필수 컬럼이 있다면 해당 컬럼에 결측치가 있는 행 제거
            essential_columns = ['공고번호']  # 필수 컬럼 목록
            essential_columns = [col for col in essential_columns if col in df.columns]
            
            if essential_columns:
                missing_in_essential = df[essential_columns].isna().any(axis=1)
                rows_to_drop = missing_in_essential.sum()
                
                if rows_to_drop > 0:
                    logger.info(f"⚠️ 필수 컬럼에 결측치가 있는 {rows_to_drop}개 행 제거")
                    df = df[~missing_in_essential]
                    self.cleaning_stats[dataset_name]['missing_values_removed'] += rows_to_drop
            
            # 나머지 결측치는 적절한 값으로 대체
            # 수치형 컬럼: 중앙값으로 대체
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                if df[col].isna().sum() > 0:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.info(f"   - '{col}' 컬럼의 결측치를 중앙값 {median_val}으로 대체")
            
            # 문자열 컬럼: 'Unknown'으로 대체
            string_cols = df.select_dtypes(include=['object']).columns
            for col in string_cols:
                if df[col].isna().sum() > 0:
                    df[col] = df[col].fillna('Unknown')
                    logger.info(f"   - '{col}' 컬럼의 결측치를 'Unknown'으로 대체")
        
        return df
    
    def _remove_duplicates(self, df, dataset_name):
        """중복 데이터 제거"""
        # '공고번호'와 같은 고유 식별자를 기준으로 중복 확인
        if '공고번호' in df.columns:
            duplicates = df.duplicated(subset=['공고번호'], keep='first')
            dup_count = duplicates.sum()
            
            if dup_count > 0:
                logger.info(f"⚠️ {dataset_name}에서 {dup_count}개의 중복 행 발견")
                df = df[~duplicates]
                self.cleaning_stats[dataset_name]['duplicates_removed'] = dup_count
        
        return df
    
    def _handle_outliers(self, df, dataset_name):
        """이상치 처리"""
        # 수치형 컬럼에 대해 IQR 방식으로 이상치 탐지 및 처리
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        for col in numeric_cols:
            # 금액이나 수치 데이터에 대해서만 이상치 처리 (예: '기초금액', '예정금액' 등)
            if any(keyword in col for keyword in ['금액', 'price', 'amount', '비용']):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 이상치 확인
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    logger.info(f"⚠️ '{col}' 컬럼에서 {outlier_count}개의 이상치 발견")
                    # 이상치를 경계값으로 대체
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound
                    logger.info(f"   - 이상치를 경계값으로 대체 ({lower_bound:.2f} ~ {upper_bound:.2f})")
        
        return df
    
    def _convert_dtypes(self, df):
        """데이터 타입 변환"""
        # 데이터 타입 최적화
        for col in df.columns:
            # 날짜 문자열을 datetime으로 변환
            if any(keyword in col.lower() for keyword in ['date', '날짜', '일자', '시작일', '종료일']):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        
        return df 