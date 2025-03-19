import pandas as pd
import numpy as np
import logging
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler
import re
import nltk
from nltk.corpus import stopwords
import warnings
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# 경고 무시
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """특성 엔지니어링을 위한 클래스"""
    
    def __init__(self):
        self.models = {}  # 특성 엔지니어링 모델 저장
        
        # NLTK 리소스 다운로드 (첫 사용시)
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
        
        # BGE-M3 모델 초기화 변수
        self.bge_model = None
        self.bge_tokenizer = None
        self.device = None
    
    def engineer_features(self, df, dataset_name, advanced_features=True):
        """
        특성 엔지니어링 수행
        
        Parameters:
            df (DataFrame): 특성 엔지니어링할 데이터프레임
            dataset_name (str): 데이터셋 이름
            advanced_features (bool): 고급 특성 엔지니어링 적용 여부
            
        Returns:
            DataFrame: 특성 엔지니어링된 데이터프레임
        """
        logger.info(f"🔧 {dataset_name} 데이터셋 특성 엔지니어링 시작")
        
        # 1. 텍스트 피처 추출 (공고 제목 등)
        df = self._extract_text_features(df, dataset_name)
        
        if advanced_features:
            # 2. BGE-M3 임베딩 적용 (텍스트 데이터가 충분한 경우)
            df = self._apply_bge_embeddings(df, dataset_name)
            
            # 3. 차원 축소 (PCA)
            df = self._apply_dimension_reduction(df, dataset_name)
            
            # 4. 특성 선택
            df = self._select_best_features(df, dataset_name)
        
        # 5. 특성 조합 (새로운 특성 생성)
        df = self._create_combined_features(df)
        
        logger.info(f"✅ {dataset_name} 데이터셋 특성 엔지니어링 완료")
        return df
    
    def _extract_text_features(self, df, dataset_name):
        """텍스트 특성 추출 (TF-IDF)"""
        # 텍스트 컬럼 찾기 (제목, 설명 등)
        text_columns = []
        for col in df.select_dtypes(include=['object']).columns:
            if any(keyword in col for keyword in ['제목', 'title', '내용', 'content', '설명', '항목', '공고']):
                text_columns.append(col)
        
        if text_columns:
            for col in text_columns:
                # 결측치 처리
                df[col] = df[col].fillna('')
                
                # 최소 30개 이상의 행이 있을 때만 TF-IDF 적용
                if len(df) >= 30 and df[col].str.len().sum() > 100:
                    # 전처리 함수
                    def preprocess_text(text):
                        if not isinstance(text, str):
                            return ""
                        # 특수문자 제거 및 소문자 변환
                        text = re.sub(r'[^\w\s]', ' ', text)
                        return text.lower()
                    
                    # 텍스트 전처리
                    df[f"{col}_preprocessed"] = df[col].apply(preprocess_text)
                    
                    # TF-IDF 벡터화
                    tfidf = TfidfVectorizer(
                        max_features=10,  # 상위 10개 특성으로 증가
                        min_df=2,        # 최소 2개 문서에 등장
                        ngram_range=(1, 2)  # 단어, 구(2단어) 모두 사용
                    )
                    
                    # 텍스트 벡터화
                    try:
                        tfidf_matrix = tfidf.fit_transform(df[f"{col}_preprocessed"])
                        
                        # 벡터화 결과를 데이터프레임에 추가
                        feature_names = tfidf.get_feature_names_out()
                        tfidf_df = pd.DataFrame(
                            tfidf_matrix.toarray(),
                            columns=[f"TFIDF_{col}_{feature}" for feature in feature_names]
                        )
                        
                        # 기존 데이터프레임과 병합
                        df = pd.concat([df, tfidf_df], axis=1)
                        
                        # 모델 저장
                        self.models[f"tfidf_{col}_{dataset_name}"] = tfidf
                        
                        # 전처리된 컬럼 제거
                        df = df.drop(columns=[f"{col}_preprocessed"])
                        
                        logger.info(f"✅ '{col}' 컬럼에 TF-IDF 적용 (특성 {len(feature_names)}개 추출)")
                    except Exception as e:
                        logger.warning(f"⚠️ '{col}' 컬럼 TF-IDF 적용 실패: {e}")
                        # 전처리된 컬럼 제거
                        if f"{col}_preprocessed" in df.columns:
                            df = df.drop(columns=[f"{col}_preprocessed"])
        
        return df
    
    def _load_bge_model(self, model_name="BAAI/bge-m3"):
        """
        BGE-M3 임베딩 모델 로드
        
        Parameters:
            model_name (str): 사용할 모델 이름
            
        Returns:
            None (내부 변수에 모델 설정)
        """
        # 모델이 이미 로드되어 있는 경우 다시 로드하지 않음
        if self.bge_model is not None and self.bge_tokenizer is not None:
            return
        
        logger.info(f"🔄 BGE-M3 임베딩 모델 로드 시작: {model_name}")
        
        try:
            # 디바이스 설정 (GPU 사용 가능 시 GPU, 아니면 CPU)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 토크나이저 및 모델 로드
            self.bge_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bge_model = AutoModel.from_pretrained(model_name)
            
            # 모델을 지정된 디바이스로 이동
            self.bge_model.to(self.device)
            self.bge_model.eval()  # 평가 모드로 설정
            
            logger.info(f"✅ BGE-M3 임베딩 모델 '{model_name}'을(를) {self.device}에 성공적으로 로드함")
            
        except Exception as e:
            logger.error(f"❌ BGE-M3 모델 로드 중 오류 발생: {e}")
            self.bge_model = None
            self.bge_tokenizer = None
            self.device = None
            raise RuntimeError(f"❌ BGE-M3 모델 로드 중 오류 발생: {e}")
    
    def _get_bge_embedding(self, texts, batch_size=32, max_length=512):
        """
        BGE-M3 모델을 사용하여 텍스트의 임베딩 벡터 생성
        
        Parameters:
            texts (list): 임베딩할 텍스트 리스트
            batch_size (int): 배치 크기
            max_length (int): 최대 토큰 길이
            
        Returns:
            list: 임베딩 벡터 리스트
        """
        if not texts:
            return []
        
        # 결측값을 빈 문자열로 변환
        texts = [str(t) if pd.notnull(t) else "" for t in texts]
        
        embeddings = []
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        # 배치 단위로 처리
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 텍스트 토큰화 및 모델 입력 데이터 준비
            inputs = self.bge_tokenizer(
                batch_texts, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=max_length
            )
            
            # 입력을 디바이스로 이동
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            # 추론 실행 (그래디언트 계산 없이)
            with torch.no_grad():
                outputs = self.bge_model(**inputs)
            
            # CLS 토큰 임베딩 추출 및 CPU로 이동
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def _apply_bge_embeddings(self, df, dataset_name):
        """BGE-M3 임베딩 모델을 적용하여 텍스트 임베딩 생성"""
        # 텍스트 컬럼 찾기
        text_columns = []
        for col in df.select_dtypes(include=['object']).columns:
            if any(keyword in col for keyword in ['제목', 'title', '내용', 'content', '설명', '항목', '공고']):
                text_columns.append(col)
        
        if not text_columns or len(df) < 30:  # 최소 30개 이상의 행이 있을 때만 적용
            return df
        
        try:
            # BGE-M3 모델 로드
            self._load_bge_model()
            
            if self.bge_model is None or self.bge_tokenizer is None:
                logger.warning("⚠️ BGE-M3 모델이 로드되지 않아 임베딩을 건너뜁니다.")
                return df
            
            # 각 텍스트 컬럼에 대해 임베딩 적용
            for col in text_columns:
                if df[col].str.len().sum() < 200:  # 텍스트 양이 너무 적으면 건너뜀
                    continue
                
                logger.info(f"🔄 '{col}' 컬럼에 BGE-M3 임베딩 적용 중...")
                
                # 텍스트 데이터 추출 및 결측치 처리
                texts = df[col].fillna("").tolist()
                
                # 임베딩 생성
                embeddings = self._get_bge_embedding(texts)
                
                if not embeddings:
                    logger.warning(f"⚠️ '{col}' 컬럼 임베딩 결과가 비어 있습니다.")
                    continue
                
                # 임베딩 결과를 데이터프레임으로 변환
                embedding_df = pd.DataFrame(
                    embeddings,
                    columns=[f"BGE_{col}_{i}" for i in range(embeddings[0].shape[0])]
                )
                
                # 임베딩 차원이 너무 크면 처음 10개 차원만 사용
                if embedding_df.shape[1] > 10:
                    logger.info(f"✂️ '{col}' 임베딩 차원을 {embedding_df.shape[1]}에서 10으로 축소합니다.")
                    embedding_df = embedding_df.iloc[:, :10]
                    embedding_df.columns = [f"BGE_{col}_{i}" for i in range(10)]
                
                # 기존 데이터프레임과 병합
                df = pd.concat([df, embedding_df.reset_index(drop=True)], axis=1)
                
                logger.info(f"✅ '{col}' 컬럼에 BGE-M3 임베딩 적용 완료 ({embedding_df.shape[1]}차원)")
                
        except Exception as e:
            logger.warning(f"⚠️ BGE-M3 임베딩 적용 중 오류 발생: {e}")
        
        return df
    
    def _apply_dimension_reduction(self, df, dataset_name):
        """PCA를 이용한 차원 축소"""
        # 연속형 수치 변수만 선택
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # TFIDF나 원-핫 인코딩 결과는 제외 (이미 특성으로 사용)
        numeric_cols = [col for col in numeric_cols if not any(prefix in col for prefix in ['TFIDF_', 'dummy_', 'BGE_', 'PCA_'])]
        
        if len(numeric_cols) >= 5:  # 최소 5개 이상의 수치형 컬럼이 있을 때만 PCA 적용
            # 결측치 채우기
            df_numeric = df[numeric_cols].fillna(0)
            
            try:
                # 데이터 정규화 (PCA 전)
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(df_numeric)
                
                # PCA 컴포넌트 수 결정 (특성 수에 따라 동적으로)
                n_components = min(3, len(numeric_cols) - 1)
                
                # PCA 적용
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(scaled_data)
                
                # PCA 결과를 데이터프레임에 추가
                pca_df = pd.DataFrame(
                    pca_result,
                    columns=[f"PCA_{i+1}" for i in range(pca_result.shape[1])]
                )
                
                # 기존 데이터프레임과 병합
                df = pd.concat([df, pca_df], axis=1)
                
                # 모델 저장
                self.models[f"pca_{dataset_name}"] = pca
                self.models[f"scaler_{dataset_name}"] = scaler
                
                # 설명된 분산 비율 기록
                variance_ratio = pca.explained_variance_ratio_
                logger.info(f"✅ PCA 적용 완료 (설명된 분산 비율: {[f'{v:.2%}' for v in variance_ratio]})")
                
            except Exception as e:
                logger.warning(f"⚠️ PCA 적용 실패: {e}")
        
        return df
    
    def _select_best_features(self, df, dataset_name):
        """특성 선택 (타겟 변수가 있는 경우)"""
        target_vars = [col for col in df.columns if col in ['거래적정성', '낙찰가격비율', '투찰률']]
        
        if not target_vars:
            return df  # 타겟 변수가 없으면 원래 데이터프레임 반환
        
        # 선택할 타겟 변수
        target_var = target_vars[0]
        
        # 예측 변수 선택 (숫자형 컬럼)
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        feature_cols = [col for col in num_cols if col != target_var and not col.startswith('selected_')]
        
        # 최소 5개 이상의 특성과 30개 이상의 행이 있는 경우에만 적용
        if len(feature_cols) >= 5 and len(df) >= 30:
            # 결측치 채우기
            X = df[feature_cols].fillna(0)
            y = df[target_var].fillna(0)
            
            try:
                # 상위 k개 특성 선택 (f_regression 사용)
                k = min(10, len(feature_cols))  # 최대 10개 특성 선택
                selector = SelectKBest(f_regression, k=k)
                X_new = selector.fit_transform(X, y)
                
                # 선택된 특성 인덱스
                selected_indices = selector.get_support(indices=True)
                selected_features = [feature_cols[i] for i in selected_indices]
                
                # 선택된 특성을 새 열로 추가
                X_selected = pd.DataFrame(X_new, columns=[f"selected_{feature}" for feature in selected_features])
                
                # 기존 데이터프레임과 병합
                df = pd.concat([df, X_selected], axis=1)
                
                # 모델 저장
                self.models[f"feature_selector_{dataset_name}"] = selector
                
                logger.info(f"✅ 특성 선택 완료 (상위 {k}개 특성 선택됨: {', '.join(selected_features[:3])}... 외 {len(selected_features)-3}개)")
            except Exception as e:
                logger.warning(f"⚠️ 특성 선택 적용 실패: {e}")
        
        return df
    
    def _create_combined_features(self, df):
        """새로운 특성 조합 생성"""
        # 1. 금액 관련 특성 조합
        amount_cols = []
        for col in df.columns:
            if 'norm_log_' in col and any(keyword in col for keyword in ['금액', 'price', 'amount']):
                amount_cols.append(col)
        
        # 2개 이상의 금액 컬럼이 있는 경우
        if len(amount_cols) >= 2:
            for i in range(len(amount_cols)):
                for j in range(i+1, len(amount_cols)):
                    col1 = amount_cols[i]
                    col2 = amount_cols[j]
                    
                    # 비율 특성 생성
                    ratio_name = f"ratio_{col1.split('_', 2)[-1]}_{col2.split('_', 2)[-1]}"
                    df[ratio_name] = df[col1] / df[col2].replace(0, np.nan)
                    df[ratio_name] = df[ratio_name].fillna(0)
                    
                    # 차이 특성 생성
                    diff_name = f"diff_{col1.split('_', 2)[-1]}_{col2.split('_', 2)[-1]}"
                    df[diff_name] = df[col1] - df[col2]
                    
                    logger.info(f"✅ '{col1}'와 '{col2}'의 비율 및 차이 특성 생성")
        
        # 2. 날짜 간 차이 특성 생성
        date_cols = [col for col in df.columns if 'date' in col.lower() or '일자' in col]
        
        if len(date_cols) >= 2:
            for i in range(len(date_cols)):
                for j in range(i+1, len(date_cols)):
                    try:
                        # 날짜 형식으로 변환
                        df[date_cols[i]] = pd.to_datetime(df[date_cols[i]], errors='coerce')
                        df[date_cols[j]] = pd.to_datetime(df[date_cols[j]], errors='coerce')
                        
                        # 일수 차이 계산
                        days_diff = f"days_between_{date_cols[i]}_{date_cols[j]}"
                        df[days_diff] = (df[date_cols[i]] - df[date_cols[j]]).dt.days
                        
                        logger.info(f"✅ '{date_cols[i]}'와 '{date_cols[j]}' 사이의 일수 차이 특성 생성")
                    except Exception as e:
                        logger.warning(f"⚠️ 날짜 차이 특성 생성 실패: {e}")
        
        return df 