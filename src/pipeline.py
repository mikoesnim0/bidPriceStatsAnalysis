# 데이터 파이프라인

## 함수목록

- transform
- restructure_data
- parse_sajeong_rate
- group_to_list
- separate_data
- process_data
- log_transforming
- normalizing
- one_hot_encoding
- load_bge_model
- embedding
- get_embedding_vector
- series_to_dataframe
- dimension_reducing_PCA
- dimension_reducing_UMAP
- calculate_target
- parse_range_level
- load_bins
- generate_bins
- extract_bounds
- process_row_fixed_bins
- data_to_target


### 로그 변환

# ============================================================
#    로그 변환 함수: Series 값에 log1p를 적용
# ============================================================

def log_transforming(series):
    """
    주어진 Pandas Series의 값을 자연로그 변환(log1p)을 수행하여 반환합니다.

    Parameters:
        series (pd.Series): 로그 변환할 데이터. 모든 값은 0 이상의 양수여야 합니다.

    Returns:
        pd.Series: 원본 인덱스와 이름을 유지한 로그 변환된 데이터.

    Raises:
        TypeError: 입력값이 Pandas Series가 아닐 경우.
        ValueError: Series에 음수 값이 포함된 경우.
    """
    # import pandas as pd
    # import numpy as np
    # import logging

    logging.info(f"✅ [log_transforming] 로그 변환 시작 - Column: {series.name}")

    # 입력 데이터 타입 확인
    if not isinstance(series, pd.Series):
        raise TypeError("입력값은 pandas Series여야 합니다.")

    # 숫자형 데이터인지 확인 (불필요한 변환 방지)
    if not np.issubdtype(series.dtype, np.number):
        raise ValueError("로그 변환을 위해 숫자형 데이터가 필요합니다.")

    # 음수 값 확인
    if (series < 0).any():
        raise ValueError("로그 변환할 데이터에 음수 값이 포함되어 있습니다.")

    # float 타입이 아닌 경우만 변환
    if not np.issubdtype(series.dtype, np.floating):
        series = series.astype(float)

    # 자연로그 변환
    result_series = np.log1p(series)

    logging.info(f"✅ [log_transforming] 로그 변환 완료 - 결과 샘플: {result_series.head(3).tolist()}")

    return result_series


### 정규화

def normalizing(series):
    """
    주어진 Pandas Series의 값을 표준화(Standardization)하여 평균 0, 표준편차 1의 분포로 변환합니다.

    Parameters:
        series (pd.Series): 표준화할 데이터.

    Returns:
        pd.Series: 표준화된 데이터 (원본 인덱스와 이름 유지).

    Raises:
        TypeError: 입력값이 Pandas Series가 아닐 경우.
    """
    # import pandas as pd
    # import numpy as np
    # import logging
    from sklearn.preprocessing import StandardScaler

    logging.info(f"✅ [normalizing] 표준화 시작 - Column: {series.name}")

    if not isinstance(series, pd.Series):
        logging.error("❌입력값이 pandas Series가 아닙니다!")
        raise TypeError("입력값은 pandas Series여야 합니다.")

    scaler = StandardScaler()

    # 1차원 데이터를 2차원 배열로 변환 후 표준화
    result_array = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

    # 원본 인덱스 및 컬럼명 유지
    result_series = pd.Series(result_array, index=series.index, name=series.name)

    logging.info(f"✅ [normalizing] 표준화 완료 - 결과 샘플: {result_series.head(3).tolist()}")

    return result_series


### 원-핫 인코딩

def one_hot_encoding(series, delimiter="/"):
    """
    주어진 Pandas Series에 대해 지정된 구분자(delimiter)를 기준으로 원-핫 인코딩을 수행합니다.

    Parameters:
        series (pd.Series): 원본 문자열 데이터. 각 셀에 여러 값이 포함되어 있을 수 있음.
        delimiter (str): 값들 간의 구분자 (기본값: "/").

    Returns:
        pd.DataFrame: 원-핫 인코딩 결과를 포함하는 DataFrame.

    Raises:
        TypeError: 입력값이 Pandas Series가 아닐 경우.
        ValueError: Series의 데이터 타입이 문자열이 아닐 경우.
    """
    # import pandas as pd
    # import logging

    logging.info(f"✅ [one_hot_encoding] 시작 - Column: {series.name}")

    if not isinstance(series, pd.Series):
        logging.error("❌ 입력값이 pandas Series가 아닙니다!")
        raise TypeError("❌ 입력값은 pandas Series여야 합니다!")

    # 문자열 타입 확인
    if not series.dtype == "object":
        logging.error("❌ 원-핫 인코딩을 위해 Series는 문자열 타입이어야 합니다!")
        raise ValueError("❌ 원-핫 인코딩을 위해 Series는 문자열 타입이어야 합니다!")

    # 문자열을 delimiter로 분리하여 dummies 생성
    result_df = series.str.get_dummies(sep=delimiter)

    # 빈 문자열 컬럼이 생성될 경우 제거
    if "" in result_df.columns:
        result_df = result_df.drop(columns="")

    # 기존 컬럼명을 접두사로 추가하여 결과 DataFrame 생성
    result_df = result_df.add_prefix(f"{series.name}_")

    # ✅ 결과 컬럼 수 & 컬럼 리스트 출력
    logging.info(f"✅ [one_hot_encoding] 완료 - 생성된 컬럼 수: {len(result_df.columns)}, 컬럼 목록: {result_df.columns.tolist()}")

    return result_df


### 텍스트 임베딩

def load_bge_model(model_name="BAAI/bge-m3"):
    """
    지정된 모델 이름의 BGE-M3 모델과 토크나이저를 로드합니다.

    Parameters:
        model_name (str, optional): 사용할 모델 이름 (기본값: "BAAI/bge-m3").

    Returns:
        tuple: (tokenizer, model, device) - 로드된 토크나이저, 모델, 실행 디바이스 (GPU/CPU).

    Raises:
        RuntimeError: 모델 로드 중 오류가 발생한 경우.
    """
    # import logging
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"✅ [load_bge_model] 모델 로드 시작: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)  # 모델을 GPU/CPU에 로드
        model.eval()  # 평가 모드 설정
        logging.info(f"✅ [load_bge_model] 모델 '{model_name}'이(가) {device}에서 성공적으로 로드됨")

    except Exception as e:
        logging.error(f"❌ 모델 로드 중 오류 발생: {e}")
        raise RuntimeError(f"❌ 모델 로드 중 오류 발생: {e}")

    return tokenizer, model, device


def get_embedding_vector(texts, tokenizer, model, device, max_length=512):
    """
    텍스트 리스트를 받아 BGE-M3 모델을 사용하여 임베딩 벡터 배열을 생성합니다.

    Parameters:
        texts (list of str): 임베딩할 텍스트들의 리스트.
        tokenizer: 로드된 토크나이저.
        model: 로드된 임베딩 모델.
        device: 실행 디바이스 (GPU/CPU).
        max_length (int, optional): 최대 토큰 길이 (기본값: 512).

    Returns:
        np.ndarray: (배치 크기, 임베딩 차원) 형태의 임베딩 벡터 배열.

    Raises:
        TypeError: texts가 문자열 리스트가 아닐 경우.
    """
    # import numpy as np
    # import pandas as pd
    import torch
    import logging

    logging.info(f"✅ [get_embedding_vector] 텍스트 개수: {len(texts)}")

    if not isinstance(texts, list):
        logging.error("❌ 입력값이 문자열 리스트가 아닙니다!")
        raise TypeError("❌ 입력값은 문자열 리스트여야 합니다!")

    if not texts:
        logging.warning("⚠️ [get_embedding_vector] 빈 리스트 입력됨. 빈 배열 반환.")
        return np.array([])

    # 결측값을 빈 문자열로 변환
    texts = [str(text) if pd.notnull(text) else "" for text in texts]

    # 텍스트 토큰화 및 모델 입력 데이터 준비
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # CLS 토큰 임베딩 추출
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    logging.info(f"✅ [get_embedding_vector] 임베딩 완료 - 결과 shape: {embeddings.shape}")

    return embeddings


def embedding(series, tokenizer, model, device, batch_size=32):
    """
    주어진 Pandas Series의 텍스트를 배치 단위로 임베딩하여,
    각 텍스트에 해당하는 임베딩 벡터 리스트를 반환합니다.

    Parameters:
        series (pd.Series): 임베딩할 텍스트 데이터가 포함된 컬럼.
        tokenizer: 로드된 토크나이저.
        model: 로드된 임베딩 모델.
        device: 실행 디바이스 (GPU/CPU).
        batch_size (int, optional): 배치 크기 (기본값: 32).

    Returns:
        pd.Series: 각 텍스트의 임베딩 벡터(리스트)를 포함하는 Series.

    Raises:
        TypeError: 입력값이 Pandas Series가 아닐 경우.
    """
    # import numpy as np
    # import pandas as pd
    # import logging
    import torch
    from tqdm import tqdm

    logging.info(f"✅ [embedding] 시작 - Column: {series.name}, 길이: {len(series)}")

    if not isinstance(series, pd.Series):
        logging.error("❌ 입력 값이 Pandas Series가 아닙니다!")
        raise TypeError("❌ 입력 값은 Pandas Series여야 합니다!")

    if series.empty:
        logging.warning("⚠️ [embedding] 빈 Series 입력됨. 빈 결과 반환.")
        return pd.Series([], index=series.index, dtype=object)

    # 결측값을 빈 문자열로 대체
    series = series.fillna("")

    embeddings = []
    num_batches = (len(series) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(series), batch_size), total=num_batches, desc="임베딩 진행 중", position=0):
        batch_texts = series.iloc[i:i + batch_size].tolist()
        batch_embeddings = get_embedding_vector(batch_texts, tokenizer, model, device)

        if batch_embeddings is None or len(batch_embeddings) == 0:
            logging.warning(f"⚠️ [embedding] {i}번째 배치에서 빈 결과 발생. 빈 배열 추가.")
            batch_embeddings = np.zeros((len(batch_texts), model.config.hidden_size))

        embeddings.extend(batch_embeddings)

    # 임베딩 벡터를 리스트 형식으로 저장
    result_series = pd.Series([emb.tolist() for emb in embeddings], index=series.index, dtype=object)

    logging.info("✅ [embedding] 완료")

    return result_series


### 차원축소

# ============================================================
#  SVD 차원 축소 함수: Truncated SVD를 사용한 차원 축소
# ============================================================
def dimension_reducing_SVD(df, prefix, components_n=2, random_state=42):
    """
    주어진 DataFrame을 Truncated SVD를 사용하여 차원 축소합니다.

    Parameters:
        df (pd.DataFrame): 차원 축소할 입력 데이터.
        prefix (str): 결과 컬럼 이름에 사용할 접두사.
        components_n (int, optional): 목표 차원 수 (기본값: 2).
        random_state (int, optional): SVD의 랜덤 시드 값 (기본값: 42).

    Returns:
        pd.DataFrame: SVD 차원 축소 후 결과 DataFrame (원본 인덱스 유지).

    Raises:
        TypeError: 입력값이 Pandas DataFrame이 아닐 경우.
        ValueError: components_n이 1보다 작을 경우.
    """
    # import pandas as pd
    # import logging
    from sklearn.decomposition import TruncatedSVD

    logging.info(f"✅ [dimension_reducing_SVD] 시작 - 입력 DataFrame shape: {df.shape}")

    # 입력 타입 검증
    if not isinstance(df, pd.DataFrame):
        logging.error("❌ 입력값은 pandas DataFrame이어야 합니다!")
        raise TypeError("❌ 입력값은 pandas DataFrame이어야 합니다!")

    # components_n 값 검증
    if components_n < 1:
        logging.error("❌ components_n 값은 1 이상이어야 합니다!")
        raise ValueError("❌ components_n 값은 1 이상이어야 합니다!")

    if components_n > df.shape[1]:
        components_n = df.shape[1]
        logging.warning("⚠️ [PCA] components_n이 입력 차원보다 크므로 입력 차원으로 설정합니다.")

    # Truncated SVD 실행
    svd = TruncatedSVD(n_components=components_n, random_state=random_state)
    reduced_data = svd.fit_transform(df)

    # 결과 컬럼명 생성
    reduced_columns = [f"SVD_{prefix}_{i + 1}" for i in range(components_n)]

    # 결과 DataFrame 생성 (원본 인덱스 유지)
    result_df = pd.DataFrame(reduced_data, columns=reduced_columns, index=df.index)

    logging.info(f"✅ [dimension_reducing_SVD] 완료 - 결과 shape: {result_df.shape}")

    return result_df


# ============================================================
#  PCA 차원 축소 함수: PCA를 사용한 차원 축소
# ============================================================
def dimension_reducing_PCA(df, prefix, components_n=2):
    """
    주어진 DataFrame을 PCA를 사용하여 차원 축소합니다.

    Parameters:
        df (pd.DataFrame): 차원 축소할 입력 데이터.
        prefix (str): 결과 컬럼 이름에 사용할 접두사.
        components_n (int, optional): 목표 차원 수 (기본값: 2).

    Returns:
        pd.DataFrame: PCA 차원 축소 후 결과 DataFrame (원본 인덱스 유지).

    Raises:
        TypeError: 입력값이 Pandas DataFrame이 아닐 경우.
        ValueError: components_n이 1보다 작을 경우.
    """
    # import pandas as pd
    # import logging
    from sklearn.decomposition import PCA

    logging.info(f"✅ [dimension_reducing_PCA] 시작 - 입력 DataFrame shape: {df.shape}")

    # 입력 타입 검증
    if not isinstance(df, pd.DataFrame):
        logging.error("❌ 입력값은 pandas DataFrame이어야 합니다!")
        raise TypeError("❌ 입력값은 pandas DataFrame이어야 합니다!")

    # components_n 값 검증
    if components_n < 1:
        logging.error("❌ components_n 값은 1 이상이어야 합니다!")
        raise ValueError("❌ components_n 값은 1 이상이어야 합니다!")

    if components_n > df.shape[1]:
        components_n = df.shape[1]
        logging.warning("⚠️ [PCA] components_n이 입력 차원보다 크므로 입력 차원으로 설정합니다.")

    # PCA 실행
    pca = PCA(n_components=components_n)
    reduced_data = pca.fit_transform(df)

    # 결과 컬럼명 생성
    reduced_columns = [f"PCA_{prefix}_{i + 1}" for i in range(components_n)]

    # 설명된 분산 계산
    explained_variance = sum(pca.explained_variance_ratio_) * 100

    # 결과 DataFrame 생성 (원본 인덱스 유지)
    result_df = pd.DataFrame(reduced_data, columns=reduced_columns, index=df.index)

    logging.info(f"✅ [dimension_reducing_PCA] 완료 - 결과 shape: {result_df.shape}, 설명된 분산: {explained_variance:.2f}%")

    return result_df


# ============================================================
# UMAP 차원 축소 함수: UMAP을 사용한 비선형 차원 축소
# ============================================================
def dimension_reducing_UMAP(df, prefix, components_n=2, n_neighbors=15, metric="euclidean", random_state=42):
    """
    주어진 DataFrame을 UMAP을 사용하여 비선형 차원 축소합니다.

    Parameters:
        df (pd.DataFrame): 차원 축소할 입력 데이터.
        prefix (str): 결과 컬럼 이름에 사용할 접두사.
        components_n (int, optional): 목표 차원 수 (기본값: 2).
        n_neighbors (int, optional): UMAP의 이웃 수 (기본값: 15).
        metric (str, optional): 거리 측정 방식 (기본값: "euclidean").
        random_state (int, optional): 랜덤 시드 값 (기본값: 42).

    Returns:
        pd.DataFrame: UMAP 차원 축소 후 결과 DataFrame (원본 인덱스 유지).

    Raises:
        TypeError: 입력값이 Pandas DataFrame이 아닐 경우.
        ValueError: components_n이 1보다 작을 경우.
    """
    # import pandas as pd
    # import logging
    import umap

    logging.info(f"✅[dimension_reducing_UMAP] 시작 - 입력 DataFrame shape: {df.shape}")

    # 입력 타입 검증
    if not isinstance(df, pd.DataFrame):
        logging.error("❌입력값은 pandas DataFrame이어야 합니다!")
        raise TypeError("❌입력값은 pandas DataFrame이어야 합니다!")

    # components_n 값 검증
    if components_n < 1:
        logging.error("❌components_n 값은 1 이상이어야 합니다!")
        raise ValueError("❌components_n 값은 1 이상이어야 합니다!")

    if components_n > df.shape[1]:
        components_n = df.shape[1]
        logging.warning("⚠️ [PCA] components_n이 입력 차원보다 크므로 입력 차원으로 설정합니다.")

    # UMAP 실행
    umap_model = umap.UMAP(n_components=components_n, n_neighbors=n_neighbors, metric=metric, random_state=random_state)
    reduced_data = umap_model.fit_transform(df)

    # 결과 컬럼명 생성
    reduced_columns = [f"UMAP_{prefix}_{i + 1}" for i in range(components_n)]

    # 결과 DataFrame 생성 (원본 인덱스 유지)
    result_df = pd.DataFrame(reduced_data, columns=reduced_columns, index=df.index)

    logging.info(f"✅[dimension_reducing_UMAP] 완료 - 결과 shape: {result_df.shape}")

    return result_df


### 데이터 분리, 가공

# ============================================================
#   데이터 분리 함수: '예가범위' 기준 그룹 분류
# ============================================================
def separate_data(df):
    """
    '예가범위' 컬럼을 기준으로 데이터를 세 그룹(예: range3, range2, others)으로 분류합니다.

    Parameters:
        df (pd.DataFrame): 입력 데이터. '예가범위' 컬럼이 포함되어 있어야 합니다.

    Returns:
        dict: {'range3': DataFrame, 'range2': DataFrame, 'others': DataFrame} 형태로 그룹별 DataFrame 반환.

    Raises:
        TypeError: 입력 데이터가 Pandas DataFrame이 아닐 경우.
        KeyError: '예가범위' 컬럼이 DataFrame에 없을 경우.
    """
    # import pandas as pd
    # import logging

    logging.info(f"✅[separate_data] 시작 - 입력 DataFrame shape: {df.shape}")

    # 입력 타입 검증
    if not isinstance(df, pd.DataFrame):
        logging.error("❌입력 데이터는 pandas DataFrame이어야 합니다!")
        raise TypeError("❌입력 데이터는 pandas DataFrame이어야 합니다!")

    # '예가범위' 컬럼 존재 여부 확인
    if "예가범위" not in df.columns:
        logging.error("❌'예가범위' 컬럼이 존재하지 않습니다!")
        raise KeyError("❌'예가범위' 컬럼이 존재하지 않습니다!")

    # 지정한 예가범위 값을 기준으로 그룹화
    range_values = ["+3%~-3%", "+2%~-2%"]
    range3_df = df[df["예가범위"] == "+3%~-3%"].reset_index(drop=True)
    range2_df = df[df["예가범위"] == "+2%~-2%"].reset_index(drop=True)
    others_df = df[~df["예가범위"].isin(range_values)].reset_index(drop=True)

    logging.info(
        f"✅[separate_data] 분류 완료 - range3: {len(range3_df)}, range2: {len(range2_df)}, others: {len(others_df)}")

    return {"range3": range3_df, "range2": range2_df, "others": others_df}


# ============================================================
#   데이터 재구성 함수: 공고 데이터와 투찰 데이터 병합 및 정리
# ============================================================
def restructure_data(df1, df2):
    """
    공고 데이터와 투찰 데이터를 정리 한 후 병합 및 재구성합니다.

    Parameters:
        df1 (pd.DataFrame): 공고 데이터.
        df2 (pd.DataFrame): 투찰 데이터.

    Returns:
        pd.DataFrame: 병합된 최종 데이터.

    Raises:
        TypeError: 입력 데이터가 Pandas DataFrame이 아닐 경우.
        KeyError: 필요한 컬럼이 존재하지 않을 경우.
    """
    # import pandas as pd
    # import logging

    logging.info("✅ [restructure_data] 데이터 재구성 시작...")

    # 입력 데이터 검증
    if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
        logging.error("❌ 입력 데이터는 Pandas DataFrame이어야 합니다!")
        raise TypeError("❌ 입력 데이터는 Pandas DataFrame이어야 합니다!")

    # 공고번호 컬럼 확인
    if "공고번호" not in df1.columns or "공고번호" not in df2.columns:
        logging.error("❌ '공고번호' 컬럼이 두 데이터프레임에 모두 존재해야 합니다!")
        raise KeyError("❌ '공고번호' 컬럼이 누락되었습니다.")

    # 필요한 컬럼 정의 및 존재 여부 확인
    df1_required_columns = ['공고번호', '공고제목', '발주처(수요기관)', '지역제한', '기초금액',
                            '예정가격', '예가범위', 'A값', '투찰률(%)', '참여업체수', '공고구분표시']
    df2_required_columns = ['공고번호', '기초대비 사정률(%)']

    df1_missing_columns = [col for col in df1_required_columns if col not in df1.columns]
    df2_missing_columns = [col for col in df2_required_columns if col not in df2.columns]

    if df1_missing_columns:
        logging.error(f"❌ 공고 데이터에서 누락된 컬럼: {df1_missing_columns}")
        raise KeyError(f"❌ 공고 데이터에서 누락된 컬럼: {df1_missing_columns}")

    if df2_missing_columns:
        logging.error(f"❌ 투찰 데이터에서 누락된 컬럼: {df2_missing_columns}")
        raise KeyError(f"❌ 투찰 데이터에서 누락된 컬럼: {df2_missing_columns}")

    # 데이터프레임 복사 (원본 데이터 보호)
    nt_df = df1.copy()
    bd_df = df2.copy()

    # 필요 컬럼 추출
    nt_df = nt_df[df1_required_columns]
    bd_df = bd_df[df2_required_columns]

    # 결측값 처리
    nt_df.dropna(subset=["공고번호"], inplace=True)
    nt_df.dropna(subset=["예가범위"], inplace=True)

    nt_df["투찰률(%)"] = nt_df["투찰률(%)"].fillna(nt_df["투찰률(%)"].mean(numeric_only=True))
    nt_df["공고구분표시"] = nt_df["공고구분표시"].fillna("")

    bd_df.dropna(subset=["공고번호"], inplace=True)

    # 공고 데이터 중복 제거
    nt_df = nt_df.drop_duplicates(subset=["공고번호"])

    # 투찰 데이터 사정률 변환 및 정리
    bd_df["사정률"] = parse_sajeong_rate(bd_df["기초대비 사정률(%)"])
    bd_df.dropna(subset=["사정률"], inplace=True)
    bd_df = bd_df[["공고번호", "사정률"]]
    bd_df = group_to_list(bd_df, "공고번호", "사정률")

    # 문자열 컬럼 정리 (공백 제거)
    str_columns = ["예가범위", "발주처(수요기관)", "지역제한", "공고구분표시"]
    for col in str_columns:
        nt_df[col] = nt_df[col].astype(str).str.replace(r"\s+", "", regex=True)

    # 데이터 병합
    merged_data = pd.merge(nt_df, bd_df, on="공고번호", how="inner").reset_index(drop=True)

    logging.info(f"✅ [restructure_data] 데이터 재구성 완료! - 데이터 shape: {merged_data.shape}")

    return merged_data


# ============================================================
#   Series를 DataFrame으로 변환 함수: 리스트 확장
# ============================================================
def series_to_dataframe(series):
    """
    Pandas Series의 각 원소가 리스트인 경우, 이를 개별 컬럼으로 확장하여 DataFrame으로 변환합니다.

    Parameters:
        series (pd.Series): 변환할 데이터.

    Returns:
        pd.DataFrame: 각 원소가 개별 컬럼으로 확장된 DataFrame.

    Raises:
        TypeError: 입력 데이터가 Pandas Series가 아닐 경우.
    """
    # import pandas as pd
    # import numpy as np
    # import logging

    logging.info(f"✅[series_to_dataframe] 시작 - 입력 Series 길이: {len(series)}")

    if not isinstance(series, pd.Series):
        logging.error("❌입력 데이터는 Pandas Series이어야 합니다!")
        raise TypeError("❌입력 데이터는 Pandas Series이어야 합니다!")

    # 컬럼명 설정 (기본값: "feature" 사용)
    column_name = series.name if series.name else "feature"

    # DataFrame 변환
    expanded_df = pd.DataFrame(series.tolist())

    # 컬럼명 지정
    expanded_df.columns = [f"{column_name}_{i}" for i in range(expanded_df.shape[1])]

    logging.info(f"✅[series_to_dataframe] 완료 - 결과 shape: {expanded_df.shape}")

    return expanded_df


# ============================================================
#   그룹별 리스트 생성 함수: 지정 컬럼 그룹화 후 리스트화
# ============================================================
def group_to_list(df, group_col, value_col):
    """
    지정된 그룹 컬럼을 기준으로 데이터를 그룹화한 후, 해당 그룹의 값을 리스트로 집계합니다.

    Parameters:
        df (pd.DataFrame): 입력 데이터.
        group_col (str): 그룹화할 컬럼 이름 (예: '공고번호').
        value_col (str): 집계할 값이 있는 컬럼 이름 (예: '사정률(%)').

    Returns:
        pd.DataFrame: 그룹과 해당 값의 리스트를 포함한 DataFrame.

    Raises:
        TypeError: 입력 데이터가 Pandas DataFrame이 아닐 경우.
        KeyError: group_col 또는 value_col이 DataFrame에 존재하지 않을 경우.
    """
    # import pandas as pd
    # import logging

    logging.info(f"✅ [group_to_list] 시작 - 그룹 컬럼: {group_col}, 값 컬럼: {value_col}")

    # 입력 타입 검증
    if not isinstance(df, pd.DataFrame):
        logging.error("❌ 입력 데이터는 Pandas DataFrame이어야 합니다!")
        raise TypeError("❌ 입력 데이터는 Pandas DataFrame이어야 합니다!")

    # group_col, value_col 존재 여부 확인
    if group_col not in df.columns or value_col not in df.columns:
        logging.error(f"❌ 컬럼 '{group_col}' 또는 '{value_col}'이(가) 데이터프레임에 존재하지 않습니다!")
        raise KeyError(f"❌ 컬럼 '{group_col}' 또는 '{value_col}'이(가) 존재하지 않습니다.")

    # 결측값 제거하여 그룹화 오류 방지
    df_filtered = df.dropna(subset=[value_col])

    # 그룹화 후 리스트 변환
    grouped_df = df_filtered.groupby(group_col)[value_col].agg(list).reset_index()

    logging.info(f"✅ [group_to_list] 완료 - 결과 shape: {grouped_df.shape}")

    return grouped_df


### taget 데이터 처리

# ============================================================
#   기초대비 사정률 파싱 함수: 문자열에서 소수점 숫자 추출
# ============================================================
def parse_sajeong_rate(series):
    """
    '기초대비 사정률(%)' 문자열에서 첫 번째 소수 형태의 숫자를 추출하여 float으로 변환합니다.

    Parameters:
        series (pd.Series): 변환할 문자열 데이터.

    Returns:
        pd.Series: 추출된 숫자를 float 형식으로 포함하는 Series.

    Raises:
        TypeError: 입력값이 Pandas Series가 아닐 경우.
    """
    # import pandas as pd
    # import logging

    logging.info(f"✅ [parse_sajeong_rate] 시작 - Series 길이: {len(series)}")

    # 입력 데이터 검증
    if not isinstance(series, pd.Series):
        logging.error("❌ 입력 데이터는 Pandas Series이어야 합니다!")
        raise TypeError("❌ 입력 데이터는 Pandas Series이어야 합니다!")

    # 결측값 제거
    series = series.dropna()

    # 정규식으로 첫 번째 소수 형태 숫자 추출
    result_series = series.astype(str).str.extract(r'(-?\d+\.\d+)')[0]

    # 숫자로 변환 (변환 실패 시 NaN 처리)
    result_series = pd.to_numeric(result_series, errors="coerce")

    logging.info(f"✅ [parse_sajeong_rate] 완료 - 결과 샘플: {result_series.head(3).tolist()}")

    return result_series


# ============================================================
#    예가범위 값 매핑 함수: 첫 행의 예가범위로 범위 값 결정
# ============================================================
def parse_range_level(range_str):
    """
    '예가범위' 값을 읽어, 미리 정의된 매핑에 따라 정수 값을 반환합니다.

    Parameters:
        range_str (str): 예가범위 문자열.

    Returns:
        int: 매핑된 범위 값 (예: 3, 2 또는 기본값 0).

    Raises:
        TypeError: 입력값이 문자열이 아닐 경우.
    """
    # import logging

    logging.info(f"✅ [parse_range_level] 시작 - 입력값: {range_str}")

    # 입력 데이터 검증
    if not isinstance(range_str, str):
        logging.error("❌ 입력 데이터는 문자열이어야 합니다!")
        raise TypeError("❌ 입력 데이터는 문자열이어야 합니다!")

    # 미리 정의된 매핑 딕셔너리: 예가범위 문자열과 대응되는 정수 값
    range_mapping = {"+3%~-3%": 3, "+2%~-2%": 2}

    # 공백 제거 후 매핑 검색
    range_level = range_mapping.get(range_str.strip(), 0)

    logging.info(f"✅ [parse_range_level] 완료 - 매핑된 값: {range_level}")

    return range_level


# ==================================================================
#   구간 하한값, 상한값 추출 함수: 예가 범위에서 시작점과 끝점 추출
# ==================================================================
def extract_bounds(range_str):
    """
    주어진 '예가범위' 문자열에서 두 개의 퍼센트 값을 추출하여,
    이들 중 최소값을 Lower, 최대값을 Upper로 반환합니다.

    Parameters:
        s (str): 예가범위 문자열 (예: "+3% ~ -3%").

    Returns:
        tuple(float, float): (Lower, Upper) - 추출된 숫자 중 최소값과 최대값.

    Raises:
        TypeError: 입력값이 문자열이 아닐 경우.
        ValueError: 입력값이 유효하지 않거나, 두 개의 퍼센트 값 추출에 실패한 경우.
    """
    # import pandas as pd
    # import re
    # import logging

    logging.info(f"✅ [extract_bounds] 시작 - 입력값: '{range_str}'")

    # 입력값 검증
    if not isinstance(range_str, str):
        logging.error("❌ 입력 데이터는 문자열이어야 합니다!")
        raise TypeError("❌ 입력 데이터는 문자열이어야 합니다!")

    if pd.isna(range_str) or range_str.strip() == "":
        logging.error(f"❌ 유효하지 않은 예가범위 값입니다: '{range_str}'")
        raise ValueError(f"❌ 유효하지 않은 예가범위 값입니다: '{range_str}'")

    range_str = range_str.strip()

    # 정규식으로 숫자(퍼센트 기호 포함) 추출
    matches = re.findall(r"([-+]?\d+(?:\.\d+)?)%", range_str)

    if len(matches) != 2:
        logging.error(f"❌ 예가범위에서 시작점과 끝점을 찾을 수 없음: '{range_str}'")
        raise ValueError(f"❌ 예가범위에서 시작점과 끝점을 찾을 수 없음: '{range_str}'")

    start = float(matches[0])
    end = float(matches[1])

    lower_bound = min(start, end)
    upper_bound = max(start, end)

    logging.info(f"✅ [extract_bounds] 완료 - 결과 (Lower, Upper): ({lower_bound}, {upper_bound})")

    return lower_bound, upper_bound


# ============================================================
#   구간 경계값 생성 함수: 예가범위에 따라 bins 생성
# ============================================================
def generate_bins(lower_bound, upper_bound):
    """
    주어진 하한(lower_bound)과 상한(upper_bound)을 기준으로 여러 개의 구간(bin) 경계값을 생성합니다.

    Parameters:
        lower_bound (float): 구간의 최소값.
        upper_bound (float): 구간의 최대값.

    Returns:
        dict: 다양한 interval 값(10, 20, 50, 100)에 대해 생성된 bins 딕셔너리.

    Raises:
        TypeError: 입력값이 숫자가 아닐 경우.
        ValueError: 중복된 구간 경계값이 발생할 경우.
    """
    # import numpy as np
    # import logging
    from collections import Counter

    logging.info(f"✅ [generate_bins] 시작 - lower_bound: {lower_bound}, upper_bound: {upper_bound}")

    # 입력값 검증
    if not isinstance(lower_bound, (int, float)) or not isinstance(upper_bound, (int, float)):
        logging.error("❌ 입력값은 숫자(float, int)여야 합니다!")
        raise TypeError("❌ 입력값은 숫자(float, int)여야 합니다!")

    if lower_bound >= upper_bound:
        logging.error("❌ lower_bound 값은 upper_bound보다 작아야 합니다!")
        raise ValueError("❌ lower_bound 값은 upper_bound보다 작아야 합니다!")

    intervals = [10, 20, 50, 100]
    bins_dict = {}

    for interval in intervals:
        # np.linspace로 구간 경계값 생성 (구간 수 = interval + 1)
        bins = np.linspace(lower_bound, upper_bound, num=interval + 1).tolist()

        # 중복 검사 (set을 활용한 최적화)
        if len(set(bins)) < len(bins):
            duplicates = [item for item, count in Counter(bins).items() if count > 1]
            logging.error(f"⚠️ 중복된 구간 경계값 발생! 중복된 값: {set(duplicates)}")
            raise ValueError(f"⚠️ 중복된 구간 경계값이 발생했습니다! 중복된 값: {set(duplicates)}")

        # 첫 번째와 마지막 경계를 -∞, ∞로 설정
        bins[0] = -np.inf
        bins[-1] = np.inf
        bins_dict[interval] = bins

        logging.info(f"✅ [generate_bins] Interval {interval}: Bins 생성 완료")

    logging.info("✅ [generate_bins] 완료 - 모든 구간 생성 완료")

    return bins_dict


# ============================================================
#   CSV에서 구간 경계값 불러오기 함수
# ============================================================
def load_bins(range_level):
    """
    지정된 range_level (2 또는 3)에 해당하는 CSV 파일에서 구간 경계값을 불러와,
    이를 딕셔너리 형태로 반환합니다.

    Parameters:
        range_level (int): 범위 값 (허용값: 2 또는 3).

    Returns:
        dict: {구간 개수: 구간 경계값 리스트} 형태의 딕셔너리.

    Raises:
        TypeError: range_level이 정수가 아닐 경우.
        ValueError: range_level이 2 또는 3이 아닐 경우.
        FileNotFoundError: 해당 CSV 파일을 찾을 수 없는 경우.
    """
    # import os
    # import pandas as pd
    # import numpy as np
    # import logging
    from pathlib import Path

    logging.info(f"✅ [load_bins] 시작 - range_level: {range_level}")

    # 입력값 검증
    if not isinstance(range_level, int):
        logging.error("❌ range_level은 정수(int)여야 합니다!")
        raise TypeError("❌ range_level은 정수(int)여야 합니다!")

    if range_level not in {2, 3}:
        logging.error("❌ 입력값 'range_level'은 2 또는 3이어야 합니다!")
        raise ValueError("❌ 입력값 'range_level'은 2 또는 3이어야 합니다!")

    intervals = [10, 20, 50, 100]
    bins_dict = {}

    for interval in intervals:
        csv_filename = f"intervals_{interval}_{range_level}.csv"
        intervals_csv_path = Path("intervals") / csv_filename

        logging.info(f"🔍 [load_bins] CSV 파일 경로: {intervals_csv_path}")

        if not intervals_csv_path.exists():
            logging.error(f"❌ 구간 경계 파일이 존재하지 않습니다: {intervals_csv_path}")
            raise FileNotFoundError(f"❌ 구간 경계 파일이 존재하지 않습니다: {intervals_csv_path}")

        # CSV 파일 읽기
        interval_df = pd.read_csv(intervals_csv_path)

        if "상한값" not in interval_df.columns:
            logging.error(f"❌ CSV 파일에서 '상한값' 컬럼을 찾을 수 없습니다: {intervals_csv_path}")
            raise KeyError(f"❌ CSV 파일에서 '상한값' 컬럼을 찾을 수 없습니다.")

        # '상한값' 정렬 후 리스트 변환
        upper_bounds = interval_df["상한값"].iloc[:-1].sort_values().tolist()
        upper_bounds = [val * 100 for val in upper_bounds]  # 퍼센트 변환

        # 첫 번째와 마지막 경계를 -∞, ∞로 설정
        bins = [-np.inf] + upper_bounds + [np.inf]
        bins_dict[interval] = bins

        logging.info(f"✅ [load_bins] {interval} 구간: {bins}")

    logging.info("✅ [load_bins] 완료 - 모든 구간 로드 완료")

    return bins_dict


# ============================================================
#   데이터 구간화 및 비율 계산 함수
# ============================================================
def data_to_target(data, bins):
    """
    데이터를 구간(bins)에 따라 분할하고, 각 구간에 속하는 비율을 빠르게 계산하는 최적화 함수.

    Parameters:
        data (list, np.array, pd.Series): 구간화할 숫자 데이터.
        bins (list): 구간 경계값 리스트 (최소 2개 이상의 값 필요).

    Returns:
        dict: {구간 레이블: 해당 구간 비율} 형태의 딕셔너리.

    Raises:
        TypeError: data가 리스트, NumPy 배열, Pandas Series가 아닐 경우.
        ValueError: bins가 최소 2개 이상의 값을 가진 리스트가 아닐 경우.
    """
    # import numpy as np
    # import pandas as pd
    # import logging

    logging.info("✅ [data_to_target] 시작")

    # 입력값 검증
    if not isinstance(data, (list, np.ndarray, pd.Series)):
        logging.error("❌ 입력 데이터는 리스트, NumPy 배열 또는 Pandas Series여야 합니다!")
        raise TypeError("❌ 입력 데이터는 리스트, NumPy 배열 또는 Pandas Series여야 합니다!")

    if not isinstance(bins, list) or len(bins) < 2:
        logging.error("❌ 입력값 'bins'는 최소 2개 이상의 값을 가진 리스트여야 합니다!")
        raise ValueError("❌ 입력값 'bins'는 최소 2개 이상의 값을 가진 리스트여야 합니다!")

    if len(data) == 0:
        logging.warning("⚠️ [data_to_target] 빈 데이터 입력됨. 빈 딕셔너리 반환.")
        return {}

    # 데이터를 NumPy 배열로 변환하여 연산 속도 향상
    data = np.asarray(data, dtype=float)

    # numpy.digitize() 사용하여 구간화
    bin_indices = np.digitize(data, bins, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)  # 범위를 벗어나는 값 보정

    # numpy.bincount() 활용하여 개수 계산
    bin_counts = np.bincount(bin_indices, minlength=len(bins) - 1)

    # 비율 계산
    total_count = len(data)
    ratios = bin_counts / total_count if total_count > 0 else bin_counts

    # 구간 레이블 생성
    labels = [f"{i + 1:03}" for i in range(len(bins) - 1)]

    result = dict(zip(labels, ratios))

    logging.info(f"✅ [data_to_target] 완료 - 결과: {result}")

    return result


# ============================================================
#   데이터 구간화 및 비율 계산 (여러 bins 사용)
# ============================================================
def process_row_fixed_bins(data, bins_dict):
    """
    여러 개의 bins를 사용하여 데이터를 구간화하고 비율을 계산하는 함수.

    Parameters:
        data (list, np.array, pd.Series): 구간화를 적용할 숫자 데이터.
        bins_dict (dict): 미리 정의된 구간 경계값 딕셔너리.

    Returns:
        dict: {구간_레이블: 비율} 형태의 딕셔너리.

    Raises:
        TypeError: data가 리스트, NumPy 배열, Pandas Series가 아닐 경우.
        TypeError: bins_dict가 딕셔너리가 아닐 경우.
    """
    # import numpy as np
    # import logging

    logging.info("✅ [process_row_fixed_bins] 시작")

    # 입력값 검증
    if not isinstance(data, (list, np.ndarray, pd.Series)):
        logging.error("❌ 입력 데이터는 리스트, NumPy 배열 또는 Pandas Series여야 합니다!")
        raise TypeError("❌ 입력 데이터는 리스트, NumPy 배열 또는 Pandas Series여야 합니다!")

    if not isinstance(bins_dict, dict):
        logging.error("❌ 입력값 'bins_dict'는 딕셔너리여야 합니다!")
        raise TypeError("❌ 입력값 'bins_dict'는 딕셔너리여야 합니다!")

    if not data:
        logging.warning("⚠️ [process_row_fixed_bins] 빈 데이터 입력됨. 빈 딕셔너리 반환.")
        return {}

    if not bins_dict:
        logging.warning("⚠️ [process_row_fixed_bins] 빈 bins_dict 입력됨. 빈 딕셔너리 반환.")
        return {}

    row_result = {}

    # data_to_target()을 반복 호출하지 않고, dict.update() 사용
    for bin_size, bins in bins_dict.items():
        bin_counts = data_to_target(data, bins)
        row_result.update({f"{bin_size:03}_{key}": value for key, value in bin_counts.items()})

    logging.info(f"✅ [process_row_fixed_bins] 완료 - 결과: {row_result}")

    return row_result


# ============================================================
#   데이터 구간화 및 목표값 계산
# ============================================================
def calculate_target(df):
    """
    입력 데이터프레임(df)의 '예가범위' 값을 기준으로 데이터를 구간화하고 비율을 계산하는 함수.

    Parameters:
        df (pd.DataFrame): 입력 데이터.

    Returns:
        pd.DataFrame: 구간화된 비율 정보를 포함한 새로운 데이터프레임.

    Raises:
        TypeError: df가 Pandas DataFrame이 아닐 경우.
    """
    # import pandas as pd
    # import logging

    logging.info("✅ [calculate_target] 시작")

    # 입력값 검증
    if not isinstance(df, pd.DataFrame):
        logging.error("❌ 입력 데이터는 Pandas DataFrame이어야 합니다!")
        raise TypeError("❌ 입력 데이터는 Pandas DataFrame이어야 합니다!")

    # 예가범위 레벨 부여
    range_level = parse_range_level(df["예가범위"].iloc[0])

    # df_bins = pd.DataFrame()  # 빈 DataFrame 생성
    rows = []  # 성능 최적화를 위한 리스트 사용

    if range_level in {2, 3}:
        logging.info(f"✅ [calculate_target] 예가범위 레벨: {range_level}")

        bins_dict = load_bins(range_level)

        for row in df.itertuples():
            row_result = process_row_fixed_bins(getattr(row, "사정률"), bins_dict)
            rows.append(row_result)

    elif range_level == 0:
        logging.info("✅ [calculate_target] 예가범위 레벨: 0 ")

        for row in df.itertuples():
            lower_bound, upper_bound = extract_bounds(row.예가범위)
            bins_dict = generate_bins(lower_bound, upper_bound)
            row_result = process_row_fixed_bins(getattr(row, "사정률"), bins_dict)
            rows.append(row_result)

    # 리스트를 DataFrame으로 변환하여 df_bins 생성
    if rows:
        df_bins = pd.DataFrame(rows)

    # 원본 데이터와 df_bins 병합
    result_df = pd.concat([df.reset_index(drop=True), df_bins.reset_index(drop=True)], axis=1)

    logging.info("✅ [calculate_target] 완료 - 데이터 변환 완료")

    return result_df


### 데이터 변환

# ==================================================================================================
#   데이터 전처리 함수: 로그 변환, 정규화, 원-핫 인코딩, 텍스트 임베딩, 차원 축소, 구간 경쟁률 계산
# ==================================================================================================
def process_data(df):
    """
    데이터 전처리 함수: 로그 변환, 정규화, 원-핫 인코딩, 텍스트 임베딩, 차원 축소 수행.

    Parameters:
        df (pd.DataFrame): 원본 데이터프레임.

    Returns:
        pd.DataFrame: 전처리된 데이터프레임.

    Raises:
        TypeError: df가 Pandas DataFrame이 아닐 경우.
        ValueError: 필요한 컬럼이 누락된 경우.
    """
    # import pandas as pd
    # import logging

    logging.info("✅ [process_data] 시작")

    # 입력값 검증
    if not isinstance(df, pd.DataFrame):
        logging.error("❌ 입력 데이터는 Pandas DataFrame이어야 합니다!")
        raise TypeError("❌ 입력 데이터는 Pandas DataFrame이어야 합니다!")

    required_columns = [
        '공고번호', '공고제목', '발주처(수요기관)', '지역제한', '기초금액', '예정가격',
        '예가범위', 'A값', '투찰률(%)', '참여업체수', '공고구분표시', "사정률"
    ]

    # 컬럼 존재 여부 확인
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"❌ 다음 컬럼이 데이터프레임에 없습니다: {missing_columns}")
        raise ValueError(f"❌ 다음 컬럼이 데이터프레임에 없습니다: {missing_columns}")

    # 데이터프레임 복사 (원본 보호)
    copy_df = df.copy()

    # 로그 변환 및 정규화 적용
    numeric_cols = ["기초금액", "예정가격", "투찰률(%)"]
    for col in numeric_cols:
        copy_df[col] = log_transforming(copy_df[col])
        copy_df[f"norm_log_{col}"] = normalizing(copy_df[col])

    # A값 처리
    copy_df['A값'] = copy_df['A값'].astype(float) / (1 + copy_df['기초금액'].astype(float))
    copy_df["norm_A값/기초금액"] = normalizing(copy_df['A값'])

    # 원-핫 인코딩을 위한 컬럼
    categorical_cols = ["발주처(수요기관)", "지역제한", "공고구분표시"]
    encoded_dfs = [one_hot_encoding(copy_df[col]) for col in categorical_cols]

    # 원본 데이터와 원-핫 인코딩 결과 병합
    copy_df = pd.concat([copy_df] + encoded_dfs, axis=1)

    # 텍스트 임베딩 적용
    logging.info("✅ [process_data] 텍스트 임베딩 모델 로드 중...")
    tokenizer, model, device = load_bge_model(model_name="BAAI/bge-m3")

    logging.info("✅ [process_data] 공고제목 임베딩 수행...")
    copy_df["embedding_공고제목"] = embedding(copy_df["공고제목"].fillna(""), tokenizer, model, device)

    # 차원 축소 수행
    logging.info("✅ [process_data] 임베딩 벡터 차원 축소 중...")
    expanded_df = series_to_dataframe(copy_df["embedding_공고제목"])
    tmp_df = dimension_reducing_PCA(expanded_df, "공고제목", 100)
    reduced_df = dimension_reducing_UMAP(tmp_df, "공고제목", 20)

    # 병합
    copy_df = pd.concat([copy_df.reset_index(drop=True), reduced_df.reset_index(drop=True)], axis=1)

    print(f"🔍 병합 후 결측치가 포함된 행:\n{copy_df[copy_df.isna().any(axis=1)]}")

    # 구간 경쟁률 계산
    logging.info("✅ [process_data] 구간 경쟁률 계산 중...")
    copy_df = calculate_target(copy_df)

    # 기존 칼럼 삭제
    copy_df.drop(['공고제목', '발주처(수요기관)', '지역제한', '기초금액', '예정가격',
                  'A값', '투찰률(%)', '참여업체수', '공고구분표시', "사정률", "embedding_공고제목"], axis=1, inplace=True)

    # 최종 결과 반환
    logging.info("✅ [process_data] 완료 - 데이터 전처리 완료")
    return copy_df.copy()


# ============================================================
# 24. 전체 데이터셋 변환 함수: 데이터 클렌징부터 Feature 및 Target 처리까지
# ============================================================
def transform(df1, df2):
    """
    공고 데이터(df1)와 투찰 데이터(df2)를 입력받아, 데이터 클렌징, Feature 처리, Target 처리
    순으로 실행한 후, 최종적으로 세 개의 데이터셋(Dataset_3_df, Dataset_2_df, Dataset_etc_df)을 생성하여 반환합니다.

    Parameters:
        df1 (pd.DataFrame): 공고 데이터를 포함한 DataFrame.
        df2 (pd.DataFrame): 투찰 데이터를 포함한 DataFrame.

    Returns:
        dict: {
            "DataSet_3": DataFrame,  # '예가범위'가 +3%~-3%인 데이터셋
            "DataSet_2": DataFrame,  # '예가범위'가 +2%~-2%인 데이터셋
            "DataSet_etc": DataFrame # '예가범위'가 그 외인 데이터셋
        }

    Raises:
        TypeError: 입력 데이터가 Pandas DataFrame이 아닐 경우.
    """
    # import logging
    # import pandas as pd

    logging.info("✅ [transform] 시작")

    if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
        logging.error("❌ 입력 데이터는 Pandas DataFrame이어야 합니다!")
        raise TypeError("❌ 입력 데이터는 Pandas DataFrame이어야 합니다!")

    # 원본 데이터 복사 및 클렌징 수행
    notices_df = df1.copy()
    bids_df = df2.copy()

    # 공고 & 투찰 데이터 재구성
    merged_df = restructure_data(notices_df, bids_df)

    # 예가범위별 분리
    df_dict = separate_data(merged_df)
    df_3 = df_dict.get("range3")
    df_2 = df_dict.get("range2")
    df_etc = df_dict.get("others")

    # 데이터 전처리 (process_data)
    Dataset_3_df = process_data(df_3)
    Dataset_2_df = process_data(df_2)
    Dataset_etc_df = process_data(df_etc)

    Dataset_dict = {
        "DataSet_3": Dataset_3_df,
        "DataSet_2": Dataset_2_df,
        "DataSet_etc": Dataset_etc_df
    }

    logging.info("🎯 [transform] 완료 - 데이터 변환 완료")

    return Dataset_dict


def load(dataset_dict):
    copy_dict = dataset_dict.copy()

    DataSet_3 = copy_dict.get("DataSet_3")
    DataSet_2 = copy_dict.get("DataSet_2")
    DataSet_etc = copy_dict.get("DataSet_etc")

    DataSet_3.to_csv("DataSet_3.csv", index=False)
    DataSet_2.to_csv("DataSet_2.csv", index=False)
    DataSet_etc.to_csv("DataSet_etc.csv", index=False)