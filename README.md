# 입찰가 데이터 분석 파이프라인

입찰 공고 및 입찰가 데이터를 전처리하고 분석하는 파이프라인입니다. 원시 데이터를 로드하여 정제, 변환, 특성 엔지니어링 과정을 거쳐 MongoDB에 저장하는 전체 과정을 자동화합니다.

## 주요 기능

- **데이터 로드**: 다양한 형식(CSV, Excel, JSON)의 원시 데이터 파일 로드
- **데이터 정제**: 결측치 처리, 중복 제거, 이상치 탐지 및 처리
- **데이터 변환**: 로그 변환, 정규화, 원-핫 인코딩 등
- **특성 엔지니어링**: 텍스트 특성 추출, 차원 축소, 특성 조합
- **입찰가 특화 처리**: 입찰가 관련 특성 자동 생성
- **MongoDB 저장**: 전처리된 데이터를 MongoDB에 저장
- **파이프라인 시각화**: 전체 파이프라인 과정 시각화 및 보고서 생성

## 시스템 요구사항

- Python 3.8 이상
- MongoDB 4.4 이상
- 필수 라이브러리: 아래 설치 방법 참조

## 설치 방법

### 1. 저장소 복제

```bash
git clone <repository-url>
cd bidPriceStatsAnalysis
```

### 2. 가상환경 설정

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. 필수 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정

`.env` 파일에 필요한 환경 변수를 설정합니다:

```
# MongoDB Connection Settings
MONGO_URI=mongodb://admin:password@localhost:27017/gfcon?authSource=admin
MONGO_DB=gfcon
MONGO_COLLECTION_PREFIX=preprocessed

# Data Directory
DATA_DIR=./data/raw
```

## 사용 방법

### 1. 샘플 데이터 생성

테스트를 위한 샘플 데이터를 생성합니다:

```bash
python src/generate_sample_data.py --bid-records 100 --notice-records 150
```

옵션:
- `--bid-records`: 생성할 입찰 데이터 레코드 수
- `--notice-records`: 생성할 공고 데이터 레코드 수
- `--output-dir`: 출력 디렉토리 경로
- `--skip-bid`: 입찰 데이터 생성 건너뛰기
- `--skip-notice`: 공고 데이터 생성 건너뛰기

### 2. 전처리 파이프라인 실행

데이터 전처리 및 MongoDB 업로드를 실행합니다:

```bash
python src/preprocess_upload_mongo.py --file-pattern="*.csv" --generate-report
```

옵션:
- `--data-dir`: 데이터 디렉토리 경로
- `--file-pattern`: 처리할 파일 패턴 (기본값: *.csv)
- `--output-dir`: 결과 저장 디렉토리
- `--skip-upload`: MongoDB 업로드 건너뛰기
- `--check-only`: MongoDB 데이터 확인만 수행
- `--clear-db`: 기존 MongoDB 컬렉션 제거 후 새로 저장
- `--generate-report`: 파이프라인 보고서 생성
- `--show-visualization`: 파이프라인 시각화 표시

### 3. MongoDB 데이터 확인

MongoDB에 저장된 데이터를 확인합니다:

```bash
python src/preprocess_upload_mongo.py --check-only
```

## 파이프라인 데이터 흐름

### 입력 데이터

파이프라인은 두 가지 주요 데이터 유형을 처리합니다:

1. **입찰 데이터 (bid_data_*.csv)**
   - 필수 컬럼: `공고번호`, `기초금액`, `예정금액`, `투찰가` 등
   - 특징: 실제 입찰 정보, 낙찰 정보 포함

2. **공고 데이터 (notice_data_*.csv)**
   - 필수 컬럼: `공고번호`, `기초금액`, `공고제목`, `공고내용` 등
   - 특징: 공고 정보, 발주처 정보 포함

### 처리 단계별 변환

```
원시 데이터 파일 → 데이터 로드 → 데이터 정제 → 데이터 변환 → 특성 엔지니어링 → 입찰가 특화 처리 → MongoDB 저장
```

각 단계에서 다음과 같은 변환이 이루어집니다:

1. **데이터 로드**
   - CSV, Excel, JSON 등 다양한 형식 지원
   - 컬럼 이름, 데이터 타입 자동 감지

2. **데이터 정제**
   - 결측치 처리: 필수 컬럼은 행 제거, 나머지는 적절한 값으로 대체
   - 중복 제거: `공고번호` 기준으로 중복 행 제거
   - 이상치 탐지 및 처리: 금액 컬럼의 이상치 조정

3. **데이터 변환**
   - 로그 변환: `기초금액`, `예정금액`, `투찰가` 등에 자연로그 적용
   - 정규화: 수치형 컬럼에 표준화 적용
   - 원-핫 인코딩: 범주형 변수에 적용

4. **특성 엔지니어링**
   - 텍스트 특성 추출: `공고제목`, `공고내용`에서 TF-IDF 특성 추출
   - 차원 축소: PCA를 통한 수치형 특성 차원 축소
   - 특성 조합: 기존 특성을 조합하여 새로운 특성 생성

5. **입찰가 특화 처리**
   - 낙찰가격비율: `투찰가/예정금액` 계산
   - 예정가비율: `예정금액/기초금액` 계산
   - 기타 입찰가 관련 비율 특성 생성

6. **MongoDB 저장**
   - 데이터셋 유형별 컬렉션 저장:
     - `preprocessed_3`: 3개 이상 입찰건 참여 업체 데이터
     - `preprocessed_2`: 2개 입찰건 참여 업체 데이터
     - `preprocessed_etc`: 기타 데이터

### 출력 데이터

전처리 결과는 MongoDB에 다음과 같은 구조로 저장됩니다:

- **데이터베이스**: `gfcon` (환경 변수에서 설정 가능)
- **컬렉션**: `preprocessed_3`, `preprocessed_2`, `preprocessed_etc`
- **문서 구조**:
  - 원본 컬럼: `공고번호`, `공고제목` 등
  - 변환 컬럼: `norm_log_기초금액`, `norm_log_예정금액` 등
  - 생성 특성: `낙찰가격비율`, `예정가비율`, `입찰일자_year` 등
  - 임베딩 특성: `TFIDF_공고제목_*`, `PCA_*` 등

## 프로젝트 구조

```
bidPriceStatsAnalysis/
├── data/
│   ├── raw/                  # 원시 데이터 저장
│   └── processed/            # 중간 처리 결과 (필요시)
├── results/                  # 결과 저장 (보고서, 시각화 등)
├── src/
│   ├── preprocessing/        # 전처리 모듈
│   │   ├── cleaner.py        # 데이터 정제
│   │   ├── transformer.py    # 데이터 변환
│   │   └── feature_eng.py    # 특성 엔지니어링
│   ├── data_loader.py        # 데이터 로드
│   ├── mongodb_handler.py    # MongoDB 연결 및 처리
│   ├── preprocess_pipeline.py # 메인 파이프라인
│   ├── preprocess_upload_mongo.py # 실행 스크립트
│   ├── pipeline_visualizer.py # 파이프라인 시각화
│   └── generate_sample_data.py # 샘플 데이터 생성
├── .env                      # 환경 변수
├── README.md                 # 사용 설명서
└── requirements.txt          # 필수 패키지
```

## 파이프라인 확장

이 파이프라인은 다음과 같은 방식으로 확장할 수 있습니다:

1. **새로운 전처리 기법 추가**
   - `src/preprocessing/` 디렉토리의 적절한 파일에 새 메서드 추가
   - `preprocess_pipeline.py`에서 설정 및 호출 로직 업데이트

2. **새로운 데이터 유형 지원**
   - `preprocess_pipeline.py`의 `_default_config` 메서드에 데이터 유형 추가
   - 필요한 전처리 단계 정의

3. **추가 분석 기능**
   - 새로운 분석 스크립트를 추가하여 MongoDB에서 데이터 로드 및 분석

## 문제 해결

### MongoDB 연결 문제

- MongoDB가 실행 중인지 확인
- `.env` 파일의 연결 정보가 올바른지 확인
- 방화벽 설정 확인
- `mongo` 명령어로 직접 연결 테스트

### 데이터 처리 오류

- 필수 컬럼이 원시 데이터에 존재하는지 확인
- 데이터 타입 변환 문제가 없는지 확인
- 파일 인코딩 문제 확인 (UTF-8 권장)

### 공간 문제

- 대용량 데이터 처리 시 디스크 공간 확인
- MongoDB 저장 공간 확인

## 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE)에 따라 배포됩니다.
