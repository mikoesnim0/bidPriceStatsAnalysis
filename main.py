"""
전체 머신러닝 파이프라인 실행 스크립트
"""
import os
import argparse
import pandas as pd
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import time
from tqdm import tqdm
from pathlib import Path
import sys
import numpy as np

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src import config, data_processing, train, evaluate, utils

def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='AutoGluon ML Pipeline')
    parser.add_argument('--data-only', action='store_true', help='데이터 전처리만 실행')
    parser.add_argument('--train-only', action='store_true', help='모델 학습만 실행')
    parser.add_argument('--evaluate-only', action='store_true', help='모델 평가만 실행')
    parser.add_argument('--num-targets', type=int, default=30, help='처리할 타겟 컬럼 수 (기본값: 30)')
    parser.add_argument('--dataset-key', type=str, default='2', help='사용할 데이터셋 키 (기본값: 2)')
    parser.add_argument('--target-prefixes', type=str, default='010,020,050,100', help='처리할 타겟 접두사, 콤마로 구분 (기본값: 010,020,050,100)')
    parser.add_argument('--gpu', type=str, default='True', help='GPU 사용 여부 (True/False)')
    parser.add_argument('--models', type=str, default=None, help='사용할 모델 목록 (콤마로 구분)')
    parser.add_argument('--preset', type=str, default='medium_quality_faster_train', 
                        help='AutoGluon 프리셋')
    parser.add_argument('--verbose', type=int, default=1, help='출력 상세 수준 (0: 간략, 1: 기본, 2: 상세)')
    parser.add_argument('--generate-targets', action='store_true', help='타겟 컬럼 자동 생성 (없을 경우)')
    args = parser.parse_args()
    
    # GPU 설정 처리
    use_gpu = args.gpu.lower() == 'true'
    
    # 데이터셋 키 및 타겟 접두사 처리
    dataset_key = args.dataset_key
    target_prefixes = args.target_prefixes.split(',')
    
    # 모델 목록 처리
    selected_models = None
    if args.models:
        selected_models = args.models.split(',')
    
    # 로깅 설정
    name = "AutoGluon ML Pipeline"
    start_time = time.time()
    name = f"{name} - {time.strftime('%Y%m%d_%H%M%S')}"
    logger = utils.setup_logger(name)
    logger.info("=== AutoGluon ML Pipeline Started ===")
    print("\n🚀 BidPrice 예측 파이프라인을 시작합니다...\n")
    
    try:
        # 1. 데이터 로드 및 전처리
        if not args.train_only and not args.evaluate_only:
            logger.info("Step 1: Loading and preprocessing data...")
            print("📊 데이터 전처리 단계를 시작합니다...")
            
            # MongoDB에서 직접 데이터 로드
            try:
                from statsModelsPredict.src.db_config.mongodb_handler import MongoDBHandler
                
                # MongoDB 연결
                handler = MongoDBHandler(db_name='data_preprocessed')
                handler.connect()
                
                # 컬렉션 목록 확인
                collections = handler.db.list_collection_names()
                print(f"사용 가능한 컬렉션: {collections}")
                
                # 데이터셋에 맞는 컬렉션 형식 준비
                train_collections = [
                    f"preprocessed_dataset{dataset_key}_train",  # 기본 형식
                    f"preprocessed_dataset_{dataset_key}_train", # 언더스코어 형식 
                    f"preprocessed_{dataset_key}_train",        # 접두사 다른 형식
                    "preprocessed_train"                        # 기본 컬렉션
                ]
                
                test_collections = [
                    f"preprocessed_dataset{dataset_key}_test",  # 기본 형식
                    f"preprocessed_dataset_{dataset_key}_test", # 언더스코어 형식
                    f"preprocessed_{dataset_key}_test",        # 접두사 다른 형식
                    "preprocessed_test"                        # 기본 컬렉션
                ]
                
                # 사용 가능한 첫 번째 컬렉션 선택
                train_collection = next((c for c in train_collections if c in collections), None)
                test_collection = next((c for c in test_collections if c in collections), None)
                
                if train_collection is None:
                    print("경고: 학습 데이터 컬렉션을 찾을 수 없습니다. 임의 컬렉션을 사용합니다.")
                    train_collection = collections[0] if collections else None
                
                if test_collection is None:
                    print("경고: 테스트 데이터 컬렉션을 찾을 수 없습니다. 학습 데이터로 대체합니다.")
                    test_collection = train_collection
                
                if train_collection is None:
                    raise ValueError("데이터 컬렉션을 찾을 수 없습니다.")
                
                print(f"학습 데이터 컬렉션: {train_collection}")
                print(f"테스트 데이터 컬렉션: {test_collection}")
                
                # 학습 데이터 로드
                print(f"컬렉션 {train_collection}에서 학습 데이터 로드 중...")
                train_data_list = list(handler.db[train_collection].find({}, {'_id': 0}))
                if not train_data_list:
                    raise ValueError(f"컬렉션 {train_collection}에 데이터가 없습니다.")
                train_data = pd.DataFrame(train_data_list)
                
                # 테스트 데이터 로드
                print(f"컬렉션 {test_collection}에서 테스트 데이터 로드 중...")
                test_data_list = list(handler.db[test_collection].find({}, {'_id': 0}))
                if not test_data_list:
                    print(f"경고: 컬렉션 {test_collection}에 데이터가 없습니다. 학습 데이터로 대체합니다.")
                    test_data = train_data.copy()
                else:
                    test_data = pd.DataFrame(test_data_list)
                
                # 연결 종료
                handler.close()
                
            except Exception as e:
                logger.error(f"MongoDB에서 데이터 로드 실패: {str(e)}")
                raise
            
            # 타겟 컬럼 식별 또는 생성
            def identify_or_create_targets(data, target_prefixes, generate=False):
                # 타겟 컬럼 식별 (010_, 020_, 050_, 100_ 로 시작하는 컬럼)
                target_columns = []
                for prefix in target_prefixes:
                    prefix_cols = [col for col in data.columns if col.startswith(f"{prefix}_")]
                    target_columns.extend(prefix_cols)
                
                if not target_columns and generate:
                    print("타겟 컬럼을 찾을 수 없어 자동 생성합니다...")
                    
                    # 가상의 타겟 컬럼 생성
                    # 기준 컬럼 (예: norm_log_기초금액)을 기반으로 변형하여 타겟 생성
                    base_col = next((col for col in data.columns if 'norm_log' in col), None)
                    
                    if base_col is None:
                        # 아무 수치형 컬럼이나 사용
                        numeric_cols = data.select_dtypes(include=['number']).columns
                        if not len(numeric_cols):
                            raise ValueError("타겟 생성에 사용할 수치형 컬럼이 없습니다.")
                        base_col = numeric_cols[0]
                    
                    # 각 타겟 접두사에 대해 여러 타겟 생성
                    target_df = pd.DataFrame(index=data.index)
                    for prefix in target_prefixes:
                        for i in range(1, 6):  # 각 접두사별로 5개 생성
                            col_name = f"{prefix}_{i:03d}"
                            # 기본 컬럼에 무작위성 추가
                            target_df[col_name] = data[base_col] * (0.8 + np.random.rand() * 0.4) + np.random.randn(len(data)) * 0.1
                    
                    # 두 데이터프레임 결합
                    result_data = pd.concat([data, target_df], axis=1)
                    target_columns = target_df.columns.tolist()
                    
                    return result_data, target_columns
                
                return data, target_columns
            
            # 전처리 단계 진행 표시기
            preprocessing_steps = ['데이터 로드', '타겟 처리', '결측치 처리', '데이터 분할', '데이터 저장']
            preprocess_pbar = tqdm(preprocessing_steps, desc="📊 데이터 전처리", position=0, leave=True)
            
            # 데이터 로드 완료
            preprocess_pbar.set_description("📊 데이터 로드 중")
            preprocess_pbar.update(1)
            
            # 타겟 처리
            preprocess_pbar.set_description("📊 타겟 처리 중")
            train_data, train_target_columns = identify_or_create_targets(
                train_data, target_prefixes, generate=args.generate_targets
            )
            test_data, test_target_columns = identify_or_create_targets(
                test_data, target_prefixes, generate=args.generate_targets
            )
            
            # 공통 타겟 컬럼만 사용
            common_targets = sorted(list(set(train_target_columns) & set(test_target_columns)))
            if not common_targets:
                raise ValueError("학습 및 테스트 데이터에 공통된 타겟 컬럼이 없습니다!")
            
            print(f"사용할 타겟 컬럼: {len(common_targets)}개")
            if args.verbose > 1:
                print(f"타겟 컬럼 목록: {common_targets}")
            
            # 특성과 타겟 분리
            train_X = train_data.drop(columns=common_targets, errors='ignore')
            train_Y = train_data[common_targets]
            test_X = test_data.drop(columns=common_targets, errors='ignore')
            test_Y = test_data[common_targets]
            
            preprocess_pbar.update(1)
            
            # 결측치 처리
            preprocess_pbar.set_description("📊 결측치 처리 중")
            
            # NaN 값 확인
            nan_count_X_train = train_X.isna().sum().sum()
            nan_count_Y_train = train_Y.isna().sum().sum()
            
            if nan_count_X_train > 0:
                print(f"학습 특성 데이터에 {nan_count_X_train}개의 NaN 값이 있습니다. 처리 중...")
                # 수치형 컬럼은 중앙값으로, 범주형 컬럼은 최빈값으로 채우기
                for col in train_X.columns:
                    if pd.api.types.is_numeric_dtype(train_X[col]):
                        train_X[col] = train_X[col].fillna(train_X[col].median())
                    else:
                        train_X[col] = train_X[col].fillna(train_X[col].mode()[0] if not train_X[col].mode().empty else "UNKNOWN")
            
            if nan_count_Y_train > 0:
                print(f"학습 타겟 데이터에 {nan_count_Y_train}개의 NaN 값이 있습니다. 해당 행 제거 중...")
                # 타겟 값에 NaN이 있는 행 제거
                nan_rows = train_Y.isna().any(axis=1)
                train_X = train_X[~nan_rows].reset_index(drop=True)
                train_Y = train_Y[~nan_rows].reset_index(drop=True)
            
            # 테스트 데이터도 동일하게 처리
            nan_count_X_test = test_X.isna().sum().sum()
            nan_count_Y_test = test_Y.isna().sum().sum()
            
            if nan_count_X_test > 0:
                print(f"테스트 특성 데이터에 {nan_count_X_test}개의 NaN 값이 있습니다. 처리 중...")
                for col in test_X.columns:
                    if pd.api.types.is_numeric_dtype(test_X[col]):
                        test_X[col] = test_X[col].fillna(test_X[col].median())
                    else:
                        test_X[col] = test_X[col].fillna(test_X[col].mode()[0] if not test_X[col].mode().empty else "UNKNOWN")
            
            if nan_count_Y_test > 0:
                print(f"테스트 타겟 데이터에 {nan_count_Y_test}개의 NaN 값이 있습니다. 해당 행 제거 중...")
                nan_rows = test_Y.isna().any(axis=1)
                test_X = test_X[~nan_rows].reset_index(drop=True)
                test_Y = test_Y[~nan_rows].reset_index(drop=True)
            
            preprocess_pbar.update(1)
            
            # 공통 특성 컬럼만 사용
            common_features = sorted(list(set(train_X.columns) & set(test_X.columns)))
            if not common_features:
                raise ValueError("학습 및 테스트 데이터에 공통된 특성 컬럼이 없습니다!")
            
            train_X = train_X[common_features]
            test_X = test_X[common_features]
            
            # 데이터 형식 일치 확인
            for col in common_features:
                if train_X[col].dtype != test_X[col].dtype:
                    # 형식이 다르면 문자열로 통일
                    print(f"컬럼 {col}의 데이터 형식이 일치하지 않아 문자열로 변환합니다.")
                    train_X[col] = train_X[col].astype(str)
                    test_X[col] = test_X[col].astype(str)
            
            # 데이터 저장
            preprocess_pbar.set_description("📊 데이터 저장 중")
            train_X, test_X, train_Y, test_Y = data_processing.split_and_save_data(train_X, train_Y, test_X, test_Y)
            preprocess_pbar.update(1)
            
            print(f"\n✅ 데이터 전처리 완료! 학습 데이터: {train_X.shape}, 테스트 데이터: {test_X.shape}\n")
            
            if args.data_only:
                logger.info("Data processing only mode - exiting")
                print("✅ 데이터 전처리만 수행하도록 설정되어 종료합니다.")
                return
        else:
            logger.info("Loading preprocessed data...")
            print("💾 저장된 전처리 데이터를 로드합니다...")
            
            # 전처리된 데이터 파일 경로 확인
            train_file = os.path.join(config.DATA_DIR, "train_data.csv")
            test_file = os.path.join(config.DATA_DIR, "test_data.csv")
            train_y_file = os.path.join(config.DATA_DIR, "train_targets.csv")
            test_y_file = os.path.join(config.DATA_DIR, "test_targets.csv")
            
            # 파일 존재 확인
            required_files = [train_file, test_file, train_y_file, test_y_file]
            for file_path in required_files:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"필수 데이터 파일을 찾을 수 없습니다: {file_path}")
            
            # 데이터 로드
            train_X = pd.read_csv(train_file)
            test_X = pd.read_csv(test_file)
            train_Y = pd.read_csv(train_y_file)
            test_Y = pd.read_csv(test_y_file)
            
            print(f"✅ 데이터 로드 완료! 학습 데이터: {train_X.shape}, 테스트 데이터: {test_X.shape}\n")
        
        # 2. 모델 학습
        if not args.data_only and not args.evaluate_only:
            logger.info("Step 2: Training models...")
            print("🧠 모델 학습 단계를 시작합니다...")
            
            # 타겟 접두사별로 처리
            total_targets = 0
            model_paths = []
            
            for target_prefix in target_prefixes:
                # 해당 접두사로 시작하는 타겟 컬럼들 찾기
                prefix_targets = [col for col in train_Y.columns if col.startswith(f"{target_prefix}_")]
                
                if not prefix_targets:
                    logger.warning(f"⚠️ 접두사 '{target_prefix}_'로 시작하는 타겟 컬럼이 없습니다.")
                    continue
                
                # 제한된 개수만 처리
                if args.num_targets is not None:
                    prefix_targets = prefix_targets[:args.num_targets]
                
                total_targets += len(prefix_targets)
                
                # 타겟별 학습 진행 표시기
                train_pbar = tqdm(total=len(prefix_targets), 
                                  desc=f"🧠 {target_prefix} 모델 학습", 
                                  position=0, 
                                  leave=True)
                
                # 각 타겟에 대해 모델 학습
                for i, target_col in enumerate(prefix_targets):
                    train_pbar.set_description(f"🧠 [{i+1}/{len(prefix_targets)}] {target_col} 학습 중")
                    
                    # GPU 사용 여부 표시
                    if use_gpu and args.verbose > 0:
                        print(f"  🔥 GPU를 사용하여 {target_col} 학습 중...")
                    
                    try:
                        # 단일 타겟 학습
                        model_path = train.train_single_target_model(
                            X_train=train_X,
                            Y_train=train_Y, 
                            target_col=target_col,
                            dataset_key=dataset_key,
                            use_gpu=use_gpu,
                            selected_models=selected_models,
                            preset=args.preset
                        )
                        model_paths.append(model_path)
                        
                        # 학습 결과 간략 출력
                        if args.verbose > 0:
                            print(f"  ✅ {target_col} 모델 학습 완료: {model_path}")
                    
                    except Exception as e:
                        logger.error(f"❌ {target_col} 모델 학습 중 오류 발생: {str(e)}")
                        print(f"  ❌ {target_col} 학습 실패: {str(e)}")
                    
                    train_pbar.update(1)
                
                train_pbar.close()
            
            print(f"\n✅ 총 {len(model_paths)}/{total_targets}개 타겟에 대한 모델 학습 완료!\n")
            
            if args.train_only:
                logger.info("Training only mode - exiting")
                print("✅ 모델 학습만 수행하도록 설정되어 종료합니다.")
                return
        
        # 3. 모델 평가
        if not args.data_only and not args.train_only:
            logger.info("Step 3: Evaluating models...")
            print("📈 모델 평가 단계를 시작합니다...")
            
            # 평가할 타겟 수 결정
            target_columns = test_Y.columns
            if args.num_targets is not None:
                target_columns = target_columns[:args.num_targets]
            
            # 타겟별 평가 진행 표시기
            eval_pbar = tqdm(total=len(target_columns), desc="📈 모델 평가", position=0, leave=True)
            
            all_results = []
            for i, target_col in enumerate(target_columns):
                eval_pbar.set_description(f"📈 [{i+1}/{len(target_columns)}] {target_col} 평가 중")
                
                # 모델 경로 설정
                model_path = os.path.join(config.MODELS_DIR, target_col)
                if not os.path.exists(model_path):
                    logger.warning(f"Model for {target_col} not found at {model_path}")
                    eval_pbar.update(1)
                    continue
                
                # 평가 과정 표시기 (상세 모드에서만)
                if args.verbose > 1:
                    eval_steps = ['모델 로드', '예측 수행', '성능 계산', '시각화 생성', '결과 저장']
                    eval_step_pbar = tqdm(eval_steps, desc=f"  {target_col} 평가", position=1, leave=False)
                    
                    # 각 평가 단계 시각화
                    for step in eval_steps:
                        eval_step_pbar.set_description(f"  {step} 중")
                        time.sleep(0.5)  # 실제로는 필요 없지만 진행 상황을 보여주기 위한 지연
                        eval_step_pbar.update(1)
                    
                    eval_step_pbar.close()
                
                # 모델 평가
                results = evaluate.evaluate_model(
                    model_path=model_path, 
                    test_X=test_X, 
                    test_Y=test_Y, 
                    target_col=target_col
                )
                
                # 간략한 결과 출력
                if args.verbose > 0:
                    best_model = utils.get_best_model(results, 'r2_score')
                    best_r2 = results[results['model'] == best_model]['r2_score'].values[0]
                    print(f"  ✅ {target_col} 평가 완료 - 최고 모델: {best_model} (R²: {best_r2:.4f})")
                
                all_results.append(results)
                eval_pbar.update(1)
            
            eval_pbar.close()
            
            # 결과 결합
            if all_results:
                combined_results = pd.concat(all_results, ignore_index=True)
                
                # 요약 저장
                summary_path = os.path.join(config.RESULTS_DIR, "all_models_evaluation.csv")
                combined_results.to_csv(summary_path, index=False)
                
                # 평균 성능 출력
                avg_performance = combined_results.groupby('model')[config.METRICS].mean()
                print("\n📊 모델별 평균 성능:")
                print(avg_performance)
                
                print(f"\n✅ 총 {len(target_columns)}개 타겟에 대한 모델 평가 완료!")
                print(f"📄 종합 결과가 {summary_path}에 저장되었습니다.\n")
            else:
                print("\n⚠️ 평가할 모델이 없습니다. 먼저 모델을 학습해주세요.\n")
        
        logger.info("=== Pipeline completed successfully ===")
        print("\n🎉 전체 파이프라인이 성공적으로 완료되었습니다!")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {e}", exc_info=True)
        print(f"\n❌ 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 