#!/usr/bin/env python
"""
입찰가 데이터 전처리 및 MongoDB 업로드 스크립트
"""
import os
import argparse
import time
import logging
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from preprocess_pipeline import PreprocessingPipeline
from mongodb_handler import MongoDBHandler
from pipeline_visualizer import create_visualizer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess_upload.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='입찰가 데이터 전처리 및 MongoDB 업로드')
    parser.add_argument('--data-dir', type=str, default=None, 
                        help='데이터 디렉토리 경로 (기본값: 환경 변수 또는 ./data)')
    parser.add_argument('--file-pattern', type=str, default='*.csv',
                        help='처리할 파일 패턴 (기본값: *.csv)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='결과 저장 디렉토리 (기본값: results)')
    parser.add_argument('--skip-upload', action='store_true',
                        help='MongoDB 업로드 건너뛰기')
    parser.add_argument('--check-only', action='store_true',
                        help='MongoDB 데이터 확인만 수행')
    parser.add_argument('--clear-db', action='store_true',
                        help='기존 MongoDB 컬렉션 제거 후 새로 저장')
    parser.add_argument('--generate-report', action='store_true',
                        help='파이프라인 보고서 생성')
    parser.add_argument('--show-visualization', action='store_true',
                        help='파이프라인 시각화 표시')
    parser.add_argument('--advanced-features', action='store_true',
                        help='고급 전처리 기능 사용 (PCA, 워드 임베딩, 특성 선택 등)')
    parser.add_argument('--use-bge-m3', action='store_true',
                        help='BGE-M3 임베딩 모델 사용 (고품질 임베딩 생성)')
    parser.add_argument('--bge-model-name', type=str, default='BAAI/bge-m3',
                        help='사용할 BGE 모델 이름 (기본값: BAAI/bge-m3)')
    args = parser.parse_args()
    
    # 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 파이프라인 시각화 객체 생성
    visualizer = create_visualizer("입찰가 전처리 파이프라인", args.output_dir)
    visualizer.start_pipeline()
    
    # MongoDB 데이터 확인만 수행하는 경우
    if args.check_only:
        step_idx = visualizer.add_step("MongoDB 데이터 확인", "MongoDB에 저장된 데이터 확인")
        check_result = check_mongodb_data(visualizer, step_idx)
        visualizer.complete_step(step_idx, {"검사한 컬렉션 수": check_result["collections_checked"]})
        visualizer.end_pipeline()
        
        if args.generate_report:
            report_file = visualizer.generate_report()
            print(f"\n📋 파이프라인 보고서가 생성되었습니다: {report_file}")
        
        return
    
    # 기존 MongoDB 컬렉션 삭제
    if args.clear_db:
        step_idx = visualizer.add_step("MongoDB 컬렉션 초기화", "기존 MongoDB 컬렉션 삭제")
        clear_result = clear_mongodb_collections(visualizer, step_idx)
        visualizer.complete_step(step_idx, {"삭제된 컬렉션 수": clear_result["collections_dropped"]})
    
    # 전처리 파이프라인 실행
    if not args.check_only:
        try:
            # 전처리 파이프라인 단계 추가
            data_load_step = visualizer.add_step("데이터 로드", "원시 데이터 파일 로드")
            
            # 고급 전처리 설정 로깅
            advanced_enabled = args.advanced_features
            logger.info(f"🔍 고급 전처리 기능: {'✅ 활성화됨' if advanced_enabled else '❌ 비활성화됨'}")
            if advanced_enabled:
                logger.info("고급 기능에는 PCA, 워드 임베딩, 특성 선택 등이 포함됩니다")
            
            # BGE-M3 임베딩 설정 로깅
            bge_enabled = args.use_bge_m3
            if bge_enabled:
                logger.info(f"🔍 BGE-M3 임베딩 모델: ✅ 활성화됨 (모델: {args.bge_model_name})")
            
            # 사용자 정의 전처리 함수 정의
            def custom_preprocessing(df, name, config):
                """
                사용자 정의 전처리 함수
                
                Parameters:
                    df (DataFrame): 전처리할 데이터프레임
                    name (str): 데이터셋 이름
                    config (dict): 전처리 설정
    
    Returns:
                    DataFrame: 전처리된 데이터프레임
                """
                # BGE-M3 임베딩 모델 사용이 활성화된 경우
                # 필요한 처리는 이미 FeatureEngineer에서 수행되므로 이 함수는 스킵됨
                return df
            
            # 전처리 파이프라인 초기화
            pipeline = PreprocessingPipeline(data_dir=args.data_dir)
            
            # 파이프라인 설정 업데이트
            pipeline.config["advanced_features"]["enabled"] = advanced_enabled
            if bge_enabled:
                pipeline.config["bge_model"] = {
                    "enabled": True,
                    "model_name": args.bge_model_name
                }
            
            # 파이프라인 실행 (각 단계별로 진행)
            processed_data = {}
            
            # 1. 데이터 로드
            try:
                # 데이터 로드
                dataset_dict = pipeline.data_loader.load_raw_data(args.file_pattern)
                
                if not dataset_dict:
                    visualizer.complete_step(data_load_step, {"로드된 데이터셋": 0})
                    raise ValueError("데이터 로드 실패: 데이터셋이 비어 있습니다.")
                
                # 데이터 로드 단계 완료
                visualizer.complete_step(data_load_step, {
                    "로드된 데이터셋": len(dataset_dict),
                    "총 레코드 수": sum(len(df) for df in dataset_dict.values())
                })
                
                # 2. 데이터 정제
                cleaning_step = visualizer.add_step("데이터 정제", "결측치 처리, 중복 제거, 데이터 타입 변환")
                cleaned_dict = {}
                
                pbar = visualizer.create_progress_bar(cleaning_step, len(dataset_dict), "🧹 데이터 정제 진행")
                for name, df in dataset_dict.items():
                    # 파일 유형 기반 설정 찾기
                    preprocessing_config = pipeline._get_preprocessing_config(name)
                    if preprocessing_config:
                        logger.info(f"🔍 '{name}' 데이터셋에 맞춤형 전처리 적용")
                        # 필수 컬럼 확인
                        if preprocessing_config.get("required_columns"):
                            pipeline._validate_required_columns(df, preprocessing_config["required_columns"], name)
                    
                    # 기본 정제 적용
                    cleaned_dict[name] = pipeline.cleaner.clean_dataset(df, name)
                    pbar.update(1)
                
                pbar.close()
                
                # 데이터셋 정제 단계 완료
                cleaning_metrics = {
                    "처리된 데이터셋": len(cleaned_dict),
                    "제거된 결측치": sum(pipeline.cleaner.cleaning_stats[name]['missing_values_removed'] for name in cleaned_dict),
                    "제거된 중복": sum(pipeline.cleaner.cleaning_stats[name]['duplicates_removed'] for name in cleaned_dict)
                }
                visualizer.complete_step(cleaning_step, cleaning_metrics)
                
                # 3. 데이터 변환
                transform_step = visualizer.add_step("데이터 변환", "로그 변환, 정규화, 인코딩")
                transformed_dict = {}
                
                pbar = visualizer.create_progress_bar(transform_step, len(cleaned_dict), "🔄 데이터 변환 진행")
                for name, df in cleaned_dict.items():
                    # 파일 유형 기반 설정 찾기
                    preprocessing_config = pipeline._get_preprocessing_config(name)
                    
                    # 맞춤형 로그 변환 및 정규화 적용
                    if preprocessing_config:
                        # 로그 변환 적용
                        if preprocessing_config.get("log_transform_columns"):
                            df = pipeline._apply_log_transform(df, preprocessing_config["log_transform_columns"], name)
                        
                        # 정규화 적용
                        if preprocessing_config.get("normalize_columns"):
                            # 기본 변환 프로세스 진행
                            df = pipeline.transformer.transform_dataset(df, name)
                            
                            # 날짜 컬럼 처리
                            if preprocessing_config.get("date_columns"):
                                df = pipeline._process_date_columns(df, preprocessing_config["date_columns"])
                        else:
                            # 기본 변환 프로세스 진행
                            df = pipeline.transformer.transform_dataset(df, name)
                    else:
                        # 기본 변환 프로세스 진행
                        df = pipeline.transformer.transform_dataset(df, name)
                    
                    transformed_dict[name] = df
                    pbar.update(1)
                
                pbar.close()
                
                # 데이터 변환 단계 완료
                transform_metrics = {
                    "처리된 데이터셋": len(transformed_dict),
                    "생성된 특성 수": sum(len(df.columns) - len(cleaned_dict[name].columns) for name, df in transformed_dict.items())
                }
                visualizer.complete_step(transform_step, transform_metrics)
                
                # 4. 특성 엔지니어링
                feature_step = visualizer.add_step("특성 엔지니어링", "텍스트 처리, 차원 축소, 특성 조합")
                engineered_dict = {}
                
                # 고급 특성 엔지니어링 활성화 시 단계 제목 업데이트
                if advanced_enabled:
                    feature_step_title = "고급 특성 엔지니어링"
                    feature_step_desc = "텍스트 임베딩, PCA, 특성 선택, 특성 조합"
                    visualizer.update_step(feature_step, feature_step_title, feature_step_desc)
                
                pbar = visualizer.create_progress_bar(feature_step, len(transformed_dict), "🔧 특성 엔지니어링 진행")
                for name, df in transformed_dict.items():
                    # 파일 유형 기반 설정 찾기
                    preprocessing_config = pipeline._get_preprocessing_config(name)
                    
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
                    df = pipeline.feature_engineer.engineer_features(df, name, advanced_enabled)
                    
                    engineered_dict[name] = df
                    pbar.update(1)
                
                pbar.close()
                
                # 특성 엔지니어링 완료 시 특성 타입별 개수 계산
                engineered_cols_stats = {}
                for name, df in engineered_dict.items():
                    # 컬럼 타입별 개수
                    col_types = {}
                    # 텍스트 임베딩 특성
                    text_embedding_cols = len([c for c in df.columns if 'W2V_' in c])
                    # PCA 특성
                    pca_cols = len([c for c in df.columns if 'PCA_' in c])
                    # TF-IDF 특성
                    tfidf_cols = len([c for c in df.columns if 'TFIDF_' in c])
                    # 선택된 특성
                    selected_cols = len([c for c in df.columns if 'selected_' in c])
                    
                    if text_embedding_cols > 0:
                        col_types['임베딩 특성'] = text_embedding_cols
                    if pca_cols > 0:
                        col_types['PCA 특성'] = pca_cols
                    if tfidf_cols > 0:
                        col_types['TF-IDF 특성'] = tfidf_cols
                    if selected_cols > 0:
                        col_types['선택된 특성'] = selected_cols
                    
                    engineered_cols_stats[name] = col_types
                
                # 특성 엔지니어링 단계 완료
                feature_metrics = {
                    "처리된 데이터셋": len(engineered_dict),
                    "최종 특성 수": sum(len(df.columns) for df in engineered_dict.values()),
                    "고급 특성 적용": "예" if advanced_enabled else "아니오"
                }
                
                # 고급 특성 통계 추가
                if advanced_enabled:
                    all_text_embedding = sum(stats.get('임베딩 특성', 0) for stats in engineered_cols_stats.values())
                    all_pca = sum(stats.get('PCA 특성', 0) for stats in engineered_cols_stats.values())
                    all_tfidf = sum(stats.get('TF-IDF 특성', 0) for stats in engineered_cols_stats.values())
                    all_selected = sum(stats.get('선택된 특성', 0) for stats in engineered_cols_stats.values())
                    
                    if all_text_embedding > 0:
                        feature_metrics['임베딩 특성'] = all_text_embedding
                    if all_pca > 0:
                        feature_metrics['PCA 특성'] = all_pca
                    if all_tfidf > 0:
                        feature_metrics['TF-IDF 특성'] = all_tfidf
                    if all_selected > 0:
                        feature_metrics['선택된 특성'] = all_selected
                
                visualizer.complete_step(feature_step, feature_metrics)
                
                # 5. 입찰가 분석 특화 처리
                price_step = visualizer.add_step("입찰가 특화 처리", "입찰가 관련 특성 생성 및 분석")
                
                pbar = visualizer.create_progress_bar(price_step, len(engineered_dict), "💰 입찰가 특화 처리 진행")
                for name, df in engineered_dict.items():
                    preprocessing_config = pipeline._get_preprocessing_config(name)
                    if preprocessing_config and "기초금액" in df.columns and "예정금액" in df.columns:
                        # 가격비율 계산
                        if "투찰가" in df.columns:
                            df["낙찰가격비율"] = df["투찰가"] / df["예정금액"]
                            logger.info(f"✅ '{name}' 데이터셋의 낙찰가격비율 계산 완료")
                        
                        # 기타 입찰가 관련 특성 생성
                        pipeline._create_bid_price_features(df, name)
                    
                    engineered_dict[name] = df
                    pbar.update(1)
                
                pbar.close()
                
                # 입찰가 분석 특화 처리 단계 완료
                price_metrics = {
                    "처리된 데이터셋": len(engineered_dict),
                    "입찰가 특성 생성": sum(1 for df in engineered_dict.values() if "투찰가_예정금액비율" in df.columns)
                }
                visualizer.complete_step(price_step, price_metrics)
                
                # 처리된 최종 데이터
                processed_data = engineered_dict
                
                # 6. MongoDB에 저장 (선택 사항)
                if not args.skip_upload:
                    upload_step = visualizer.add_step("MongoDB 업로드", "전처리된 데이터를 MongoDB에 저장")
                    
                    try:
                        with pipeline.mongodb_handler as mongo:
                            pbar = visualizer.create_progress_bar(upload_step, 1, "💾 MongoDB 저장 진행")
                            collection_names = mongo.save_datasets(processed_data)
                            pbar.update(1)
                            pbar.close()
                            
                            upload_metrics = {
                                "저장된 데이터셋": len(collection_names),
                                "총 문서 수": sum(len(df) for df in processed_data.values())
                            }
                            visualizer.complete_step(upload_step, upload_metrics)
                    except Exception as e:
                        logger.error(f"❌ MongoDB 저장 중 오류 발생: {e}")
                        visualizer.complete_step(upload_step, {"오류": str(e)})
                        raise
            
            except Exception as e:
                logger.error(f"파이프라인 처리 중 오류 발생: {e}", exc_info=True)
                raise
            
            # 결과 요약
            if processed_data:
                summary_step = visualizer.add_step("결과 요약", "파이프라인 처리 결과 요약")
                
                summary_metrics = {
                    "처리된 데이터셋": len(processed_data),
                    "총 레코드 수": sum(len(df) for df in processed_data.values()),
                    "총 특성 수": sum(len(df.columns) for df in processed_data.values()),
                    "고급 특성 적용": "예" if advanced_enabled else "아니오"
                }
                
                print("\n✅ 처리 결과 요약:")
                for name, df in processed_data.items():
                    print(f"  - {name}: {df.shape[0]} 행 x {df.shape[1]} 열")
                    # 특성 목록 (처음 5개)
                    if len(df.columns) > 0:
                        print(f"    주요 특성: {', '.join(df.columns[:5])}... 외 {max(0, len(df.columns)-5)}개")
                    
                    # 고급 특성 정보 출력
                    if advanced_enabled and name in engineered_cols_stats:
                        stats = engineered_cols_stats[name]
                        if stats:
                            print(f"    고급 특성:")
                            for feature_type, count in stats.items():
                                print(f"      - {feature_type}: {count}개")
                
                visualizer.complete_step(summary_step, summary_metrics)
                
                # MongoDB 관련 메시지
                if args.skip_upload:
                    print("\n⚠️ MongoDB 업로드를 건너뛰었습니다.")
                else:
                    print("\n✅ MongoDB에 데이터가 성공적으로 업로드되었습니다.")
                    print("   확인을 위해 다음 명령어를 사용할 수 있습니다:")
                    print("   python src/preprocess_upload_mongo.py --check-only")
                
                # 고급 기능 관련 메시지
                if advanced_enabled:
                    print("\n🔍 고급 전처리 기능이 활성화되었습니다:")
                    if sum(stats.get('임베딩 특성', 0) for stats in engineered_cols_stats.values()) > 0:
                        print("   ✅ 워드 임베딩 (Word2Vec) 기능이 적용되었습니다.")
                    if sum(stats.get('PCA 특성', 0) for stats in engineered_cols_stats.values()) > 0:
                        print("   ✅ PCA 차원 축소가 적용되었습니다.")
                    if sum(stats.get('TF-IDF 특성', 0) for stats in engineered_cols_stats.values()) > 0:
                        print("   ✅ TF-IDF 텍스트 특성 추출이 적용되었습니다.")
                    if sum(stats.get('선택된 특성', 0) for stats in engineered_cols_stats.values()) > 0:
                        print("   ✅ 특성 선택 알고리즘이 적용되었습니다.")
            else:
                print("\n❌ 처리할 데이터가 없습니다. 데이터 디렉토리와 파일 패턴을 확인하세요.")
            
    except Exception as e:
            logger.error(f"파이프라인 실행 중 오류 발생: {e}", exc_info=True)
            print(f"\n❌ 오류 발생: {e}")
    
    # 파이프라인 종료
    visualizer.end_pipeline()
    
    # 보고서 생성
    if args.generate_report:
        report_file = visualizer.generate_report({
            "데이터 디렉토리": args.data_dir or os.getenv('DATA_DIR', './data'),
            "파일 패턴": args.file_pattern,
            "MongoDB 업로드": "건너뜀" if args.skip_upload else "완료"
        })
        print(f"\n📋 파이프라인 보고서가 생성되었습니다: {report_file}")
    
    # 시각화 표시
    if args.show_visualization:
        visualizer.visualize_pipeline(show_plot=True, save_plot=True)
    
    logger.info("=== 입찰가 데이터 전처리 및 MongoDB 업로드 완료 ===")

def check_mongodb_data(visualizer=None, step_idx=None):
    """
    MongoDB에 저장된 데이터 확인
    
    Parameters:
        visualizer (PipelineVisualizer, optional): 시각화 객체
        step_idx (int, optional): 단계 인덱스
        
    Returns:
        dict: 검사 결과
    """
    print("\n🔍 MongoDB 데이터 확인 중...")
    
    result = {
        "collections_checked": 0,
        "total_documents": 0,
        "collections_info": {}
    }
    
    try:
        with MongoDBHandler() as mongo:
            # 기본 컬렉션 이름 가져오기
            collection_names = mongo.get_default_collection_names()
            
            # 진행 표시줄 생성
            if visualizer and step_idx is not None:
                pbar = visualizer.create_progress_bar(step_idx, len(collection_names), "MongoDB 컬렉션 검사")
            else:
                pbar = None
            
            # 컬렉션별 데이터 확인
            for key, collection_name in collection_names.items():
                try:
                    # 해당 컬렉션 접근
                    collection = mongo.db[collection_name]
                    
                    # 문서 수 확인
                    count = collection.count_documents({})
                    result["total_documents"] += count
                    result["collections_checked"] += 1
                    
                    if count > 0:
                        # 샘플 데이터 조회
                        sample = list(collection.find({}).limit(1))[0]
                        sample_keys = list(sample.keys())
                        
                        # 컬렉션 정보 저장
                        result["collections_info"][collection_name] = {
                            "document_count": count,
                            "field_count": len(sample_keys),
                            "sample_fields": sample_keys[:5]
                        }
                        
                        # 공고번호 분포 확인 (있는 경우)
                        if '공고번호' in sample_keys:
                            # 공고번호 개수
                            unique_notices = len(collection.distinct('공고번호'))
                            result["collections_info"][collection_name]["unique_notices"] = unique_notices
                        
                        # 결과 출력
                        print(f"\n✅ 컬렉션: {collection_name}")
                        print(f"  - 문서 수: {count}개")
                        print(f"  - 필드 수: {len(sample_keys)}개")
                        print(f"  - 주요 필드: {', '.join(sample_keys[:5])}... 외 {max(0, len(sample_keys)-5)}개")
                        
                        # 공고번호 분포 출력
                        if '공고번호' in sample_keys:
                            print(f"  - 공고번호 수: {unique_notices}개")
                    else:
                        print(f"\n⚠️ 컬렉션 {collection_name}에 데이터가 없습니다.")
                        result["collections_info"][collection_name] = {
                            "document_count": 0,
                            "status": "empty"
                        }
                
                except Exception as e:
                    print(f"\n❌ 컬렉션 {collection_name} 확인 중 오류 발생: {e}")
                    result["collections_info"][collection_name] = {
                        "status": "error",
                        "error": str(e)
                    }
                
                # 진행 표시줄 업데이트
                if pbar:
                    pbar.update(1)
            
            # 진행 표시줄 닫기
            if pbar:
                pbar.close()
        
        print(f"\n✅ MongoDB 데이터 확인 완료: {result['collections_checked']} 컬렉션, 총 {result['total_documents']} 문서")
    
    except Exception as e:
        logger.error(f"MongoDB 연결 중 오류 발생: {e}", exc_info=True)
        print(f"\n❌ MongoDB 연결 오류: {e}")
        result["status"] = "error"
        result["error"] = str(e)
    
    return result

def clear_mongodb_collections(visualizer=None, step_idx=None):
    """
    MongoDB 컬렉션 초기화
    
    Parameters:
        visualizer (PipelineVisualizer, optional): 시각화 객체
        step_idx (int, optional): 단계 인덱스
        
    Returns:
        dict: 삭제 결과
    """
    print("\n🗑️ MongoDB 컬렉션 초기화 중...")
    
    result = {
        "collections_dropped": 0,
        "status": "success",
        "details": {}
    }
    
    try:
        with MongoDBHandler() as mongo:
            # 기본 컬렉션 이름 가져오기
            collection_names = mongo.get_default_collection_names()
            
            # 진행 표시줄 생성
            if visualizer and step_idx is not None:
                pbar = visualizer.create_progress_bar(step_idx, len(collection_names), "MongoDB 컬렉션 삭제")
            else:
                pbar = None
            
            # 각 컬렉션 삭제
            for key, collection_name in collection_names.items():
                try:
                    mongo.db.drop_collection(collection_name)
                    print(f"  ✅ 컬렉션 {collection_name} 삭제 완료")
                    result["collections_dropped"] += 1
                    result["details"][collection_name] = "dropped"
                except Exception as e:
                    print(f"  ⚠️ 컬렉션 {collection_name} 삭제 중 오류: {e}")
                    result["details"][collection_name] = str(e)
                
                # 진행 표시줄 업데이트
                if pbar:
                    pbar.update(1)
            
            # 진행 표시줄 닫기
            if pbar:
                pbar.close()
        
        print(f"\n✅ MongoDB 컬렉션 초기화 완료: {result['collections_dropped']} 컬렉션 삭제됨")
    
    except Exception as e:
        logger.error(f"MongoDB 연결 중 오류 발생: {e}", exc_info=True)
        print(f"\n❌ MongoDB 연결 오류: {e}")
        result["status"] = "error"
        result["error"] = str(e)
    
    return result

if __name__ == "__main__":
    main() 