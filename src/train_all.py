"""
전체 데이터셋과 타겟 접두사에 대한 일괄 학습 모듈입니다.
"""
import os
import sys
import logging
import argparse
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train_model_for_dataset, init_directories

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def train_all_combinations(datasets=None, target_prefixes=None, model_types=None, time_limit=60):
    """
    지정된 모든 데이터셋과 타겟 접두사 조합에 대해 모델을 학습합니다.
    
    Args:
        datasets (list): 학습할 데이터셋 목록 (기본값: ["dataset3", "dataset2", "datasetetc"])
        target_prefixes (list): 학습할 타겟 접두사 목록 (기본값: ["010", "020", "050", "100"])
        model_types (list): 학습할 모델 타입 목록 (기본값: ["lgbm", "autogluon"])
        time_limit (int): 모델당 학습 시간 제한 (초) (기본값: 60)
        
    Returns:
        dict: 학습 결과 요약
    """
    # 기본값 설정
    if datasets is None:
        datasets = ["dataset3", "dataset2", "datasetetc"]
    
    if target_prefixes is None:
        target_prefixes = ["010", "020", "050", "100"]
        
    if model_types is None:
        model_types = ["lgbm", "autogluon"]
        
    logger.info(f"전체 모델 학습 시작: {len(datasets)} 데이터셋 x {len(target_prefixes)} 타겟 접두사 x {len(model_types)} 모델 타입")
    
    # 결과 저장을 위한 딕셔너리
    all_results = {}
    
    # 디렉토리 초기화
    init_directories()
    
    # 시작 시간 기록
    start_time = datetime.now()
    logger.info(f"학습 시작 시간: {start_time}")
    
    # 모든 조합에 대해 학습 진행
    for model_type in model_types:
        logger.info(f"===== 모델 타입: {model_type} =====")
        all_results[model_type] = {}
        
        for dataset in datasets:
            all_results[model_type][dataset] = {}
            
            for prefix in target_prefixes:
                try:
                    logger.info(f"[학습] {model_type} - {dataset} - {prefix} 시작")
                    results = train_model_for_dataset(
                        dataset_key=dataset,
                        target_prefix=prefix,
                        model_type=model_type,
                        time_limit_per_model=time_limit
                    )
                    
                    # 결과 저장
                    all_results[model_type][dataset][prefix] = {
                        'success': True,
                        'models_trained': len(results),
                        'details': results
                    }
                    
                    logger.info(f"[완료] {model_type} - {dataset} - {prefix}: {len(results)}개 모델 학습")
                    
                except Exception as e:
                    logger.error(f"[실패] {model_type} - {dataset} - {prefix}: {str(e)}")
                    all_results[model_type][dataset][prefix] = {
                        'success': False,
                        'error': str(e)
                    }
    
    # 종료 시간 및 총 소요 시간 기록
    end_time = datetime.now()
    total_time = end_time - start_time
    logger.info(f"학습 종료 시간: {end_time}")
    logger.info(f"총 소요 시간: {total_time}")
    
    # 결과 요약
    successes = 0
    failures = 0
    total_models = 0
    
    logger.info("===== 학습 결과 요약 =====")
    
    for model_type, datasets_results in all_results.items():
        model_successes = 0
        model_failures = 0
        model_total_models = 0
        
        for dataset, prefixes in datasets_results.items():
            for prefix, result in prefixes.items():
                if result['success']:
                    successes += 1
                    model_successes += 1
                    models_trained = result.get('models_trained', 0)
                    total_models += models_trained
                    model_total_models += models_trained
                else:
                    failures += 1
                    model_failures += 1
        
        logger.info(f"[{model_type}] 성공: {model_successes}, 실패: {model_failures}, 총 모델 수: {model_total_models}")
    
    logger.info("=====")
    logger.info(f"총 학습 시도: {successes + failures}")
    logger.info(f"성공: {successes}")
    logger.info(f"실패: {failures}")
    logger.info(f"학습된 총 모델 수: {total_models}")
    logger.info("========================")
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="전체 데이터셋과 타겟 접두사에 대한 일괄 모델 학습")
    parser.add_argument("--models", type=str, nargs='+', 
                        default=["lgbm", "autogluon"],
                        help="학습할 모델 타입 목록 (기본값: lgbm autogluon)")
    parser.add_argument("--time-limit", type=int, default=60,
                        help="모델당 학습 시간 제한 (초) (기본값: 60)")
    parser.add_argument("--datasets", type=str, nargs='+',
                        default=["dataset3", "dataset2", "datasetetc"],
                        help="학습할 데이터셋 목록 (기본값: dataset3 dataset2 datasetetc)")
    parser.add_argument("--prefixes", type=str, nargs='+',
                        default=["010", "020", "050", "100"],
                        help="학습할 타겟 접두사 목록 (기본값: 010 020 050 100)")
    
    args = parser.parse_args()
    
    try:
        train_all_combinations(
            datasets=args.datasets,
            target_prefixes=args.prefixes,
            model_types=args.models,
            time_limit=args.time_limit
        )
        
        logger.info("모든 모델 학습 완료")
        
    except Exception as e:
        logger.error(f"전체 학습 과정 중 오류 발생: {str(e)}")
        raise 