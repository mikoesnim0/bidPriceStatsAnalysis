#!/usr/bin/env python
"""
입찰가 예측 샘플 스크립트
공고번호를 입력받아 예측을 수행하는 예제입니다.
"""
import os
import sys
import json
import argparse
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 필요한 모듈 임포트
from src.predict import load_and_predict

def main():
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(description='입찰가 예측 샘플 스크립트')
    parser.add_argument('notice_id', type=str, help='예측할 공고번호')
    parser.add_argument('--dataset', type=str, default='2', help='사용할 데이터셋 (기본값: 2)')
    parser.add_argument('--prefix', type=str, default='050', help='타겟 접두사 (기본값: 050)')
    args = parser.parse_args()
    
    print(f"입찰가 예측 시작: 공고번호 {args.notice_id}, 데이터셋 {args.dataset}, 타겟 접두사 {args.prefix}")
    print("="*50)
    
    try:
        # 예측 수행
        result = load_and_predict(
            notice_id=args.notice_id,
            dataset_key=args.dataset,
            target_prefix=args.prefix
        )
        
        # 결과 출력
        if "error" in result and result.get("success", True) is False:
            print(f"예측 실패: {result['error']}")
        else:
            # 예측 결과 출력
            print(f"예측 성공!")
            print(f"공고번호: {result['notice_id']}")
            print(f"사용 모델: {result['dataset_key']}/{result['target_prefix']}")
            print(f"컬렉션: {result['collection_name']}")
            
            # 예측 값 출력
            print("\n예측 결과:")
            for key, value in result['predictions'].items():
                print(f"  {key}: {value}")
            
            # 메타데이터 일부 출력
            print("\n공고 메타데이터 요약:")
            meta_summary = {k: v for k, v in list(result['metadata'].items())[:5]}
            print(json.dumps(meta_summary, ensure_ascii=False, indent=2))
            
            # 전체 결과 저장
            output_file = f"prediction_{args.notice_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n전체 예측 결과가 {output_file}에 저장되었습니다.")
    
    except Exception as e:
        print(f"예측 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 