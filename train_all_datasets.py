#!/usr/bin/env python
"""
세 가지 데이터셋(dataset2, dataset3, datasetetc)에 대해 모델을 학습하는 스크립트
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

# 현재 디렉토리를 프로젝트 루트로 설정
project_root = Path(__file__).parent
os.chdir(project_root)

def run_command(cmd):
    """명령어 실행 및 출력 표시"""
    print(f"\n실행 명령어: {cmd}\n" + "="*50)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    # 실시간 출력 표시
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    
    process.stdout.close()
    return_code = process.wait()
    
    if return_code:
        print(f"\n명령어 실행 실패 (코드: {return_code})\n" + "="*50)
    else:
        print("\n명령어 실행 성공\n" + "="*50)
    
    return return_code

def main():
    # 인자 파싱
    parser = argparse.ArgumentParser(description='여러 데이터셋에 대해 모델 학습')
    parser.add_argument('--datasets', type=str, default="2,3,etc", help='학습할 데이터셋 목록 (콤마로 구분, 기본값: 2,3,etc)')
    parser.add_argument('--prefixes', type=str, default="050,100", help='학습할 타겟 접두사 (콤마로 구분, 기본값: 050,100)')
    parser.add_argument('--num-targets', type=int, default=5, help='각 접두사당 학습할 타겟 수 (기본값: 5)')
    parser.add_argument('--train-only', action='store_true', help='학습만 수행 (데이터 전처리 건너뛰기)')
    parser.add_argument('--gpu', type=str, default='True', help='GPU 사용 여부 (True/False)')
    args = parser.parse_args()
    
    # 데이터셋과 타겟 접두사 분리
    datasets = args.datasets.split(',')
    prefixes = args.prefixes.split(',')
    
    print(f"학습 설정:")
    print(f"- 데이터셋: {datasets}")
    print(f"- 타겟 접두사: {prefixes}")
    print(f"- 각 접두사당 타겟 수: {args.num_targets}")
    print(f"- 학습만 수행: {args.train_only}")
    print(f"- GPU 사용: {args.gpu}")
    
    # 각 데이터셋에 대해 학습 실행
    for dataset in datasets:
        print(f"\n데이터셋 {dataset} 처리 시작...\n")
        
        # 명령어 구성
        cmd = f"python main.py --dataset-key {dataset} --target-prefixes {args.prefixes}"
        cmd += f" --num-targets {args.num_targets} --gpu {args.gpu}"
        
        if args.train_only:
            cmd += " --train-only"
        
        # 명령어 실행
        if run_command(cmd) != 0:
            print(f"데이터셋 {dataset} 처리 중 오류 발생")
        else:
            print(f"데이터셋 {dataset} 처리 완료")
    
    print("\n모든 데이터셋 처리 완료")

if __name__ == "__main__":
    main() 