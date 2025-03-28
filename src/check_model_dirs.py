"""
모델 디렉토리 구조를 확인하는 스크립트입니다.
"""
import os
import sys
from pathlib import Path

def print_directory_structure(path, indent=0, max_depth=4, current_depth=0):
    """지정된 경로의 디렉토리 구조를 출력합니다."""
    if current_depth > max_depth:
        return
    
    try:
        if os.path.isdir(path):
            print(' ' * indent + os.path.basename(path) + '/')
            for item in sorted(os.listdir(path)):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    print_directory_structure(item_path, indent + 4, max_depth, current_depth + 1)
                else:
                    # 파일인 경우는 깊이가 최대 깊이보다 작을 때만 출력
                    if current_depth < max_depth:
                        print(' ' * (indent + 4) + item)
    except Exception as e:
        print(f"Error accessing {path}: {e}")

def main():
    """메인 함수"""
    # 프로젝트 루트 디렉토리
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 모델 디렉토리 경로
    models_dir = os.path.join(project_root, 'models')
    
    if not os.path.exists(models_dir):
        print(f"모델 디렉토리가 존재하지 않습니다: {models_dir}")
        return
    
    print(f"모델 디렉토리 구조:")
    print_directory_structure(models_dir)
    
    # 세부 정보: 각 데이터셋별 모델 수 확인
    print("\n\n각 데이터셋별 모델 수:")
    autogluon_dir = os.path.join(models_dir, 'autogluon')
    
    if os.path.exists(autogluon_dir):
        datasets = [d for d in os.listdir(autogluon_dir) if os.path.isdir(os.path.join(autogluon_dir, d))]
        
        for dataset in sorted(datasets):
            dataset_dir = os.path.join(autogluon_dir, dataset)
            models = [m for m in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, m))]
            
            print(f"  {dataset}: {len(models)} 모델")
            for model in sorted(models):
                print(f"    - {model}")

if __name__ == "__main__":
    main() 