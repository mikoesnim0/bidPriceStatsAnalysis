import random
import numpy as np
import bisect
import time
import pickle
import os
from pathlib import Path


def simulate(n, price_range=3, seed=42):
    """
    입찰률 분포를 시뮬레이션하는 함수
    
    Parameters:
    -----------
    n : int
        시뮬레이션 횟수
    price_range : int
        범위 선택 (2 또는 3)
    seed : int
        난수 발생 시드값
        
    Returns:
    --------
    list
        시뮬레이션 결과값 리스트
    """
    random.seed(seed)
    
    ranges = []
    
    if price_range == 2:
        # -0.02 ~ 0.02 까지 15등분
        start, end = -0.02, 0.02
        step = (end - start) / 15.0
        for i in range(15):
            low = start + i*step
            high = start + (i+1)*step
            ranges.append((low, high))
    else:  # price_range == 3
        # 1~8 번: -0.03 ~ 0.0 까지 8등분
        # 9~15번: 0.0 ~ 0.03 까지 7등분
        
        # 1~8
        neg_start, neg_end = -0.03, 0.0
        neg_step = (neg_end - neg_start) / 8.0
        for i in range(1, 9):
            low = neg_start + (i-1)*neg_step
            high = neg_start + i*neg_step
            ranges.append((low, high))
        
        # 9~15
        pos_start, pos_end = 0.0, 0.03
        pos_step = (pos_end - pos_start) / 7.0
        for i in range(9, 16):
            j = i - 9
            low = pos_start + j*pos_step
            high = pos_start + (j+1)*pos_step
            ranges.append((low, high))
    
    results = []
    for _ in range(n):
        # 15개 중 4개 랜덤 추출
        chosen_indices = random.sample(range(15), 4)
        
        # 각 선택된 인덱스에 해당하는 범위 내의 랜덤 값
        values = []
        for idx in chosen_indices:
            low, high = ranges[idx]
            val = random.uniform(low, high)
            values.append(val)
        
        # 평균 계산
        avg_val = sum(values) * 25.0
        results.append(avg_val)
    
    return results


def create_probability_lookup(data):
    """
    정렬된 데이터를 기반으로 확률 조회가 가능한 자료구조 생성
    
    Parameters:
    -----------
    data : list
        시뮬레이션 결과값 리스트
        
    Returns:
    --------
    tuple
        (정렬된 데이터, 전체 데이터 수)
    """
    # 데이터 정렬
    sorted_data = sorted(data)
    total_count = len(sorted_data)
    
    return sorted_data, total_count


def get_probability_between(sorted_data, total_count, lower_bound, upper_bound):
    """
    두 값 사이의 확률을 계산하는 함수 (이진 탐색 이용)
    
    Parameters:
    -----------
    sorted_data : list
        정렬된 데이터 리스트
    total_count : int
        전체 데이터 수
    lower_bound : float
        하한값
    upper_bound : float
        상한값
        
    Returns:
    --------
    float
        두 값 사이의 확률
    """
    # 이진 탐색을 통해 lower_bound, upper_bound의 인덱스 찾기
    lower_idx = bisect.bisect_left(sorted_data, lower_bound)
    upper_idx = bisect.bisect_right(sorted_data, upper_bound)
    
    # 두 값 사이에 존재하는 데이터 개수
    count_between = upper_idx - lower_idx
    
    # 확률 계산
    probability = count_between / total_count
    
    return probability


def save_data(data, filename):
    """데이터를 파일로 저장"""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"데이터가 {filename}에 저장되었습니다.")


def load_data(filename):
    """저장된 데이터 로드"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def main():
    """
    메인 실행 함수 - 시뮬레이션 실행 및 결과 분석
    """
    # 파일 경로 설정
    base_dir = Path("bidPriceStatsAnalysis/simulate/data")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # 예가범위 선택 (2 또는 3)
    price_range = 3
    
    # 데이터 파일 경로
    data_file = base_dir / f"simulation_data_range_{price_range}_2pow24.pkl"
    lookup_file = base_dir / f"sorted_lookup_range_{price_range}_2pow24.pkl"
    
    # 데이터가 이미 존재하는지 확인
    if os.path.exists(data_file) and os.path.exists(lookup_file):
        print(f"기존 데이터를 로드합니다: {data_file}")
        sorted_data, total_count = load_data(lookup_file)
    else:
        print("시뮬레이션을 시작합니다...")
        start_time = time.time()
        
        # 2^24 = 16,777,216개 시뮬레이션 (큰 데이터셋)
        n = 2**24
        data = simulate(n, price_range=price_range)
        
        # 데이터 기본 통계
        print(f"전체 데이터 수: {len(data)}")
        print(f"평균: {sum(data)/len(data)}")
        print(f"최소값: {min(data)}")
        print(f"최대값: {max(data)}")
        
        # 조회용 데이터 구조 생성
        print("조회용 데이터 구조를 생성합니다...")
        sorted_data, total_count = create_probability_lookup(data)
        
        # 데이터 저장
        save_data(data, data_file)
        save_data((sorted_data, total_count), lookup_file)
        
        end_time = time.time()
        print(f"시뮬레이션 및 데이터 처리 완료: {end_time - start_time:.2f}초")
    
    # 조회 기능 테스트
    print("\n구간별 확률 조회 테스트:")
    test_ranges = [
        (-3.0, -2.0),
        (-2.0, -1.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (1.0, 2.0),
        (2.0, 3.0),
        (-0.5, 0.5),
        (-1.0, 1.0)
    ]
    
    for lower, upper in test_ranges:
        start_time = time.time()
        prob = get_probability_between(sorted_data, total_count, lower, upper)
        query_time = (time.time() - start_time) * 1000  # 밀리초 단위
        print(f"구간 [{lower}, {upper}] 확률: {prob:.6f} (조회 시간: {query_time:.3f}ms)")
    
    # 사용자 입력 구간에 대한 확률 조회
    while True:
        try:
            print("\n구간 확률 조회 (종료하려면 'q' 입력)")
            user_input = input("두 값 사이의 확률을 조회하려면 두 숫자를 입력하세요 (예: -1 1): ")
            
            if user_input.lower() == 'q':
                break
                
            lower, upper = map(float, user_input.split())
            if lower > upper:
                lower, upper = upper, lower
                
            start_time = time.time()
            prob = get_probability_between(sorted_data, total_count, lower, upper)
            query_time = (time.time() - start_time) * 1000  # 밀리초 단위
            
            print(f"구간 [{lower}, {upper}] 확률: {prob:.6f} (조회 시간: {query_time:.3f}ms)")
            
        except ValueError:
            print("올바른 형식으로 입력해주세요.")
        except Exception as e:
            print(f"오류 발생: {e}")


if __name__ == "__main__":
    main()
