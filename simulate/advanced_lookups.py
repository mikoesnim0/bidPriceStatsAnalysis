import numpy as np
import bisect
import time
import pickle
import os
from pathlib import Path


class BidPriceDistribution:
    """
    입찰가격 분포 데이터에 대한 효율적인 조회를 제공하는 클래스
    
    이진 탐색을 활용하여 O(log n) 시간 복잡도로 구간별 확률을 계산
    """
    
    def __init__(self, data=None):
        """
        입찰가격 분포 객체 초기화
        
        Parameters:
        -----------
        data : list or numpy.ndarray, optional
            원본 시뮬레이션 데이터
        """
        if data is not None:
            self.load_data(data)
        else:
            self.sorted_data = None
            self.total_count = 0
            self.percentile_cache = {}
    
    def load_data(self, data):
        """
        데이터 로드 및 전처리
        
        Parameters:
        -----------
        data : list or numpy.ndarray
            원본 시뮬레이션 데이터
        """
        if not isinstance(data, np.ndarray):
            self.sorted_data = np.array(sorted(data))
        else:
            self.sorted_data = np.sort(data)
        
        self.total_count = len(self.sorted_data)
        self.percentile_cache = {}  # 자주 사용되는 백분위수 캐싱
    
    def save(self, filename):
        """
        분포 객체를 파일로 저장
        
        Parameters:
        -----------
        filename : str
            저장할 파일 경로
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"분포 객체가 {filename}에 저장되었습니다.")
    
    @classmethod
    def load(cls, filename):
        """
        파일에서 분포 객체 로드
        
        Parameters:
        -----------
        filename : str
            로드할 파일 경로
            
        Returns:
        --------
        BidPriceDistribution
            로드된 분포 객체
        """
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        print(f"분포 객체를 {filename}에서 로드했습니다.")
        return obj
    
    def get_probability_between(self, lower_bound, upper_bound):
        """
        주어진 범위 내에 값이 존재할 확률 계산
        
        Parameters:
        -----------
        lower_bound : float
            하한값
        upper_bound : float
            상한값
            
        Returns:
        --------
        float
            계산된 확률값
        """
        if self.sorted_data is None:
            raise ValueError("데이터가 로드되지 않았습니다.")
        
        # 이진 탐색으로 lower_bound, upper_bound 위치 찾기
        lower_idx = bisect.bisect_left(self.sorted_data, lower_bound)
        upper_idx = bisect.bisect_right(self.sorted_data, upper_bound)
        
        # 확률 계산
        count_between = upper_idx - lower_idx
        probability = count_between / self.total_count
        
        return probability
    
    def get_value_at_percentile(self, percentile):
        """
        지정된 백분위수에 해당하는 값 반환
        
        Parameters:
        -----------
        percentile : float
            백분위수 (0-100)
            
        Returns:
        --------
        float
            해당 백분위수의 값
        """
        if self.sorted_data is None:
            raise ValueError("데이터가 로드되지 않았습니다.")
        
        # 캐시된 값이 있으면 반환
        if percentile in self.percentile_cache:
            return self.percentile_cache[percentile]
        
        # 백분위수 계산
        value = np.percentile(self.sorted_data, percentile)
        
        # 자주 사용되는 백분위수 캐싱
        if percentile in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            self.percentile_cache[percentile] = value
            
        return value
    
    def get_percentile_of_value(self, value):
        """
        주어진 값의 백분위수 계산
        
        Parameters:
        -----------
        value : float
            백분위수를 구할 값
            
        Returns:
        --------
        float
            해당 값의 백분위수 (0-100)
        """
        if self.sorted_data is None:
            raise ValueError("데이터가 로드되지 않았습니다.")
        
        # 이진 탐색으로 위치 찾기
        idx = bisect.bisect_left(self.sorted_data, value)
        
        # 백분위수 계산
        percentile = (idx / self.total_count) * 100
        
        return percentile
    
    def get_bounds_for_probability(self, target_prob, center=0.0, tolerance=1e-4, max_iterations=100):
        """
        주어진 확률을 갖는 대칭적 구간 [center-x, center+x] 찾기
        
        Parameters:
        -----------
        target_prob : float
            목표 확률 (0-1 사이)
        center : float
            구간의 중심값
        tolerance : float
            허용 오차
        max_iterations : int
            최대 반복 횟수
            
        Returns:
        --------
        tuple
            (lower_bound, upper_bound) - 목표 확률을 갖는 구간
        """
        if self.sorted_data is None:
            raise ValueError("데이터가 로드되지 않았습니다.")
        
        if target_prob <= 0 or target_prob >= 1:
            raise ValueError("목표 확률은 0과 1 사이의 값이어야 합니다.")
        
        # 이진 탐색으로 적절한 구간 찾기
        lower = 0.0
        upper = max(abs(self.sorted_data[0] - center), abs(self.sorted_data[-1] - center))
        
        iteration = 0
        while iteration < max_iterations:
            current_width = (lower + upper) / 2
            current_prob = self.get_probability_between(center - current_width, center + current_width)
            
            if abs(current_prob - target_prob) < tolerance:
                return (center - current_width, center + current_width)
            
            if current_prob < target_prob:
                lower = current_width
            else:
                upper = current_width
                
            iteration += 1
        
        # 최대 반복 후 가장 가까운 결과 반환
        final_width = (lower + upper) / 2
        return (center - final_width, center + final_width)
    
    def get_statistics(self):
        """
        분포의 기본 통계량 반환
        
        Returns:
        --------
        dict
            기본 통계량 정보
        """
        if self.sorted_data is None:
            raise ValueError("데이터가 로드되지 않았습니다.")
        
        stats = {
            "개수": self.total_count,
            "평균": np.mean(self.sorted_data),
            "중앙값": np.median(self.sorted_data),
            "표준편차": np.std(self.sorted_data),
            "최소값": self.sorted_data[0],
            "최대값": self.sorted_data[-1],
            "제1사분위수": self.get_value_at_percentile(25),
            "제3사분위수": self.get_value_at_percentile(75)
        }
        
        return stats


def create_distribution_from_simulation(price_range=3, n=2**24, seed=42, force_rerun=False):
    """
    시뮬레이션을 실행하고 분포 객체 생성
    
    Parameters:
    -----------
    price_range : int
        예가범위 선택 (2 또는 3)
    n : int
        시뮬레이션 횟수
    seed : int
        난수 발생 시드값
    force_rerun : bool
        기존 데이터가 있어도 강제로 재실행할지 여부
        
    Returns:
    --------
    BidPriceDistribution
        생성된 분포 객체
    """
    from simulate_output import simulate
    
    # 파일 경로 설정
    base_dir = Path("bidPriceStatsAnalysis/simulate/data")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    dist_file = base_dir / f"distribution_range_{price_range}_2pow24.pkl"
    
    # 기존 분포 객체가 있고 강제 재실행이 아니면 기존 객체 로드
    if os.path.exists(dist_file) and not force_rerun:
        print(f"기존 분포 객체를 로드합니다: {dist_file}")
        return BidPriceDistribution.load(dist_file)
    
    print(f"시뮬레이션을 시작합니다 (n={n}, range={price_range})...")
    start_time = time.time()
    
    # 시뮬레이션 실행
    data = simulate(n, price_range=price_range, seed=seed)
    
    # 분포 객체 생성
    dist = BidPriceDistribution(data)
    
    # 분포 객체 저장
    dist.save(dist_file)
    
    end_time = time.time()
    print(f"시뮬레이션 및 분포 객체 생성 완료: {end_time - start_time:.2f}초")
    
    return dist


def demo():
    """
    분포 조회 기능 데모
    """
    # 분포 객체 생성 또는 로드
    dist = create_distribution_from_simulation(price_range=3)
    
    # 기본 통계량 출력
    print("\n기본 통계량:")
    stats = dist.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 구간별 확률 조회 예시
    print("\n구간별 확률 조회 예시:")
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
        prob = dist.get_probability_between(lower, upper)
        query_time = (time.time() - start_time) * 1000  # 밀리초 단위
        print(f"구간 [{lower}, {upper}] 확률: {prob:.6f} (조회 시간: {query_time:.3f}ms)")
    
    # 백분위수 조회 예시
    print("\n백분위수 조회 예시:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    
    for p in percentiles:
        start_time = time.time()
        value = dist.get_value_at_percentile(p)
        query_time = (time.time() - start_time) * 1000  # 밀리초 단위
        print(f"{p}% 백분위수: {value:.6f} (조회 시간: {query_time:.3f}ms)")
    
    # 특정 확률을 갖는 구간 찾기 예시
    print("\n특정 확률을 갖는 구간 찾기 예시:")
    target_probs = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    
    for prob in target_probs:
        start_time = time.time()
        lower, upper = dist.get_bounds_for_probability(prob)
        query_time = (time.time() - start_time) * 1000  # 밀리초 단위
        print(f"확률 {prob:.2f}을 갖는 구간: [{lower:.6f}, {upper:.6f}] (조회 시간: {query_time:.3f}ms)")


if __name__ == "__main__":
    demo() 