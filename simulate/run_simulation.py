import argparse
import time
from pathlib import Path

from simulate_output import simulate, save_data, load_data, get_probability_between
from advanced_lookups import BidPriceDistribution, create_distribution_from_simulation


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description='입찰가격 분포 시뮬레이션 실행')
    
    parser.add_argument('--range', type=int, default=3, choices=[2, 3],
                        help='예가범위 선택 (2 또는 3)')
    
    parser.add_argument('--n', type=int, default=2**24,
                        help='시뮬레이션 횟수 (기본값: 2^24)')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='난수 발생 시드값 (기본값: 42)')
    
    parser.add_argument('--force', action='store_true',
                        help='기존 데이터가 있어도 강제로 재실행')
    
    parser.add_argument('--mode', type=str, default='advanced', choices=['basic', 'advanced'],
                        help='실행 모드 선택 (basic: 기본 시뮬레이션, advanced: 고급 조회 클래스 사용)')
    
    parser.add_argument('--query', action='store_true',
                        help='대화형 모드로 구간 확률 조회')
    
    return parser.parse_args()


def run_basic_simulation(args):
    """기본 시뮬레이션 모드 실행"""
    print(f"기본 시뮬레이션 모드 (range: {args.range}, n: {args.n})")
    
    # 파일 경로 설정
    base_dir = Path("bidPriceStatsAnalysis/simulate/data")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    data_file = base_dir / f"simulation_data_range_{args.range}_2pow24.pkl"
    lookup_file = base_dir / f"sorted_lookup_range_{args.range}_2pow24.pkl"
    
    # 데이터가 이미 존재하는지 확인
    if Path(data_file).exists() and Path(lookup_file).exists() and not args.force:
        print(f"기존 데이터를 로드합니다: {data_file}")
        sorted_data, total_count = load_data(lookup_file)
    else:
        print("시뮬레이션을 시작합니다...")
        start_time = time.time()
        
        # 시뮬레이션 실행
        data = simulate(args.n, price_range=args.range, seed=args.seed)
        
        # 데이터 기본 통계
        print(f"전체 데이터 수: {len(data)}")
        print(f"평균: {sum(data)/len(data)}")
        print(f"최소값: {min(data)}")
        print(f"최대값: {max(data)}")
        
        # 정렬된 데이터와 전체 수 준비
        sorted_data = sorted(data)
        total_count = len(sorted_data)
        
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
    
    # 대화형 모드
    if args.query:
        interactive_query_basic(sorted_data, total_count)


def run_advanced_simulation(args):
    """고급 시뮬레이션 모드 실행 (BidPriceDistribution 클래스 사용)"""
    print(f"고급 시뮬레이션 모드 (range: {args.range}, n: {args.n})")
    
    # BidPriceDistribution 객체 생성 또는 로드
    dist = create_distribution_from_simulation(
        price_range=args.range, 
        n=args.n, 
        seed=args.seed,
        force_rerun=args.force
    )
    
    # 기본 통계량 출력
    stats = dist.get_statistics()
    print("\n기본 통계량:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 구간별 확률 조회 테스트
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
        prob = dist.get_probability_between(lower, upper)
        query_time = (time.time() - start_time) * 1000  # 밀리초 단위
        print(f"구간 [{lower}, {upper}] 확률: {prob:.6f} (조회 시간: {query_time:.3f}ms)")
    
    # 대화형 모드
    if args.query:
        interactive_query_advanced(dist)


def interactive_query_basic(sorted_data, total_count):
    """기본 모드에서 대화형 구간 확률 조회"""
    print("\n대화형 구간 확률 조회 모드 시작 (종료하려면 'q' 입력)")
    
    while True:
        try:
            user_input = input("\n두 값 사이의 확률을 조회하려면 두 숫자를 입력하세요 (예: -1 1): ")
            
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


def interactive_query_advanced(dist):
    """고급 모드에서 대화형 구간 확률 조회"""
    print("\n대화형 구간 확률 조회 모드 시작")
    print("옵션: ")
    print("1. 구간 확률 조회 (예: '-1 1')")
    print("2. 백분위수의 값 조회 (예: 'p 50')")
    print("3. 값의 백분위수 조회 (예: 'v 0.5')")
    print("4. 확률에 해당하는 구간 찾기 (예: 'r 0.9')")
    print("q. 종료")
    
    while True:
        try:
            user_input = input("\n명령어와 값을 입력하세요: ")
            
            if user_input.lower() == 'q':
                break
            
            parts = user_input.split()
            
            if len(parts) == 2 and parts[0].lower() == 'p':
                # 백분위수 조회
                percentile = float(parts[1])
                start_time = time.time()
                value = dist.get_value_at_percentile(percentile)
                query_time = (time.time() - start_time) * 1000
                print(f"{percentile}% 백분위수 값: {value:.6f} (조회 시간: {query_time:.3f}ms)")
                
            elif len(parts) == 2 and parts[0].lower() == 'v':
                # 값의 백분위수 조회
                value = float(parts[1])
                start_time = time.time()
                percentile = dist.get_percentile_of_value(value)
                query_time = (time.time() - start_time) * 1000
                print(f"값 {value}의 백분위수: {percentile:.2f}% (조회 시간: {query_time:.3f}ms)")
                
            elif len(parts) == 2 and parts[0].lower() == 'r':
                # 확률에 해당하는 구간 찾기
                prob = float(parts[1])
                start_time = time.time()
                lower, upper = dist.get_bounds_for_probability(prob)
                query_time = (time.time() - start_time) * 1000
                print(f"확률 {prob:.2f}에 해당하는 구간: [{lower:.6f}, {upper:.6f}] (조회 시간: {query_time:.3f}ms)")
                
            elif len(parts) == 2:
                # 구간 확률 조회
                lower, upper = map(float, parts)
                if lower > upper:
                    lower, upper = upper, lower
                    
                start_time = time.time()
                prob = dist.get_probability_between(lower, upper)
                query_time = (time.time() - start_time) * 1000
                
                print(f"구간 [{lower}, {upper}] 확률: {prob:.6f} (조회 시간: {query_time:.3f}ms)")
                
            else:
                print("올바른 형식으로 입력해주세요.")
                
        except ValueError:
            print("올바른 형식으로 입력해주세요.")
        except Exception as e:
            print(f"오류 발생: {e}")


def main():
    """메인 함수"""
    args = parse_args()
    
    if args.mode == 'basic':
        run_basic_simulation(args)
    else:
        run_advanced_simulation(args)


if __name__ == "__main__":
    main() 