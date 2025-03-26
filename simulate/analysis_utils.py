import numpy as np
import pandas as pd
from scipy import stats


def analyze_statistics(data):
    """
    데이터의 기본 통계 정보를 분석하여 요약 정보 반환
    
    Parameters:
    -----------
    data : list or numpy.ndarray
        분석할 데이터
        
    Returns:
    --------
    pandas.DataFrame
        통계 요약 정보를 담은 DataFrame
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # 기본 통계량 계산
    mean = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)
    
    # 사분위수
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    
    # 특정 확률 범위에 대한 값 계산
    percentiles = np.percentile(data, [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99])
    
    # 정규성 검정 (Shapiro-Wilk)
    # 데이터가 너무 큰 경우 샘플링하여 검정 (Shapiro-Wilk는 최대 5000개 샘플까지만 지원)
    if len(data) > 5000:
        sample_indices = np.random.choice(len(data), 5000, replace=False)
        sample_data = data[sample_indices]
        _, normality_p = stats.shapiro(sample_data)
    else:
        _, normality_p = stats.shapiro(data)
    
    # 왜도와 첨도
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    # 특정 구간에 속하는 데이터 비율 계산
    intervals = [
        (-3, -2), (-2, -1), (-1, 0), 
        (0, 1), (1, 2), (2, 3),
        (-0.5, 0.5), (-1, 1)
    ]
    
    interval_probs = {}
    for lower, upper in intervals:
        count = np.sum((data >= lower) & (data <= upper))
        interval_probs[f"{lower}~{upper}"] = count / len(data)
    
    # 요약 정보 DataFrame 생성
    summary = {
        "통계량": [
            "데이터 개수", "평균", "중앙값", "표준편차", 
            "최소값", "최대값", "제1사분위수", "제3사분위수", "IQR",
            "왜도", "첨도", "정규성 검정 p값"
        ],
        "값": [
            len(data), mean, median, std_dev, 
            min_val, max_val, q1, q3, iqr,
            skewness, kurtosis, normality_p
        ]
    }
    
    # 백분위수 정보 추가
    percentile_names = ["1%", "5%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "95%", "99%"]
    
    for name, value in zip(percentile_names, percentiles):
        summary["통계량"].append(f"{name} 백분위수")
        summary["값"].append(value)
    
    # 구간별 확률 정보 추가
    for interval, prob in interval_probs.items():
        summary["통계량"].append(f"구간 {interval} 확률")
        summary["값"].append(prob)
    
    return pd.DataFrame(summary)


def compare_distributions(data1, data2, sample_size=5000):
    """
    두 데이터셋의 분포를 비교하는 함수
    
    Parameters:
    -----------
    data1, data2 : list or numpy.ndarray
        비교할 두 데이터셋
    sample_size : int
        표본 크기 (데이터가 클 경우 샘플링)
        
    Returns:
    --------
    dict
        다양한 통계 검정 결과가 포함된 사전
    """
    if not isinstance(data1, np.ndarray):
        data1 = np.array(data1)
    if not isinstance(data2, np.ndarray):
        data2 = np.array(data2)
    
    # 데이터가 너무 클 경우 샘플링
    if len(data1) > sample_size:
        idx1 = np.random.choice(len(data1), sample_size, replace=False)
        data1_sample = data1[idx1]
    else:
        data1_sample = data1
        
    if len(data2) > sample_size:
        idx2 = np.random.choice(len(data2), sample_size, replace=False)
        data2_sample = data2[idx2]
    else:
        data2_sample = data2
    
    # Kolmogorov-Smirnov 검정 (분포 비교)
    ks_stat, ks_pvalue = stats.ks_2samp(data1, data2)
    
    # Mann-Whitney U 검정 (위치 비교)
    mw_stat, mw_pvalue = stats.mannwhitneyu(data1_sample, data2_sample)
    
    # t-검정 (평균 비교)
    t_stat, t_pvalue = stats.ttest_ind(data1_sample, data2_sample, equal_var=False)
    
    # 기본 통계량 비교
    stats1 = {
        "평균": np.mean(data1),
        "중앙값": np.median(data1),
        "표준편차": np.std(data1),
        "최소값": np.min(data1),
        "최대값": np.max(data1),
        "왜도": stats.skew(data1_sample),
        "첨도": stats.kurtosis(data1_sample)
    }
    
    stats2 = {
        "평균": np.mean(data2),
        "중앙값": np.median(data2),
        "표준편차": np.std(data2),
        "최소값": np.min(data2),
        "최대값": np.max(data2),
        "왜도": stats.skew(data2_sample),
        "첨도": stats.kurtosis(data2_sample)
    }
    
    # 결과 반환
    return {
        "KS_검정": {"통계량": ks_stat, "p값": ks_pvalue},
        "Mann_Whitney_U_검정": {"통계량": mw_stat, "p값": mw_pvalue},
        "t_검정": {"통계량": t_stat, "p값": t_pvalue},
        "데이터1_통계": stats1,
        "데이터2_통계": stats2
    }


def create_histogram_bins(data, num_bins=100):
    """
    데이터에 대한 히스토그램 구간과 빈도수 계산
    
    Parameters:
    -----------
    data : list or numpy.ndarray
        히스토그램을 생성할 데이터
    num_bins : int
        히스토그램 구간 수
        
    Returns:
    --------
    tuple
        (구간 경계값, 빈도수)
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # 히스토그램 계산
    counts, bin_edges = np.histogram(data, bins=num_bins)
    
    return bin_edges, counts 