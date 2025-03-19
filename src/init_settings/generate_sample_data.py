#!/usr/bin/env python
"""
입찰가 분석을 위한 샘플 데이터 생성 스크립트
"""
import os
import pandas as pd
import numpy as np
import argparse
import random
from datetime import datetime, timedelta
from tqdm import tqdm

def generate_bid_data(num_records=100, output_dir="data/raw"):
    """
    입찰 데이터 샘플 생성
    
    Parameters:
        num_records (int): 생성할 레코드 수
        output_dir (str): 출력 디렉토리
    
    Returns:
        str: 생성된 파일 경로
    """
    print(f"🔄 {num_records}개의 입찰 데이터 샘플 생성 중...")
    
    # 기준 날짜 설정 (2022년부터 현재까지)
    start_date = datetime(2022, 1, 1)
    end_date = datetime.now()
    date_range = (end_date - start_date).days
    
    # 더미 데이터 생성
    data = {
        "공고번호": [f"BID{str(i).zfill(6)}" for i in range(1, num_records + 1)],
        "공고제목": [f"입찰 공고 샘플 {i}" for i in range(1, num_records + 1)],
        "공고내용": [f"이것은 입찰 공고 {i}의 상세 내용입니다. 자세한 사항은 첨부파일을 참고하세요." for i in range(1, num_records + 1)],
        "공고종류": np.random.choice(["물품", "공사", "용역", "기타"], num_records),
        "업종": np.random.choice(["건설", "IT", "제조", "서비스", "컨설팅"], num_records),
        "낙찰방법": np.random.choice(["최저가", "적격심사", "협상계약", "제한경쟁"], num_records),
        "입찰일자": [start_date + timedelta(days=random.randint(0, date_range)) for _ in range(num_records)],
        "개찰일시": [start_date + timedelta(days=random.randint(14, date_range)) for _ in range(num_records)],
        "기초금액": np.random.uniform(10000000, 1000000000, num_records),
        "예정금액": np.random.uniform(9000000, 950000000, num_records),
        "예가": np.random.uniform(8500000, 900000000, num_records),
        "투찰가": np.random.uniform(8000000, 850000000, num_records),
        "업체명": [f"업체{random.randint(1, 20)}" for _ in range(num_records)],
        "참여자수": np.random.randint(1, 30, num_records),
        "거래적정성": np.random.choice([0, 1], num_records, p=[0.9, 0.1])  # 10%만 비정상 거래로 표시
    }
    
    # DataFrame 생성
    df = pd.DataFrame(data)
    
    # 날짜 형식 변환
    df["입찰일자"] = pd.to_datetime(df["입찰일자"]).dt.strftime("%Y-%m-%d")
    df["개찰일시"] = pd.to_datetime(df["개찰일시"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # 출력 디렉토리 확인 및 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"bid_data_{timestamp}.csv")
    df.to_csv(output_file, index=False, encoding="utf-8")
    
    print(f"✅ 입찰 데이터 샘플 생성 완료: {output_file}")
    return output_file

def generate_notice_data(num_records=150, output_dir="data/raw"):
    """
    공고 데이터 샘플 생성
    
    Parameters:
        num_records (int): 생성할 레코드 수
        output_dir (str): 출력 디렉토리
    
    Returns:
        str: 생성된 파일 경로
    """
    print(f"🔄 {num_records}개의 공고 데이터 샘플 생성 중...")
    
    # 기준 날짜 설정 (2022년부터 현재까지)
    start_date = datetime(2022, 1, 1)
    end_date = datetime.now()
    date_range = (end_date - start_date).days
    
    # 더미 데이터 생성
    data = {
        "공고번호": [f"NOTICE{str(i).zfill(6)}" for i in range(1, num_records + 1)],
        "공고제목": [f"공고 샘플 {i}" for i in range(1, num_records + 1)],
        "공고내용": [f"이것은 공고 {i}의 상세 내용입니다. 발주처의 요구사항을 확인하세요." for i in range(1, num_records + 1)],
        "공고종류": np.random.choice(["물품", "공사", "용역", "기타"], num_records),
        "업종": np.random.choice(["건설", "IT", "제조", "서비스", "컨설팅"], num_records),
        "낙찰방법": np.random.choice(["최저가", "적격심사", "협상계약", "제한경쟁"], num_records),
        "입찰일자": [start_date + timedelta(days=random.randint(0, date_range)) for _ in range(num_records)],
        "개찰일시": [start_date + timedelta(days=random.randint(14, date_range)) for _ in range(num_records)],
        "기초금액": np.random.uniform(10000000, 1000000000, num_records),
        "예정금액": np.random.uniform(9000000, 950000000, num_records),
        "발주처": np.random.choice(["중앙정부", "지방자치단체", "공공기관", "기타"], num_records),
        "지역": np.random.choice(["서울", "경기", "인천", "부산", "대구", "광주", "대전", "전국"], num_records)
    }
    
    # DataFrame 생성
    df = pd.DataFrame(data)
    
    # 날짜 형식 변환
    df["입찰일자"] = pd.to_datetime(df["입찰일자"]).dt.strftime("%Y-%m-%d")
    df["개찰일시"] = pd.to_datetime(df["개찰일시"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # 출력 디렉토리 확인 및 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"notice_data_{timestamp}.csv")
    df.to_csv(output_file, index=False, encoding="utf-8")
    
    print(f"✅ 공고 데이터 샘플 생성 완료: {output_file}")
    return output_file

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="입찰가 분석을 위한 샘플 데이터 생성")
    parser.add_argument("--bid-records", type=int, default=100, help="생성할 입찰 데이터 레코드 수 (기본값: 100)")
    parser.add_argument("--notice-records", type=int, default=150, help="생성할 공고 데이터 레코드 수 (기본값: 150)")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="출력 디렉토리 (기본값: data/raw)")
    parser.add_argument("--skip-bid", action="store_true", help="입찰 데이터 생성 건너뛰기")
    parser.add_argument("--skip-notice", action="store_true", help="공고 데이터 생성 건너뛰기")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 확인 및 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("📊 입찰가 분석을 위한 샘플 데이터 생성을 시작합니다.")
    
    if not args.skip_bid:
        bid_file = generate_bid_data(args.bid_records, args.output_dir)
    
    if not args.skip_notice:
        notice_file = generate_notice_data(args.notice_records, args.output_dir)
    
    print("\n✅ 샘플 데이터 생성이 완료되었습니다.")
    print("이제 다음 명령어로 데이터 전처리 파이프라인을 실행할 수 있습니다:")
    print("python src/preprocess_upload_mongo.py --file-pattern=\"*.csv\"")

if __name__ == "__main__":
    main() 