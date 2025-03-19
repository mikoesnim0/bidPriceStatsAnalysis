import os
import pandas as pd
import logging
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

class DataLoader:
    """원시 데이터를 로드하는 클래스"""
    
    def __init__(self, data_dir=None):
        """
        초기화 함수
        
        Parameters:
            data_dir (str): 데이터 디렉토리 경로
        """
        self.data_dir = data_dir or os.getenv('DATA_DIR', './data')
        logger.info(f"데이터 디렉토리: {self.data_dir}")
    
    def load_raw_data(self, file_pattern=None):
        """
        원시 데이터 파일을 로드
        
        Parameters:
            file_pattern (str): 파일 패턴 (예: '*.csv')
            
        Returns:
            dict: 파일명을 키로, DataFrame을 값으로 하는 딕셔너리
        """
        if file_pattern is None:
            file_pattern = "*.csv"  # 기본 파일 패턴
        
        dataset_dict = {}
        
        try:
            # 디렉토리 내 모든 CSV 파일 찾기
            import glob
            file_paths = glob.glob(os.path.join(self.data_dir, file_pattern))
            
            if not file_paths:
                logger.warning(f"⚠️ {self.data_dir} 디렉토리에서 {file_pattern} 패턴의 파일을 찾을 수 없습니다.")
                return dataset_dict
            
            # 각 파일 로드
            for file_path in file_paths:
                file_name = os.path.basename(file_path).split('.')[0]
                logger.info(f"🔄 파일 로드 중: {file_path}")
                
                try:
                    # 파일 확장자에 따라 다른 로드 방식 적용
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path)
                    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                        df = pd.read_excel(file_path)
                    elif file_path.endswith('.json'):
                        df = pd.read_json(file_path)
                    else:
                        logger.warning(f"⚠️ 지원되지 않는 파일 형식: {file_path}")
                        continue
                    
                    # 데이터셋 저장
                    dataset_dict[f"DataSet_{file_name}"] = df
                    logger.info(f"✅ 로드 완료: {file_name}, 레코드 수: {len(df)}")
                    
                except Exception as e:
                    logger.error(f"❌ 파일 로드 실패: {file_path}, 오류: {e}")
            
            return dataset_dict
            
        except Exception as e:
            logger.error(f"❌ 데이터 로드 중 오류 발생: {e}")
            return {} 