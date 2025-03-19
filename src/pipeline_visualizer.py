#!/usr/bin/env python
"""
파이프라인 시각화 모듈
"""
import os
import time
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime

logger = logging.getLogger(__name__)

class PipelineVisualizer:
    """파이프라인 시각화 클래스"""
    
    def __init__(self, pipeline_name="입찰가 데이터 전처리 파이프라인", output_dir="results"):
        """
        초기화 함수
        
        Parameters:
            pipeline_name (str): 파이프라인 이름
            output_dir (str): 결과 저장 디렉토리
        """
        self.pipeline_name = pipeline_name
        self.output_dir = output_dir
        self.pipeline_steps = []
        self.metrics = {}
        self.progress_bars = {}
        self.start_time = None
        self.end_time = None
        
        # 결과 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
    
    def start_pipeline(self):
        """파이프라인 시작"""
        self.start_time = time.time()
        logger.info(f"🚀 파이프라인 시작: {self.pipeline_name}")
        print(f"\n{'='*50}")
        print(f"🚀 {self.pipeline_name} 시작")
        print(f"{'='*50}")
    
    def add_step(self, step_name, description=None):
        """
        파이프라인 단계 추가
        
        Parameters:
            step_name (str): 단계 이름
            description (str, optional): 단계 설명
        
        Returns:
            int: 단계 인덱스
        """
        step_idx = len(self.pipeline_steps)
        step_info = {
            "index": step_idx,
            "name": step_name,
            "description": description or step_name,
            "start_time": time.time(),
            "end_time": None,
            "status": "진행 중",
            "metrics": {}
        }
        self.pipeline_steps.append(step_info)
        
        logger.info(f"📊 파이프라인 단계 {step_idx+1}: {step_name} 시작")
        print(f"\n📊 단계 {step_idx+1}: {step_name}")
        if description:
            print(f"   {description}")
        print(f"{'-'*50}")
        
        return step_idx
    
    def complete_step(self, step_idx, metrics=None):
        """
        파이프라인 단계 완료
        
        Parameters:
            step_idx (int): 단계 인덱스
            metrics (dict, optional): 단계 지표
        """
        if step_idx < len(self.pipeline_steps):
            step_info = self.pipeline_steps[step_idx]
            step_info["end_time"] = time.time()
            step_info["status"] = "완료"
            
            duration = step_info["end_time"] - step_info["start_time"]
            
            if metrics:
                step_info["metrics"] = metrics
                self.metrics.update(metrics)
            
            logger.info(f"✅ 파이프라인 단계 {step_idx+1}: {step_info['name']} 완료 (소요시간: {duration:.2f}초)")
            print(f"\n✅ 단계 {step_idx+1}: {step_info['name']} 완료 (소요시간: {duration:.2f}초)")
            
            if metrics:
                print("   단계 지표:")
                for k, v in metrics.items():
                    print(f"   - {k}: {v}")
    
    def create_progress_bar(self, step_idx, total, description=None):
        """
        진행 표시줄 생성
        
        Parameters:
            step_idx (int): 단계 인덱스
            total (int): 전체 항목 수
            description (str, optional): 진행 표시줄 설명
        
        Returns:
            tqdm: 진행 표시줄 객체
        """
        if step_idx < len(self.pipeline_steps):
            step_info = self.pipeline_steps[step_idx]
            desc = description or f"단계 {step_idx+1}: {step_info['name']}"
            
            pbar = tqdm(total=total, desc=desc, position=0, leave=True)
            self.progress_bars[step_idx] = pbar
            return pbar
        return None
    
    def end_pipeline(self):
        """파이프라인 종료"""
        self.end_time = time.time()
        total_duration = self.end_time - self.start_time
        
        # 종료되지 않은 단계가 있다면 종료
        for i, step in enumerate(self.pipeline_steps):
            if step["status"] != "완료":
                step["end_time"] = time.time()
                step["status"] = "중단됨"
        
        # 진행 표시줄 닫기
        for pbar in self.progress_bars.values():
            pbar.close()
        
        logger.info(f"✅ 파이프라인 종료: {self.pipeline_name} (총 소요시간: {total_duration:.2f}초)")
        print(f"\n{'='*50}")
        print(f"✅ {self.pipeline_name} 종료 (총 소요시간: {total_duration:.2f}초)")
        print(f"{'='*50}")
        
        # 요약 지표 출력
        if self.metrics:
            print("\n📊 파이프라인 지표 요약:")
            for k, v in self.metrics.items():
                print(f"   - {k}: {v}")
    
    def visualize_pipeline(self, show_plot=True, save_plot=True):
        """
        파이프라인 시각화
        
        Parameters:
            show_plot (bool): 그래프 표시 여부
            save_plot (bool): 그래프 저장 여부
        
        Returns:
            str: 저장된 파일 경로 (저장한 경우)
        """
        try:
            import matplotlib
            matplotlib.use('Agg' if not show_plot else 'TkAgg')
            
            # 그래프 생성
            G = nx.DiGraph()
            
            # 노드 추가
            G.add_node("시작", pos=(0, 0))
            
            for i, step in enumerate(self.pipeline_steps):
                G.add_node(step["name"], pos=(i+1, 0))
                
                # 이전 노드와 연결
                if i == 0:
                    G.add_edge("시작", step["name"])
                else:
                    G.add_edge(self.pipeline_steps[i-1]["name"], step["name"])
            
            G.add_node("종료", pos=(len(self.pipeline_steps)+1, 0))
            if self.pipeline_steps:
                G.add_edge(self.pipeline_steps[-1]["name"], "종료")
            
            # 그래프 레이아웃 설정
            pos = nx.get_node_attributes(G, 'pos')
            
            # 그래프 그리기
            plt.figure(figsize=(12, 6))
            
            # 노드 색상 설정
            node_colors = []
            for node in G.nodes:
                if node == "시작":
                    node_colors.append('lightgreen')
                elif node == "종료":
                    node_colors.append('lightblue')
                else:
                    # 단계 상태에 따른 색상
                    step_found = False
                    for step in self.pipeline_steps:
                        if step["name"] == node:
                            if step["status"] == "완료":
                                node_colors.append('lightgreen')
                            elif step["status"] == "진행 중":
                                node_colors.append('yellow')
                            else:
                                node_colors.append('lightcoral')
                            step_found = True
                            break
                    if not step_found:
                        node_colors.append('lightgray')
            
            # 노드 그리기
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, alpha=0.8)
            
            # 엣지 그리기
            nx.draw_networkx_edges(G, pos, edge_color='gray', width=2, arrowsize=20)
            
            # 레이블 그리기
            nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
            
            # 제목 및 축 설정
            plt.title(f"{self.pipeline_name} 구조")
            plt.axis('off')
            
            # 파일로 저장
            if save_plot:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(self.output_dir, f"pipeline_visualization_{timestamp}.png")
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"파이프라인 시각화 저장: {output_file}")
                
                if show_plot:
                    plt.show()
                plt.close()
                
                return output_file
            elif show_plot:
                plt.show()
                plt.close()
            
        except ImportError as e:
            logger.warning(f"시각화 라이브러리 로드 실패: {e}")
            return None
        except Exception as e:
            logger.error(f"파이프라인 시각화 실패: {e}")
            return None
    
    def generate_report(self, additional_info=None):
        """
        파이프라인 보고서 생성
        
        Parameters:
            additional_info (dict, optional): 추가 정보
        
        Returns:
            str: 보고서 파일 경로
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"pipeline_report_{timestamp}.html")
        
        # HTML 보고서 템플릿
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.pipeline_name} 실행 보고서</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #4a86e8; }}
                .step {{ background-color: #f5f5f5; padding: 10px; margin: 10px 0; border-left: 5px solid #4a86e8; }}
                .step-complete {{ border-left-color: #6aa84f; }}
                .step-failed {{ border-left-color: #e06666; }}
                .metrics {{ background-color: #ebf5fb; padding: 10px; margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4a86e8; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>{self.pipeline_name} 실행 보고서</h1>
            <p>실행 시간: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')} ~ {datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>총 소요시간: {self.end_time - self.start_time:.2f}초</p>
            
            <h2>파이프라인 단계</h2>
        """
        
        # 각 단계 정보 추가
        for step in self.pipeline_steps:
            status_class = "step-complete" if step["status"] == "완료" else "step-failed"
            duration = (step["end_time"] or time.time()) - step["start_time"]
            
            html_content += f"""
            <div class="step {status_class}">
                <h3>단계 {step['index']+1}: {step['name']}</h3>
                <p>{step['description']}</p>
                <p>상태: {step['status']}</p>
                <p>소요시간: {duration:.2f}초</p>
            """
            
            # 단계별 지표 추가
            if step["metrics"]:
                html_content += """
                <div class="metrics">
                    <h4>단계 지표</h4>
                    <table>
                        <tr><th>지표</th><th>값</th></tr>
                """
                
                for k, v in step["metrics"].items():
                    html_content += f"""
                    <tr><td>{k}</td><td>{v}</td></tr>
                    """
                
                html_content += """
                    </table>
                </div>
                """
            
            html_content += """
            </div>
            """
        
        # 전체 지표 요약
        if self.metrics:
            html_content += """
            <h2>파이프라인 지표 요약</h2>
            <div class="metrics">
                <table>
                    <tr><th>지표</th><th>값</th></tr>
            """
            
            for k, v in self.metrics.items():
                html_content += f"""
                <tr><td>{k}</td><td>{v}</td></tr>
                """
            
            html_content += """
                </table>
            </div>
            """
        
        # 추가 정보
        if additional_info:
            html_content += """
            <h2>추가 정보</h2>
            <div class="metrics">
                <table>
                    <tr><th>항목</th><th>값</th></tr>
            """
            
            for k, v in additional_info.items():
                html_content += f"""
                <tr><td>{k}</td><td>{v}</td></tr>
                """
            
            html_content += """
                </table>
            </div>
            """
        
        # 시각화 이미지 추가
        visual_file = self.visualize_pipeline(show_plot=False, save_plot=True)
        if visual_file:
            html_content += f"""
            <h2>파이프라인 시각화</h2>
            <img src="{os.path.relpath(visual_file, self.output_dir)}" alt="Pipeline Visualization" style="max-width: 100%;">
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # HTML 파일로 저장
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"파이프라인 보고서 생성: {report_file}")
        return report_file
    
    def update_step(self, step_idx, step_name, description=None):
        """
        파이프라인 단계 정보 업데이트
        
        Parameters:
            step_idx (int): 단계 인덱스
            step_name (str): 새 단계 이름
            description (str, optional): 새 단계 설명
            
        Returns:
            bool: 업데이트 성공 여부
        """
        if step_idx < len(self.pipeline_steps):
            step_info = self.pipeline_steps[step_idx]
            step_info["name"] = step_name
            if description:
                step_info["description"] = description
            
            logger.info(f"🔄 파이프라인 단계 {step_idx+1} 정보 업데이트: {step_name}")
            print(f"\n🔄 단계 {step_idx+1}: {step_name}")
            if description:
                print(f"   {description}")
            print(f"{'-'*50}")
            
            return True
        return False

def create_visualizer(pipeline_name="입찰가 데이터 전처리 파이프라인", output_dir="results"):
    """
    시각화 객체 생성 헬퍼 함수
    
    Parameters:
        pipeline_name (str): 파이프라인 이름
        output_dir (str): 결과 저장 디렉토리
    
    Returns:
        PipelineVisualizer: 시각화 객체
    """
    return PipelineVisualizer(pipeline_name, output_dir) 