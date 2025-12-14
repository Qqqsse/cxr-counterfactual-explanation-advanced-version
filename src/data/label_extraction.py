"""
標籤提取模組
從 IU X-ray Dataset 的放射科報告中提取疾病標籤
"""

import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import sys
import datetime

# --- 加入 Logger 類別 ---
class DualLogger:
    def __init__(self, filename, original_stdout):
        self.terminal = original_stdout
        self.log = open(filename, "w", encoding='utf-8-sig')

    def write(self, message):
        try:
            if message:
                self.terminal.write(message)
                self.terminal.flush()
        except Exception:
            sys.__stderr__.write(message) # 出錯時改用標準錯誤輸出

        try:
            self.log.write(message)
            self.log.flush()
        except Exception:
            pass

    def flush(self):
        try:
            self.terminal.flush()
            self.log.flush()
        except:
            pass

class LabelExtractor:
    """從放射科報告提取疾病標籤"""
    
    # 疾病關鍵字對應表
    DISEASE_KEYWORDS = {
        'Cardiomegaly': [
            'cardiomegaly', 'cardiac enlargement', 'enlarged heart',
            'heart is enlarged', 'cardiac silhouette is enlarged'
        ],
        'Infiltration': [
            'infiltrate', 'infiltration', 'opacity', 'opacification',
            'consolidation', 'airspace disease'
        ],
        'Pneumonia': [
            'pneumonia', 'pneumonic'
        ],
        'Edema': [
            'edema', 'pulmonary edema', 'fluid overload',
            'congestive heart failure', 'chf'
        ],
        'Atelectasis': [
            'atelectasis', 'collapse', 'volume loss'
        ],
        'Pneumothorax': [
            'pneumothorax', 'air in pleural space'
        ],
        'Pleural_Effusion': [
            'pleural effusion', 'effusion', 'fluid in pleural space'
        ],
        'Normal': []  # 將由排除法決定
    }
    
    # 排除關鍵字（表示正常）
    NORMAL_KEYWORDS = [
        'normal', 'clear', 'no acute', 'no active', 'unremarkable',
        'within normal limits', 'no abnormality'
    ]
    
    # 否定詞（表示不存在某疾病）
    NEGATION_KEYWORDS = [
        'no', 'not', 'without', 'negative for', 'rule out',
        'no evidence', 'free of', 'absence of'
    ]
    
    def __init__(self, reports_path: Path, projections_path: Path):
        """
        初始化標籤提取器
        
        Args:
            reports_path: 報告 CSV 路徑
            projections_path: 影像對應 CSV 路徑
        """
        self.reports_path = Path(reports_path)
        self.projections_path = Path(projections_path)
        self.reports_data = []
        
    def load_reports_from_csv(self) -> pd.DataFrame:
        """從 CSV 檔案載入報告並整理"""
        
        print(f"正在讀取報告: {self.reports_path}")
        df_reports = pd.read_csv(self.reports_path)
        df_proj = pd.read_csv(self.projections_path)
        
        # 欄位整理
        df_reports = df_reports.rename(columns={'uid': 'report_id', 'MeSH': 'mesh_tags'})
        df_reports['findings'] = df_reports['findings'].fillna('')
        df_reports['impression'] = df_reports['impression'].fillna('')
        
        # 產生 full_text 用於標籤提取
        df_reports['full_text'] = (df_reports['findings'] + ' ' + df_reports['impression']).str.lower()
        
        # 處理影像對應 (一對多)
        image_map = df_proj.groupby('uid')['filename'].apply(list).to_dict()
        df_reports['image_ids'] = df_reports['report_id'].map(image_map)
        
        # 移除沒有對應影像的報告
        df_reports = df_reports.dropna(subset=['image_ids'])
        
        print(f"✅ 成功載入 {len(df_reports)} 份報告")
        return df_reports
            
    def extract_disease_labels(self, report_text: str) -> Dict[str, bool]:
        """
        從報告文字提取疾病標籤
        
        Args:
            report_text: 報告文字（已轉小寫）
        
        Returns:
            疾病標籤字典 {疾病名稱: True/False}
        """
        labels = {}
        report_text = report_text.lower()
        
        for disease, keywords in self.DISEASE_KEYWORDS.items():
            if disease == 'Normal':
                continue
            
            # 檢查是否存在疾病關鍵字
            disease_found = False
            for keyword in keywords:
                if keyword in report_text:
                    # 檢查前後是否有否定詞
                    if not self._has_negation(report_text, keyword):
                        disease_found = True
                        break
            
            labels[disease] = disease_found
        
        # 判斷是否為正常
        # 如果沒有任何疾病，且有正常關鍵字，則標記為正常
        has_any_disease = any(labels.values())
        has_normal_keyword = any(kw in report_text for kw in self.NORMAL_KEYWORDS)
        
        labels['Normal'] = (not has_any_disease) and has_normal_keyword
        
        return labels
    
    def _has_negation(self, text: str, keyword: str) -> bool:
        """
        檢查關鍵字前是否有否定詞
        
        Args:
            text: 完整文字
            keyword: 目標關鍵字
        
        Returns:
            是否有否定詞
        """
        # 找到關鍵字位置
        idx = text.find(keyword)
        if idx == -1:
            return False
        
        # 檢查前 50 個字元
        context = text[max(0, idx-50):idx]
        
        # 檢查是否有否定詞
        for neg_word in self.NEGATION_KEYWORDS:
            if neg_word in context:
                return True
        
        return False
    
    def create_binary_labels(self, df_reports: pd.DataFrame, 
                           positive_class: str,
                           negative_class: str = 'Normal') -> pd.DataFrame:
        """
        建立二元分類標籤
        
        Args:
            df_reports: 報告 DataFrame
            positive_class: 正例疾病（如 'Cardiomegaly'）
            negative_class: 負例疾病（如 'Normal'）
        
        Returns:
            包含二元標籤的 DataFrame
        """
        print(f"\n建立二元分類標籤: {positive_class} vs. {negative_class}")
        
        binary_labels = []
        
        for idx, row in df_reports.iterrows():
            # 提取所有疾病標籤
            disease_labels = self.extract_disease_labels(row['full_text'])
            
            # 判斷是否屬於正例或負例
            is_positive = disease_labels.get(positive_class, False)
            is_negative = disease_labels.get(negative_class, False)
            
            # 排除同時存在多種疾病的樣本（混雜因素）
            other_diseases = [d for d in disease_labels.keys() 
                            if d not in [positive_class, negative_class]]
            has_other_disease = any(disease_labels[d] for d in other_diseases)
            
            # 為每個影像建立標籤
            for img_id in row['image_ids']:
                if is_positive and not has_other_disease:
                    label = 1  # 正例
                    label_name = positive_class
                elif is_negative and not has_other_disease:
                    label = 0  # 負例
                    label_name = negative_class
                else:
                    continue  # 排除混雜或不確定的樣本
                
                binary_labels.append({
                    'image_id': img_id,
                    'report_id': row['report_id'],
                    'label': label,
                    'label_name': label_name,
                    'findings': row['findings'],
                    'impression': row['impression']
                })
        
        df_binary = pd.DataFrame(binary_labels)
        
        # 統計
        print(f"✅ 建立完成")
        print(f"   {positive_class}: {(df_binary['label'] == 1).sum()} 個樣本")
        print(f"   {negative_class}: {(df_binary['label'] == 0).sum()} 個樣本")
        print(f"   總計: {len(df_binary)} 個樣本")
        
        return df_binary
    
    def validate_labels(self, df_labels: pd.DataFrame, n_samples: int = 20) -> None:
        """
        驗證標籤正確性（人工抽樣檢查）
        
        Args:
            df_labels: 標籤 DataFrame
            n_samples: 抽樣數量
        """
        print(f"\n📋 隨機抽樣 {n_samples} 個樣本進行標籤驗證：")
        print("=" * 80)
        
        samples = df_labels.sample(min(n_samples, len(df_labels)))
        
        for idx, row in samples.iterrows():
            print(f"\n樣本 {idx + 1}:")
            print(f"  Image ID: {row['image_id']}")
            print(f"  Label: {row['label_name']} (值: {row['label']})")
            print(f"  Findings: {row['findings'][:200]}...")
            print(f"  Impression: {row['impression'][:150]}...")
            print("-" * 80)
        
        print("\n⚠️ 請人工檢查上述標籤是否正確！")
    
    def save_labels(self, df_labels: pd.DataFrame, output_path: Path) -> None:
        """
        儲存標籤檔案
        
        Args:
            df_labels: 標籤 DataFrame
            output_path: 輸出檔案路徑
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_labels.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 標籤已儲存至: {output_path}")


def main():
    # --- 1. 路徑設定 ---
    try:
        script_path = Path(__file__).resolve()
        project_root = script_path.parents[2] # 往上找三層 (src/data -> src -> root)
    except NameError:
        project_root = Path.cwd()
        if (project_root / "src").exists() and not (project_root / "data").exists():
             project_root = project_root.parent

    # --- 2. 設定 Log 儲存位置 (新增功能) ---
    LOGS_DIR = project_root / "results" / "logs"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = LOGS_DIR / f"label_extraction_log_{timestamp}.txt"
    
    # 啟動 Logger (這行之後所有的 print 都會被存起來)
    sys.stdout = DualLogger(log_filename, sys.stdout)
    
    print("="*60)
    print(f"🚀 程式啟動時間: {timestamp}")
    print(f"📂 專案根目錄: {project_root}")
    print(f"📝 執行紀錄將儲存於: {log_filename}")
    print("="*60)

    # --- 3. 設定資料路徑 ---
    RAW_DATA_DIR = project_root / "data" / "raw"
    REPORTS_CSV = RAW_DATA_DIR / "indiana_reports.csv"
    PROJECTIONS_CSV = RAW_DATA_DIR / "indiana_projections.csv"
    OUTPUT_DIR = project_root / "data" / "labels"
    
    # 檢查檔案
    if not REPORTS_CSV.exists():
        print(f"❌ [Error] 找不到報告檔案: {REPORTS_CSV}")
        return
    if not PROJECTIONS_CSV.exists():
        print(f"❌ [Error] 找不到對應檔案: {PROJECTIONS_CSV}")
        return

    # --- 4. 執行核心邏輯 ---
    print("\n🔄 [Step 1] 初始化提取器...")
    extractor = LabelExtractor(REPORTS_CSV, PROJECTIONS_CSV)
    
    print("📖 [Step 2] 載入報告...")
    df_reports = extractor.load_reports_from_csv()
    
    if df_reports.empty:
        print("⚠️ [Warning] 報告是空的！")
        return

    print("\n🏷️ [Step 3] 建立二元分類標籤 (Cardiomegaly vs Normal)...")
    df_binary = extractor.create_binary_labels(
        df_reports,
        positive_class='Cardiomegaly',
        negative_class='Normal'
    )
    
    print("\n🔍 [Step 4] 驗證標籤 (人工抽樣檢查)...")
    extractor.validate_labels(df_binary, n_samples=5)
    
    print("\n💾 [Step 5] 儲存結果 CSV...")
    extractor.save_labels(df_binary, OUTPUT_DIR / "binary_labels_cardiomegaly.csv")
    
    print("\n" + "="*60)
    print("🎉 所有步驟執行完畢！")
    print(f"✅ 紀錄檔已儲存: {log_filename}")
    print("="*60)

# --- 執行入口 ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 發生未預期的錯誤: {e}")
        import traceback
        traceback.print_exc()