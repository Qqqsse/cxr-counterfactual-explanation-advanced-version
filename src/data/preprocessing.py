"""
資料前處理模組 (v2.0 - 包含平衡機制與自動記錄)
功能：
1. 影像檢查與路徑對應
2. 解決類別不平衡 (Undersampling)
3. 劃分 Train/Val/Test
4. 定義影像增強與標準化流程
"""

import cv2
import sys
import datetime
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------------------------------------------------
# 1. 工具類別：自動記錄器 (跟之前一樣)
# ---------------------------------------------------------
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
            sys.__stderr__.write(message)
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

# ---------------------------------------------------------
# 2. 核心類別：影像前處理器
# ---------------------------------------------------------
class ImagePreprocessor:
    """影像前處理器"""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (512, 512),
                 apply_clahe: bool = True,
                 normalize: bool = True):
        self.target_size = target_size
        self.apply_clahe = apply_clahe
        self.normalize = normalize
        
        # 建立 CLAHE 物件 (對比度限制自適應直方圖均衡化)
        if apply_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def load_image(self, image_path: Path) -> np.ndarray:
        try:
            # 使用 PIL 載入並轉灰階
            img = Image.open(image_path).convert('L')
            img_array = np.array(img)
            return img_array
        except Exception as e:
            raise ValueError(f"無法載入影像 {image_path}: {e}")
    
    def apply_clahe_enhancement(self, image: np.ndarray) -> np.ndarray:
        if not self.apply_clahe:
            return image
        
        if image.dtype != np.uint8:
            image = self._to_uint8(image)
        
        enhanced = self.clahe.apply(image)
        return enhanced
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image, 
                           (self.target_size[1], self.target_size[0]),
                           interpolation=cv2.INTER_AREA)
        return resized
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return image
        
        image = image.astype(np.float32)
        image = image / 255.0 # 歸一化至 [0, 1]
        return image
    
    def preprocess(self, image_path: Path) -> np.ndarray:
        """完整前處理流程"""
        image = self.load_image(image_path)
        image = self.apply_clahe_enhancement(image)
        image = self.resize_image(image)
        image = self.normalize_image(image)
        return image
    
    def _to_uint8(self, image: np.ndarray) -> np.ndarray:
        if image.max() <= 1.0:
            image = image * 255.0
        return image.astype(np.uint8)

# ---------------------------------------------------------
# 3. 核心類別：資料增強器
# ---------------------------------------------------------
class DataAugmentor:
    """資料增強器（用於訓練時增加數據多樣性）"""
    
    def __init__(self, image_size: Tuple[int, int] = (512, 512)):
        self.image_size = image_size
        
        # 訓練增強策略
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5), # 水平翻轉
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5), # 旋轉縮放
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5), # 亮度對比
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.3), # 高斯雜訊
            A.Normalize(mean=(0.5,), std=(0.5,)), # 標準化 (轉成 -1 ~ 1)
            ToTensorV2()
        ])
        
        # 驗證/測試策略 (只做標準化)
        self.val_transform = A.Compose([
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ])

# ---------------------------------------------------------
# 4. 功能函式：平衡、檢查、分割
# ---------------------------------------------------------
def check_image_exists(df_labels, images_dir: Path) -> pd.DataFrame:
    """檢查影像檔案是否存在，並建立完整路徑"""
    print(f"🔍 檢查影像檔案是否存在於: {images_dir}")
    
    # 建立 filename -> full_path 的映射字典，加快搜尋速度
    # 假設所有圖片都在 images_dir 或是其子資料夾內
    all_image_paths = {p.name: p for p in images_dir.glob("**/*.png")}
    
    valid_indices = []
    full_paths = []
    missing_count = 0
    
    for idx, row in df_labels.iterrows():
        img_name = row['image_id']
        
        if img_name in all_image_paths:
            valid_indices.append(idx)
            full_paths.append(str(all_image_paths[img_name]))
        else:
            missing_count += 1
            # print(f"  Missing: {img_name}") # 除錯用
    
    df_filtered = df_labels.loc[valid_indices].reset_index(drop=True)
    df_filtered['image_path'] = full_paths # 新增一欄完整路徑
    
    print(f"✅ 有效影像: {len(df_filtered)}")
    if missing_count > 0:
        print(f"⚠️ 缺失影像: {missing_count} (已過濾)")
    
    return df_filtered

def balance_dataset(df_labels, target_ratio: float = 1.0, random_state: int = 42) -> pd.DataFrame:
    """
    平衡資料集 (欠取樣 Normal 類別)
    
    
    Args:
        target_ratio: 正負樣本比例 (1.0 代表 1:1, 2.0 代表 1:2)
    """
    print("\n⚖️ 執行資料集平衡 (Undersampling)...")
    
    # 分離正負樣本
    pos_df = df_labels[df_labels['label'] == 1]
    neg_df = df_labels[df_labels['label'] == 0]
    
    n_pos = len(pos_df)
    n_neg = len(neg_df)
    
    print(f"  原始分布 - Positive: {n_pos}, Negative: {n_neg}")
    
    # 計算目標負樣本數量 (例如正樣本 200，target_ratio=2 -> 負樣本取 400)
    n_neg_target = int(n_pos * target_ratio)
    
    if n_neg > n_neg_target:
        # 隨機抽取負樣本
        neg_sampled = neg_df.sample(n=n_neg_target, random_state=random_state)
        print(f"  隨機抽取 {n_neg_target} 個 Negative 樣本 (比例 1:{target_ratio})")
    else:
        neg_sampled = neg_df
        print("  Negative 樣本數不足，保留所有樣本")
        
    # 合併並打亂
    df_balanced = pd.concat([pos_df, neg_sampled]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"✅ 平衡後總數: {len(df_balanced)} (Pos: {len(pos_df)}, Neg: {len(neg_sampled)})")
    return df_balanced

def split_dataset(df_labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """劃分資料集（確保同一病人的影像都在同一組）"""
    from sklearn.model_selection import train_test_split
    
    # 提取唯一的 report_id（代表患者）
    unique_reports = df_labels['report_id'].unique()
    
    # 第一次分割：分出訓練集
    train_reports, temp_reports = train_test_split(
        unique_reports, test_size=(1 - train_ratio), random_state=random_state
    )
    
    # 第二次分割：分出驗證集與測試集
    val_size_adjusted = val_ratio / (val_ratio + test_ratio)
    val_reports, test_reports = train_test_split(
        temp_reports, test_size=(1 - val_size_adjusted), random_state=random_state
    )
    
    # 根據 report_id 篩選影像
    train_df = df_labels[df_labels['report_id'].isin(train_reports)].reset_index(drop=True)
    val_df = df_labels[df_labels['report_id'].isin(val_reports)].reset_index(drop=True)
    test_df = df_labels[df_labels['report_id'].isin(test_reports)].reset_index(drop=True)
    
    print(f"\n📊 資料集劃分完成:")
    print(f"  Train: {len(train_df)} ({len(train_reports)} reports)")
    print(f"  Val  : {len(val_df)} ({len(val_reports)} reports)")
    print(f"  Test : {len(test_df)} ({len(test_reports)} reports)")
    
    return train_df, val_df, test_df

# ---------------------------------------------------------
# 5. 主程式
# ---------------------------------------------------------
def main():
    # --- 路徑設定 ---
    try:
        script_path = Path(__file__).resolve()
        project_root = script_path.parents[2]
    except NameError:
        project_root = Path.cwd()
        if (project_root / "src").exists(): project_root = project_root
        elif (project_root.parent / "src").exists(): project_root = project_root.parent

    # 設定 Log
    LOGS_DIR = project_root / "results" / "logs"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sys.stdout = DualLogger(LOGS_DIR / f"preprocessing_log_{timestamp}.txt", sys.stdout)

    print("="*60)
    print(f"🚀 資料前處理程序啟動 - {timestamp}")
    print("="*60)

    # 檔案路徑
    LABELS_PATH = project_root / "data" / "labels" / "binary_labels_cardiomegaly.csv"
    IMAGES_DIR = project_root / "data" / "raw" / "images"
    PROCESSED_DATA_DIR = project_root / "data" / "processed"
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 讀取標籤
    if not LABELS_PATH.exists():
        print(f"❌ 找不到標籤檔: {LABELS_PATH}")
        return
    df = pd.read_csv(LABELS_PATH)
    print(f"📖 讀取標籤檔: {len(df)} 筆資料")

    # 2. 檢查影像存在性
    df = check_image_exists(df, IMAGES_DIR)

    # 3. 資料集平衡 (設定 1:1 或 1:2)
    # 建議設為 1.0 (1:1) 或 2.0 (1:2)，避免正常樣本過多
    df_balanced = balance_dataset(df, target_ratio=1.5, random_state=42)

    # 4. 資料集劃分
    train_df, val_df, test_df = split_dataset(df_balanced)

    # 5. 儲存分割後的 CSV
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    print(f"\n💾 分割檔案已儲存至: {PROCESSED_DATA_DIR}")

    # 6. (選用) 測試前處理器
    print("\n🧪 測試影像前處理器...")
    try:
        sample_path = Path(train_df.iloc[0]['image_path'])
        preprocessor = ImagePreprocessor()
        processed_img = preprocessor.preprocess(sample_path)
        print(f"  測試成功! 影像: {sample_path.name}")
        print(f"  處理後尺寸: {processed_img.shape}, 範圍: {processed_img.min():.2f}~{processed_img.max():.2f}")
    except Exception as e:
        print(f"  測試失敗: {e}")

    print("\n" + "="*60)
    print("🎉 資料前處理完成！請執行 git commit 記錄進度。")
    print("="*60)

if __name__ == "__main__":
    main()