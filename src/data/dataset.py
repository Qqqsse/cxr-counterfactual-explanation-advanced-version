"""
PyTorch Dataset 類別 (v2.0 - 整合前處理與自動記錄)
用於載入 IU X-ray 資料集，並將影像轉換為 Tensor
"""

import sys
import torch
import datetime
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Callable
from PIL import Image

# 嘗試匯入隔壁的 preprocessing 模組
# 這樣做是為了確保即使直接執行這支程式，也能找到 ImagePreprocessor
try:
    sys.path.append(str(Path(__file__).parent))
    from preprocessing import ImagePreprocessor, DataAugmentor
except ImportError:
    print("⚠️ 警告: 無法匯入 preprocessing.py，請確認該檔案位於同一目錄下")
    ImagePreprocessor = None
    DataAugmentor = None

# ---------------------------------------------------------
# 1. 自動記錄器 (標準配備)
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
# 2. 核心類別：CXRDataset
# ---------------------------------------------------------
class CXRDataset(Dataset):
    """胸腔 X 光資料集"""
    
    def __init__(self,
                 df_labels: pd.DataFrame,
                 images_dir: Path,
                 transform: Optional[Callable] = None,
                 preprocessor: Optional[Callable] = None):
        """
        初始化資料集
        Args:
            df_labels: 標籤 DataFrame (需包含 'image_id' 或 'image_path')
            images_dir: 影像根目錄
            transform: 資料增強函數 (Albumentations)
            preprocessor: 基礎前處理器 (Resize, CLAHE)
        """
        self.df_labels = df_labels.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.preprocessor = preprocessor
        
        # 為了加速，如果在前處理階段已經生成了 'image_path' 欄位，就直接用
        # 如果沒有，才動態組裝路徑
        self.use_precomputed_path = 'image_path' in self.df_labels.columns
        
        print(f"✅ 資料集初始化完成: {len(self)} 個樣本")
    
    def _get_image_path(self, row) -> Path:
        """取得影像完整路徑"""
        if self.use_precomputed_path:
            # 如果 CSV 裡已經有完整路徑 (train.csv 通常會有)
            return Path(row['image_path'])
        
        # 否則嘗試搜尋
        image_id = row['image_id']
        possible_paths = [
            self.images_dir / f"{image_id}.png",
            self.images_dir / image_id,
            self.images_dir / f"{image_id}.jpg",
        ]
        for path in possible_paths:
            if path.exists():
                return path
        return possible_paths[0] # 回傳預設值讓後面報錯
    
    def __len__(self) -> int:
        return len(self.df_labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        取得單個樣本
        Returns: (image_tensor, label, image_id)
        """
        # 1. 取得資訊
        row = self.df_labels.iloc[idx]
        image_id = row['image_id']
        label = int(row['label'])
        
        # 2. 載入影像
        img_path = self._get_image_path(row)
        
        try:
            # 先使用 Preprocessor 載入與基礎處理 (Resize, CLAHE)
            if self.preprocessor is not None:
                image = self.preprocessor.preprocess(img_path)
            else:
                # 備用方案: 直接用 PIL 載入
                image = Image.open(img_path).convert('L')
                image = np.array(image)
            
            # 3. 資料增強 (Data Augmentation)
            if self.transform is not None:
                # Albumentations 需要 HWC 格式 (Height, Width, Channel)
                if image.ndim == 2:
                    image = np.expand_dims(image, axis=-1)
                
                # 執行增強 (含 ToTensorV2，會轉成 Tensor CHW)
                augmented = self.transform(image=image)
                image = augmented['image']
                
            else:
                # 如果沒有增強器，手動轉 Tensor
                image = torch.from_numpy(image).float()
                if image.ndim == 2:
                    image = image.unsqueeze(0) # (H, W) -> (1, H, W)
                if image.max() > 1.0:
                    image = image / 255.0      # 簡單歸一化
            
            # 4. 確保通道數 (ImageNet 模型通常需要 3 通道)
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            
            return image, label, image_id
            
        except Exception as e:
            print(f"⚠️ 讀取錯誤 {image_id}: {e}")
            # 回傳全黑影像避免程式崩潰
            return torch.zeros(3, 512, 512), label, image_id

    def get_label_distribution(self):
        """顯示正負樣本分布"""
        counts = self.df_labels['label'].value_counts()
        total = len(self)
        print(f"  Total: {total}")
        print(f"  Normal (0): {counts.get(0, 0)} ({counts.get(0, 0)/total*100:.1f}%)")
        print(f"  Disease(1): {counts.get(1, 0)} ({counts.get(1, 0)/total*100:.1f}%)")

# ---------------------------------------------------------
# 3. Helper: 建立 DataLoader
# ---------------------------------------------------------
def create_dataloaders(train_df, val_df, test_df, images_dir, 
                       batch_size=16, num_workers=0, 
                       preprocessor=None, train_aug=None, val_aug=None):
    
    # 建立 Datasets
    train_dataset = CXRDataset(train_df, images_dir, transform=train_aug, preprocessor=preprocessor)
    val_dataset   = CXRDataset(val_df,   images_dir, transform=val_aug,   preprocessor=preprocessor)
    test_dataset  = CXRDataset(test_df,  images_dir, transform=val_aug,   preprocessor=preprocessor)
    
    # 建立 DataLoaders
    # 注意: Windows 上建議測試時 num_workers=0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader

# ---------------------------------------------------------
# 4. 主測試程式
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

    # Log 設定
    LOGS_DIR = project_root / "results" / "logs"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sys.stdout = DualLogger(LOGS_DIR / f"dataset_test_log_{timestamp}.txt", sys.stdout)

    print("="*60)
    print(f"🚀 Dataset 模組測試啟動 - {timestamp}")
    print("="*60)

    # 讀取剛剛前處理好的 train.csv
    PROCESSED_DIR = project_root / "data" / "processed"
    IMAGES_DIR = project_root / "data" / "raw" / "images"
    TRAIN_CSV = PROCESSED_DIR / "train.csv"
    
    if not TRAIN_CSV.exists():
        print(f"❌ 找不到訓練資料: {TRAIN_CSV}")
        print("   請先執行 preprocessing.py")
        return

    # 1. 載入資料
    print("📖 載入訓練集 CSV...")
    df_train = pd.read_csv(TRAIN_CSV)
    
    # 2. 初始化處理器 (嘗試從 preprocessing.py 載入)
    if ImagePreprocessor is not None:
        preprocessor = ImagePreprocessor(target_size=(512, 512))
        augmentor = DataAugmentor(image_size=(512, 512))
        train_transform = augmentor.train_transform
    else:
        print("⚠️ 使用 Dummy 處理器 (因找不到 preprocessing 模組)")
        preprocessor = None
        train_transform = None

    # 3. 建立 Dataset
    print("\n🛠️ 建立 CXRDataset...")
    dataset = CXRDataset(
        df_labels=df_train.head(100), # 先拿 100 筆測試
        images_dir=IMAGES_DIR,
        transform=train_transform,
        preprocessor=preprocessor
    )
    
    # 4. 顯示分布
    dataset.get_label_distribution()

    # 5. 測試讀取單張
    print("\n🔍 測試讀取單一樣本 (Index 0)...")
    img, label, img_id = dataset[0]
    print(f"  Image Shape: {img.shape} (預期: 3, 512, 512)")
    print(f"  Type: {img.dtype}")
    print(f"  Range: [{img.min():.2f}, {img.max():.2f}]")
    print(f"  Label: {label}")
    print(f"  ID: {img_id}")

    # 6. 測試 DataLoader (Batch 測試)
    print("\n🚚 測試 DataLoader (Batch=8)...")
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0) # Windows 測試用 0
    
    try:
        batch_imgs, batch_labels, batch_ids = next(iter(loader))
        print(f"  Batch Images: {batch_imgs.shape}")
        print(f"  Batch Labels: {batch_labels.shape}")
        print(f"  Labels Content: {batch_labels}")
        print("✅ DataLoader 運作正常！")
    except Exception as e:
        print(f"❌ DataLoader 測試失敗: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("🎉 Dataset 測試完成！請記得 git commit")
    print("="*60)

if __name__ == "__main__":
    main()