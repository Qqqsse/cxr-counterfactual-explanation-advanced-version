"""
PyTorch Dataset 類別
用於載入 IU X-ray 資料集
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Callable
from PIL import Image


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
            df_labels: 標籤 DataFrame，需包含 'image_id' 和 'label'
            images_dir: 影像目錄
            transform: 資料增強函數（albumentations）
            preprocessor: 前處理器（ImagePreprocessor）
        """
        self.df_labels = df_labels.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.preprocessor = preprocessor
        
        # 驗證所有影像都存在
        self._validate_images()
        
        print(f"✅ 資料集初始化完成: {len(self)} 個樣本")
    
    def _validate_images(self):
        """驗證影像檔案存在"""
        missing_images = []
        
        for idx, row in self.df_labels.iterrows():
            img_path = self._get_image_path(row['image_id'])
            if not img_path.exists():
                missing_images.append(row['image_id'])
        
        if missing_images:
            print(f"⚠️ 警告：{len(missing_images)} 個影像檔案不存在")
            # 移除不存在的影像
            self.df_labels = self.df_labels[
                ~self.df_labels['image_id'].isin(missing_images)
            ].reset_index(drop=True)
    
    def _get_image_path(self, image_id: str) -> Path:
        """取得影像完整路徑"""
        # 嘗試多種可能的檔案格式
        possible_paths = [
            self.images_dir / f"{image_id}.png",
            self.images_dir / image_id,
            self.images_dir / f"{image_id}.jpg",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # 如果都不存在，返回第一個（後續會報錯）
        return possible_paths[0]
    
    def __len__(self) -> int:
        """返回資料集大小"""
        return len(self.df_labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        取得單個樣本
        
        Args:
            idx: 索引
        
        Returns:
            (image_tensor, label, image_id)
            - image_tensor: (1, H, W) 或 (3, H, W)
            - label: 0 或 1
            - image_id: 影像 ID
        """
        # 取得標籤資料
        row = self.df_labels.iloc[idx]
        image_id = row['image_id']
        label = int(row['label'])
        
        # 載入影像
        img_path = self._get_image_path(image_id)
        
        try:
            # 載入為灰階
            image = Image.open(img_path).convert('L')
            image = np.array(image)
            
            # 前處理（如果有）
            if self.preprocessor is not None:
                image = self.preprocessor.preprocess(img_path)
            
            # 資料增強（如果有）
            if self.transform is not None:
                # Albumentations 需要 HWC 格式
                if image.ndim == 2:
                    image = np.expand_dims(image, axis=-1)
                
                augmented = self.transform(image=image)
                image = augmented['image']
            else:
                # 手動轉換為 tensor
                image = torch.from_numpy(image).float()
                
                # 確保為 CHW 格式
                if image.ndim == 2:
                    image = image.unsqueeze(0)  # (H, W) -> (1, H, W)
            
            # 如果是單通道，複製為三通道（用於 ImageNet 預訓練模型）
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            
            return image, label, image_id
            
        except Exception as e:
            print(f"⚠️ 無法載入影像 {image_id}: {e}")
            # 返回空白影像作為替代
            image = torch.zeros(3, 512, 512)
            return image, label, image_id
    
    def get_class_weights(self) -> torch.Tensor:
        """
        計算類別權重（用於處理類別不平衡）
        
        Returns:
            類別權重 tensor
        """
        label_counts = self.df_labels['label'].value_counts().sort_index()
        total = len(self.df_labels)
        
        # 計算權重（inverse frequency）
        weights = torch.tensor([
            total / label_counts[0],
            total / label_counts[1]
        ], dtype=torch.float32)
        
        # 歸一化
        weights = weights / weights.sum() * 2
        
        print(f"類別權重: 負例={weights[0]:.3f}, 正例={weights[1]:.3f}")
        
        return weights
    
    def get_label_distribution(self):
        """顯示標籤分布"""
        label_counts = self.df_labels['label'].value_counts()
        print(f"\n標籤分布:")
        print(f"  負例 (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(self)*100:.1f}%)")
        print(f"  正例 (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(self)*100:.1f}%)")


def create_dataloaders(train_df: pd.DataFrame,
                       val_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       images_dir: Path,
                       batch_size: int = 16,
                       num_workers: int = 4,
                       preprocessor = None,
                       train_transform = None,
                       val_transform = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    建立 DataLoader
    
    Args:
        train_df: 訓練集 DataFrame
        val_df: 驗證集 DataFrame
        test_df: 測試集 DataFrame
        images_dir: 影像目錄
        batch_size: batch 大小
        num_workers: 工作執行緒數
        preprocessor: 前處理器
        train_transform: 訓練集增強
        val_transform: 驗證/測試集增強
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # 建立 Dataset
    train_dataset = CXRDataset(
        train_df, images_dir, 
        transform=train_transform, 
        preprocessor=preprocessor
    )
    
    val_dataset = CXRDataset(
        val_df, images_dir, 
        transform=val_transform, 
        preprocessor=preprocessor
    )
    
    test_dataset = CXRDataset(
        test_df, images_dir, 
        transform=val_transform, 
        preprocessor=preprocessor
    )
    
    # 顯示分布
    print("\n訓練集:")
    train_dataset.get_label_distribution()
    print("\n驗證集:")
    val_dataset.get_label_distribution()
    print("\n測試集:")
    test_dataset.get_label_distribution()
    
    # 建立 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # 避免最後一個 batch 太小
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n✅ DataLoader 建立完成")
    print(f"  訓練集: {len(train_loader)} batches")
    print(f"  驗證集: {len(val_loader)} batches")
    print(f"  測試集: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def main():
    """測試腳本"""
    import pandas as pd
    from pathlib import Path
    
    # 設定路徑
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    LABELS_PATH = PROJECT_ROOT / "data" / "labels" / "binary_labels_cardiomegaly.csv"
    IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images"
    
    # 載入標籤
    if LABELS_PATH.exists():
        df_labels = pd.read_csv(LABELS_PATH)
        print(f"載入 {len(df_labels)} 個標籤")
        
        # 建立測試資料集
        dataset = CXRDataset(
            df_labels.head(100),  # 只用前 100 個測試
            IMAGES_DIR
        )
        
        # 測試取得樣本
        image, label, image_id = dataset[0]
        print(f"\n測試樣本:")
        print(f"  Image shape: {image.shape}")
        print(f"  Label: {label}")
        print(f"  Image ID: {image_id}")
        
        # 測試 DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch_images, batch_labels, batch_ids = next(iter(loader))
        
        print(f"\n測試 batch:")
        print(f"  Batch images shape: {batch_images.shape}")
        print(f"  Batch labels shape: {batch_labels.shape}")
        print(f"  Batch labels: {batch_labels.tolist()}")
    else:
        print(f"標籤檔案不存在: {LABELS_PATH}")


if __name__ == "__main__":
    main()