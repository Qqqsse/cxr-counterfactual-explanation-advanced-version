"""
資料前處理模組
包含影像載入、CLAHE 增強、Resize、歸一化等功能
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ImagePreprocessor:
    """影像前處理器"""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (512, 512),
                 apply_clahe: bool = True,
                 normalize: bool = True):
        """
        初始化前處理器
        
        Args:
            target_size: 目標影像大小 (H, W)
            apply_clahe: 是否應用 CLAHE
            normalize: 是否歸一化至 [0, 1]
        """
        self.target_size = target_size
        self.apply_clahe = apply_clahe
        self.normalize = normalize
        
        # 建立 CLAHE 物件
        if apply_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def load_image(self, image_path: Path) -> np.ndarray:
        """
        載入影像
        
        Args:
            image_path: 影像路徑
        
        Returns:
            灰階影像陣列 (H, W)
        """
        try:
            # 使用 PIL 載入
            img = Image.open(image_path).convert('L')
            img_array = np.array(img)
            return img_array
        except Exception as e:
            raise ValueError(f"無法載入影像 {image_path}: {e}")
    
    def apply_clahe_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        應用 CLAHE 對比度增強
        
        Args:
            image: 輸入影像 (H, W)
        
        Returns:
            增強後的影像 (H, W)
        """
        if not self.apply_clahe:
            return image
        
        # 確保為 uint8 格式
        if image.dtype != np.uint8:
            image = self._to_uint8(image)
        
        enhanced = self.clahe.apply(image)
        return enhanced
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        調整影像大小
        
        Args:
            image: 輸入影像 (H, W)
        
        Returns:
            調整後的影像 (target_H, target_W)
        """
        resized = cv2.resize(image, 
                           (self.target_size[1], self.target_size[0]),
                           interpolation=cv2.INTER_AREA)
        return resized
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        歸一化影像至 [0, 1]
        
        Args:
            image: 輸入影像 (H, W)
        
        Returns:
            歸一化後的影像 (H, W)
        """
        if not self.normalize:
            return image
        
        # 轉換為 float32
        image = image.astype(np.float32)
        
        # 歸一化至 [0, 1]
        image = image / 255.0
        
        return image
    
    def preprocess(self, image_path: Path) -> np.ndarray:
        """
        完整前處理流程
        
        Args:
            image_path: 影像路徑
        
        Returns:
            前處理後的影像 (H, W) 或 (H, W, 1)
        """
        # 1. 載入影像
        image = self.load_image(image_path)
        
        # 2. CLAHE 增強
        image = self.apply_clahe_enhancement(image)
        
        # 3. Resize
        image = self.resize_image(image)
        
        # 4. 歸一化
        image = self.normalize_image(image)
        
        return image
    
    def _to_uint8(self, image: np.ndarray) -> np.ndarray:
        """轉換為 uint8 格式"""
        if image.max() <= 1.0:
            image = image * 255.0
        return image.astype(np.uint8)


class DataAugmentor:
    """資料增強器（僅用於訓練集）"""
    
    def __init__(self, image_size: Tuple[int, int] = (512, 512)):
        """
        初始化增強器
        
        Args:
            image_size: 影像大小 (H, W)
        """
        self.image_size = image_size
        
        # 定義訓練時的增強
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ])
        
        # 定義驗證/測試時的增強（無隨機性）
        self.val_transform = A.Compose([
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ])
    
    def apply_train_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        應用訓練時增強
        
        Args:
            image: 輸入影像 (H, W) 或 (H, W, 1)
        
        Returns:
            增強後的影像 tensor (1, H, W)
        """
        # 確保為 3D 陣列
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        
        # 應用增強
        augmented = self.train_transform(image=image)
        return augmented['image']
    
    def apply_val_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        應用驗證/測試時增強
        
        Args:
            image: 輸入影像 (H, W) 或 (H, W, 1)
        
        Returns:
            增強後的影像 tensor (1, H, W)
        """
        # 確保為 3D 陣列
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        
        # 應用增強
        augmented = self.val_transform(image=image)
        return augmented['image']


def split_dataset(df_labels, 
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_state: int = 42) -> Tuple:
    """
    劃分資料集（患者層級）
    
    Args:
        df_labels: 標籤 DataFrame（需包含 'image_id' 和 'report_id'）
        train_ratio: 訓練集比例
        val_ratio: 驗證集比例
        test_ratio: 測試集比例
        random_state: 隨機種子
    
    Returns:
        (train_df, val_df, test_df) 三個 DataFrame
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    # 驗證比例
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "比例總和必須為 1.0"
    
    # 提取唯一的 report_id（代表患者）
    unique_reports = df_labels['report_id'].unique()
    
    print(f"總共 {len(unique_reports)} 個唯一報告（患者）")
    print(f"總共 {len(df_labels)} 張影像")
    
    # 第一次分割：分出訓練集
    train_reports, temp_reports = train_test_split(
        unique_reports,
        test_size=(1 - train_ratio),
        random_state=random_state
    )
    
    # 第二次分割：分出驗證集與測試集
    val_size = val_ratio / (val_ratio + test_ratio)
    val_reports, test_reports = train_test_split(
        temp_reports,
        test_size=(1 - val_size),
        random_state=random_state
    )
    
    # 根據 report_id 篩選影像
    train_df = df_labels[df_labels['report_id'].isin(train_reports)].reset_index(drop=True)
    val_df = df_labels[df_labels['report_id'].isin(val_reports)].reset_index(drop=True)
    test_df = df_labels[df_labels['report_id'].isin(test_reports)].reset_index(drop=True)
    
    # 統計
    print(f"\n資料集劃分結果：")
    print(f"  訓練集: {len(train_df)} 張影像 ({len(train_reports)} 個報告)")
    print(f"  驗證集: {len(val_df)} 張影像 ({len(val_reports)} 個報告)")
    print(f"  測試集: {len(test_df)} 張影像 ({len(test_reports)} 個報告)")
    
    # 檢查類別分布
    print(f"\n類別分布：")
    for split_name, split_df in [('訓練集', train_df), ('驗證集', val_df), ('測試集', test_df)]:
        pos_count = (split_df['label'] == 1).sum()
        neg_count = (split_df['label'] == 0).sum()
        print(f"  {split_name}: 正例 {pos_count} ({pos_count/len(split_df)*100:.1f}%), "
              f"負例 {neg_count} ({neg_count/len(split_df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def check_image_exists(df_labels, images_dir: Path) -> pd.DataFrame:
    """
    檢查影像檔案是否存在，過濾掉不存在的樣本
    
    Args:
        df_labels: 標籤 DataFrame
        images_dir: 影像目錄
    
    Returns:
        過濾後的 DataFrame
    """
    print(f"檢查影像檔案是否存在...")
    
    valid_indices = []
    missing_count = 0
    
    for idx, row in df_labels.iterrows():
        # 嘗試多種可能的檔案路徑格式
        possible_paths = [
            images_dir / f"{row['image_id']}.png",
            images_dir / row['image_id'],
            images_dir / f"{row['image_id']}.jpg",
        ]
        
        exists = any(p.exists() for p in possible_paths)
        
        if exists:
            valid_indices.append(idx)
        else:
            missing_count += 1
    
    df_filtered = df_labels.loc[valid_indices].reset_index(drop=True)
    
    print(f"✅ 有效影像: {len(df_filtered)}")
    print(f"⚠️ 缺失影像: {missing_count}")
    
    return df_filtered


def main():
    """測試腳本"""
    from pathlib import Path
    
    # 設定路徑
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    IMAGE_PATH = PROJECT_ROOT / "data" / "raw" / "images" / "sample.png"
    
    # 建立前處理器
    preprocessor = ImagePreprocessor(
        target_size=(512, 512),
        apply_clahe=True,
        normalize=True
    )
    
    # 測試前處理
    if IMAGE_PATH.exists():
        processed = preprocessor.preprocess(IMAGE_PATH)
        print(f"前處理完成: {processed.shape}, dtype: {processed.dtype}")
        print(f"像素值範圍: [{processed.min():.3f}, {processed.max():.3f}]")
    else:
        print(f"測試影像不存在: {IMAGE_PATH}")


if __name__ == "__main__":
    main()