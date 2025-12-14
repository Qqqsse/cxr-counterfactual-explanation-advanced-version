"""
視覺化工具模組 (v2.0 - 整合中文支援與自動記錄)
包含各種繪圖函數，支援 GAN 訓練曲線與反事實對比圖
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict
import torch
from pathlib import Path
import datetime
import matplotlib.font_manager as fm

# ---------------------------------------------------------
# 1. 自動記錄器
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
# 2. 中文顯示修正
# ---------------------------------------------------------
def set_chinese_font():
    """嘗試設定中文字型以避免亂碼"""
    # 常見的中文字型列表
    font_names = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'Heiti TC']
    
    # 檢查系統中有哪個字型可用
    found_font = None
    for name in font_names:
        if name in [f.name for f in fm.fontManager.ttflist]:
            found_font = name
            break
            
    if found_font:
        plt.rcParams['font.sans-serif'] = [found_font] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False # 讓負號正常顯示
        return True
    return False

# 嘗試設定一次
set_chinese_font()

# ---------------------------------------------------------
# 3. 繪圖函式庫
# ---------------------------------------------------------

def plot_images_grid(images: List[np.ndarray],
                     titles: List[str] = None,
                     n_cols: int = 4,
                     figsize: Tuple[int, int] = None,
                     save_path: Optional[Path] = None,
                     cmap: str = 'gray') -> None:
    """繪製影像網格"""
    n_images = len(images)
    n_rows = int(np.ceil(n_images / n_cols))
    
    if figsize is None:
        figsize = (n_cols * 4, n_rows * 4)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_images == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, img in enumerate(images):
        # 確保影像在 [0, 1] 範圍
        img = np.clip(img, 0, 1)
        
        # 如果是 3D 且最後維度為 1，移除
        if img.ndim == 3 and img.shape[2] == 1:
            img = img.squeeze(-1)
        
        axes[idx].imshow(img, cmap=cmap)
        
        if titles and idx < len(titles):
            axes[idx].set_title(titles[idx], fontsize=10)
        
        axes[idx].axis('off')
    
    # 隱藏多餘的子圖
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        # 確保目錄存在
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 圖片已儲存: {save_path}")
    
    plt.show()


def plot_counterfactual_comparison(original: np.ndarray,
                                   counterfactual: np.ndarray,
                                   difference: np.ndarray,
                                   original_label: str,
                                   cf_label: str,
                                   save_path: Optional[Path] = None) -> None:
    """繪製反事實影像對比 (原圖 vs 反事實 vs 差異圖)"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始影像
    axes[0].imshow(original, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'Original\n({original_label})', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 反事實影像
    axes[1].imshow(counterfactual, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Counterfactual\n({cf_label})', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # 差異圖
    # 使用 jet 或 hot 來強調差異，並固定 scale 以便比較
    im = axes[2].imshow(difference, cmap='hot', vmin=0, vmax=np.max(difference) + 1e-8)
    axes[2].set_title('Difference Map', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 圖片已儲存: {save_path}")
    
    plt.show()


def plot_difference_heatmap(original: np.ndarray,
                            counterfactual: np.ndarray,
                            overlay_alpha: float = 0.5,
                            save_path: Optional[Path] = None) -> None:
    """繪製差異熱圖疊加在原圖上"""
    # 計算差異
    diff = np.abs(original - counterfactual)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左：差異熱圖
    im1 = axes[0].imshow(diff, cmap='hot')
    axes[0].set_title('Difference Heatmap', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 右：疊加在原圖上
    axes[1].imshow(original, cmap='gray')
    im2 = axes[1].imshow(diff, cmap='hot', alpha=overlay_alpha)
    axes[1].set_title('Overlay on Original', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 圖片已儲存: {save_path}")
    
    plt.show()


def plot_gradcam_comparison(original: np.ndarray,
                            counterfactual: np.ndarray,
                            original_cam: np.ndarray,
                            cf_cam: np.ndarray,
                            save_path: Optional[Path] = None) -> None:
    """繪製 Grad-CAM 對比"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # 原始影像
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 原始影像 + CAM
    axes[0, 1].imshow(original, cmap='gray')
    im1 = axes[0, 1].imshow(original_cam, cmap='jet', alpha=0.5)
    axes[0, 1].set_title('Original Grad-CAM', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # 反事實影像
    axes[1, 0].imshow(counterfactual, cmap='gray')
    axes[1, 0].set_title('Counterfactual Image', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 反事實影像 + CAM
    axes[1, 1].imshow(counterfactual, cmap='gray')
    im2 = axes[1, 1].imshow(cf_cam, cmap='jet', alpha=0.5)
    axes[1, 1].set_title('Counterfactual Grad-CAM', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 圖片已儲存: {save_path}")
    
    plt.show()


def plot_training_curves(history: dict,
                         metrics: List[str] = ['loss', 'acc'],
                         save_path: Optional[Path] = None) -> None:
    """繪製訓練曲線"""
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    # 嘗試抓取 epochs
    if f'train_{metrics[0]}' in history:
        epochs = range(1, len(history[f'train_{metrics[0]}']) + 1)
    else:
        # Fallback for GAN history format
        epochs = range(1, len(next(iter(history.values()))) + 1)
    
    for idx, metric in enumerate(metrics):
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        
        # 檢查是否存在這些 key
        if train_key in history:
            axes[idx].plot(epochs, history[train_key], 'b-', label=f'Train {metric}', linewidth=2)
        if val_key in history:
            axes[idx].plot(epochs, history[val_key], 'r-', label=f'Val {metric}', linewidth=2)
            
        axes[idx].set_xlabel('Epoch', fontsize=12)
        axes[idx].set_ylabel(metric.capitalize(), fontsize=12)
        axes[idx].set_title(f'{metric.capitalize()} Curve', fontsize=14, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 圖片已儲存: {save_path}")
    
    plt.show()


def plot_gan_losses(g_losses: List[float],
                    d_losses: List[float],
                    save_path: Optional[Path] = None) -> None:
    """繪製 GAN 訓練損失"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    iterations = range(1, len(g_losses) + 1)
    
    ax.plot(iterations, g_losses, 'b-', label='Generator Loss', linewidth=1.5, alpha=0.7)
    ax.plot(iterations, d_losses, 'r-', label='Discriminator Loss', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('GAN Training Loss', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 圖片已儲存: {save_path}")
    
    plt.show()


def plot_metrics_comparison(metrics_dict: dict,
                            save_path: Optional[Path] = None) -> None:
    """繪製多個指標的比較 (Bar Chart)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    bars = ax.barh(names, values, color='steelblue')
    
    # 加上數值標籤
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}',
                ha='left', va='center', fontsize=10, padding=3)
    
    ax.set_xlabel('Value', fontsize=12)
    ax.set_title('Metrics Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 圖片已儲存: {save_path}")
    
    plt.show()

# ---------------------------------------------------------
# 4. 主測試程式
# ---------------------------------------------------------
def main():
    # --- Log 設定 ---
    try:
        script_path = Path(__file__).resolve()
        project_root = script_path.parents[2]
    except NameError:
        project_root = Path.cwd()
        
    LOGS_DIR = project_root / "results" / "logs"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sys.stdout = DualLogger(LOGS_DIR / f"plotting_test_log_{timestamp}.txt", sys.stdout)

    print("="*60)
    print(f"🚀 Visualization 模組測試啟動 - {timestamp}")
    print("="*60)
    
    # 測試圖片輸出路徑
    TEST_FIG_DIR = project_root / "results" / "figures" / "test_plots"
    TEST_FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Test 1] 測試影像網格 (Image Grid)...")
    images = [np.random.rand(256, 256) for _ in range(6)]
    titles = [f'Test Img {i+1}' for i in range(6)]
    plot_images_grid(images, titles, n_cols=3, save_path=TEST_FIG_DIR / "test_grid.png")
    
    print("\n[Test 2] 測試反事實對比 (Counterfactual Comparison)...")
    original = np.random.rand(256, 256)
    counterfactual = original + np.random.randn(256, 256) * 0.1
    counterfactual = np.clip(counterfactual, 0, 1)
    difference = np.abs(original - counterfactual)
    
    plot_counterfactual_comparison(
        original, counterfactual, difference,
        'Normal', 'Cardiomegaly',
        save_path=TEST_FIG_DIR / "test_comparison.png"
    )
    
    print("\n[Test 3] 測試 GAN Loss 曲線...")
    g_losses = [2.0 - i*0.01 + np.random.rand()*0.1 for i in range(100)]
    d_losses = [0.5 + i*0.005 + np.random.rand()*0.1 for i in range(100)]
    plot_gan_losses(g_losses, d_losses, save_path=TEST_FIG_DIR / "test_gan_loss.png")

    print("\n" + "="*60)
    print(f"🎉 Visualization 模組測試完成！")
    print(f"請檢查資料夾: {TEST_FIG_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()