"""
評估指標模組 (v2.0 - 整合自動記錄與 LPIPS/SSIM 優化)
計算分類準確度與反事實影像生成的品質指標
"""

import sys
import torch
import numpy as np
import datetime
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from typing import Tuple, Dict
import lpips
from skimage.metrics import structural_similarity as ssim

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
# 2. 分類指標計算器
# ---------------------------------------------------------
class ClassificationMetrics:
    """分類評估指標"""
    
    @staticmethod
    def compute_all_metrics(y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            y_probs: np.ndarray = None) -> Dict[str, float]:
        """
        計算所有分類指標
        Args:
            y_true: 真實標籤
            y_pred: 預測標籤
            y_probs: 預測機率（用於 AUC）
        """
        metrics = {}
        
        # 基本指標
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # AUC（需要機率）
        if y_probs is not None:
            try:
                # 處理只有一個類別的情況 (避免 roc_auc_score 報錯)
                if len(np.unique(y_true)) > 1:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_probs)
                    metrics['auc_pr'] = average_precision_score(y_true, y_probs)
                else:
                    metrics['auc_roc'] = 0.5 # 無法區分
                    metrics['auc_pr'] = 0.0
            except Exception as e:
                print(f"⚠️ AUC 計算失敗: {e}")
                metrics['auc_roc'] = 0.0
                metrics['auc_pr'] = 0.0
        
        # 混淆矩陣衍生指標
        try:
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics['true_negative'] = int(tn)
                metrics['false_positive'] = int(fp)
                metrics['false_negative'] = int(fn)
                metrics['true_positive'] = int(tp)
                
                # 計算 Specificity (TNR) 和 Sensitivity (TPR, same as Recall)
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        except Exception:
            pass
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float]):
        """格式化列印指標"""
        print("\n" + "="*60)
        print("📊 分類評估指標報告")
        print("="*60)
        
        print(f"\n準確性指標:")
        print(f"  Accuracy:    {metrics.get('accuracy', 0):.4f}")
        print(f"  Precision:   {metrics.get('precision', 0):.4f}")
        print(f"  Recall:      {metrics.get('recall', 0):.4f}")
        print(f"  F1 Score:    {metrics.get('f1', 0):.4f}")
        
        if 'auc_roc' in metrics:
            print(f"\nAUC 指標:")
            print(f"  AUC-ROC:     {metrics.get('auc_roc', 0):.4f}")
            print(f"  AUC-PR:      {metrics.get('auc_pr', 0):.4f}")
        
        if 'specificity' in metrics:
            print(f"\n靈敏度與特異度:")
            print(f"  Sensitivity: {metrics.get('sensitivity', 0):.4f}")
            print(f"  Specificity: {metrics.get('specificity', 0):.4f}")
        
        if 'true_positive' in metrics:
            print(f"\n混淆矩陣:")
            print(f"  [ TP: {metrics.get('true_positive', 0):4d} | FP: {metrics.get('false_positive', 0):4d} ]")
            print(f"  [ FN: {metrics.get('false_negative', 0):4d} | TN: {metrics.get('true_negative', 0):4d} ]")
        
        print("="*60 + "\n")

# ---------------------------------------------------------
# 3. 反事實生成指標 (LPIPS, SSIM)
# ---------------------------------------------------------
class CounterfactualMetrics:
    """反事實影像評估指標"""
    
    def __init__(self, device='cuda'):
        """初始化 LPIPS 模型"""
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"正在載入 LPIPS 模型 (Device: {self.device})...")
        try:
            # 使用 AlexNet 作為骨幹 (計算較快，且符合主流論文標準)
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
            self.lpips_model.eval()
            print("✅ LPIPS 模型載入完成")
        except Exception as e:
            print(f"⚠️ 無法載入 LPIPS 模型: {e}")
            print("   請確認已安裝: pip install lpips")
            self.lpips_model = None
    
    def compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """計算 SSIM (結構相似度) - 相容新舊版 skimage"""
        # 確保在 [0, 1] 範圍
        img1 = np.clip(img1, 0, 1)
        img2 = np.clip(img2, 0, 1)
        
        try:
            # 判斷是否為多通道 (H, W, C)
            is_multichannel = (img1.ndim == 3 and img1.shape[2] > 1)
            
            # 嘗試新版參數 (scikit-image >= 0.19)
            return float(ssim(img1, img2, channel_axis=2 if is_multichannel else None, data_range=1.0))
        except TypeError:
            # 如果失敗，嘗試舊版參數 (scikit-image < 0.19)
            return float(ssim(img1, img2, multichannel=is_multichannel, data_range=1.0))
        except Exception as e:
            print(f"⚠️ SSIM 計算錯誤: {e}")
            return 0.0
    
    def compute_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """計算 LPIPS (感知相似度)"""
        if self.lpips_model is None:
            return -1.0
            
        # 確保形狀為 (B, C, H, W)
        if img1.ndim == 3: img1 = img1.unsqueeze(0)
        if img2.ndim == 3: img2 = img2.unsqueeze(0)
        
        # 移至 GPU 並確保範圍是 [-1, 1] (LPIPS 預期輸入)
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        
        with torch.no_grad():
            lpips_value = self.lpips_model(img1, img2)
        
        return float(lpips_value.item())
    
    def compute_l1_distance(self, img1: np.ndarray, img2: np.ndarray) -> float:
        return float(np.mean(np.abs(img1 - img2)))
    
    def compute_l2_distance(self, img1: np.ndarray, img2: np.ndarray) -> float:
        return float(np.sqrt(np.mean((img1 - img2) ** 2)))
    
    def compute_all_metrics(self, 
                            original: np.ndarray, 
                            counterfactual: np.ndarray,
                            original_tensor: torch.Tensor = None,
                            cf_tensor: torch.Tensor = None) -> Dict[str, float]:
        """計算所有指標"""
        metrics = {}
        
        # 1. 像素級指標 (使用 Numpy)
        metrics['ssim'] = self.compute_ssim(original, counterfactual)
        metrics['l1_distance'] = self.compute_l1_distance(original, counterfactual)
        metrics['l2_distance'] = self.compute_l2_distance(original, counterfactual)
        
        # 2. 感知級指標 (使用 Tensor, 需要 [-1, 1])
        if original_tensor is not None and cf_tensor is not None:
            metrics['lpips'] = self.compute_lpips(original_tensor, cf_tensor)
        
        return metrics

    def print_metrics(self, metrics: Dict[str, float]):
        print("\n" + "="*60)
        print("🖼️ 反事實影像品質指標")
        print("="*60)
        print(f"\n相似度 (越高越好):")
        print(f"  SSIM:       {metrics.get('ssim', 0):.4f} (結構相似度)")
        
        print(f"\n差異度 (越低越好):")
        print(f"  L1 Dist:    {metrics.get('l1_distance', 0):.4f} (像素絕對誤差)")
        print(f"  L2 Dist:    {metrics.get('l2_distance', 0):.4f} (均方根誤差)")
        
        if 'lpips' in metrics:
            print(f"  LPIPS:      {metrics.get('lpips', 0):.4f} (感知差異 - 關鍵指標)")
        print("="*60 + "\n")

# ---------------------------------------------------------
# 4. 差異圖分析工具
# ---------------------------------------------------------
def analyze_difference_map(original: np.ndarray, 
                           counterfactual: np.ndarray,
                           regions: Dict[str, Tuple[int, int, int, int]] = None) -> Dict:
    """分析原始影像與生成的差異分布"""
    # 計算差異圖
    diff = np.abs(original - counterfactual)
    
    analysis = {
        'mean_diff': float(np.mean(diff)),
        'max_diff': float(np.max(diff)),
        'std_diff': float(np.std(diff)),
        # 變化像素佔比 (假設閾值 0.05，即 5% 的變化量才算變動)
        'changed_pixels_pct': float(np.mean(diff > 0.05) * 100)
    }
    
    # 前景 vs 背景 分析 (假設邊緣 10% 為背景)
    h, w = original.shape[:2]
    margin = int(min(h, w) * 0.1)
    
    mask_fg = np.zeros_like(diff, dtype=bool)
    mask_fg[margin:-margin, margin:-margin] = True
    
    diff_fg = diff[mask_fg]
    diff_bg = diff[~mask_fg]
    
    analysis['foreground_mean'] = float(np.mean(diff_fg))
    analysis['background_mean'] = float(np.mean(diff_bg))
    # 聚焦度: 前景變化量 / 背景變化量 (越高代表修改越精準)
    analysis['focus_ratio'] = float(analysis['foreground_mean'] / (analysis['background_mean'] + 1e-8))
    
    return analysis

# ---------------------------------------------------------
# 5. 主測試程式
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
    sys.stdout = DualLogger(LOGS_DIR / f"metrics_test_log_{timestamp}.txt", sys.stdout)

    print("="*60)
    print(f"🚀 Metrics 評估模組測試啟動 - {timestamp}")
    print("="*60)

    # 1. 測試分類指標
    print("\n[Test 1] 分類指標測試...")
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    y_probs = np.array([0.1, 0.6, 0.8, 0.9, 0.2, 0.4, 0.3, 0.85])
    
    clf_metrics = ClassificationMetrics()
    metrics = clf_metrics.compute_all_metrics(y_true, y_pred, y_probs)
    clf_metrics.print_metrics(metrics)
    
    # 2. 測試反事實指標 (LPIPS, SSIM)
    print("\n[Test 2] 反事實影像指標測試...")
    # 模擬一張 256x256 的雜訊圖
    original_np = np.random.rand(256, 256).astype(np.float32)
    # 模擬稍微修改後的圖 (加一點雜訊)
    cf_np = np.clip(original_np + np.random.randn(256, 256) * 0.1, 0, 1).astype(np.float32)
    
    # 轉成 Tensor (LPIPS 需要) - 範圍要轉成 [-1, 1]
    orig_tensor = torch.from_numpy(original_np).unsqueeze(0).unsqueeze(0) * 2 - 1
    cf_tensor = torch.from_numpy(cf_np).unsqueeze(0).unsqueeze(0) * 2 - 1
    
    cf_evaluator = CounterfactualMetrics() # 會自動偵測 GPU
    
    try:
        cf_metrics_dict = cf_evaluator.compute_all_metrics(
            original_np, cf_np, orig_tensor, cf_tensor
        )
        cf_evaluator.print_metrics(cf_metrics_dict)
    except Exception as e:
        print(f"❌ 評估失敗: {e}")
        import traceback
        traceback.print_exc()

    # 3. 測試差異分析
    print("\n[Test 3] 差異圖分析測試...")
    diff_analysis = analyze_difference_map(original_np, cf_np)
    print(f"  全局平均差異: {diff_analysis['mean_diff']:.4f}")
    print(f"  變化像素佔比: {diff_analysis['changed_pixels_pct']:.2f}%")
    print(f"  聚焦度 (FG/BG): {diff_analysis['focus_ratio']:.2f} (大於 1 代表主要改在中間)")

    print("\n" + "="*60)
    print("🎉 Metrics 模組測試完成！")
    print("="*60)

if __name__ == "__main__":
    main()