"""
損失函數模組 (v2.0 - 整合 VGG 權重修正與輸入正規化)
包含 GAN 損失、分類損失、感知損失 (Perceptual Loss) 等
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights
import datetime
from pathlib import Path

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
# 2. GAN 損失 (對抗損失)
# ---------------------------------------------------------
class GANLoss(nn.Module):
    """
    GAN 損失 (支援 Vanilla BCE, LSGAN MSE, WGAN)
    """
    def __init__(self, gan_mode: str = 'vanilla'):
        super(GANLoss, self).__init__()
        self.gan_mode = gan_mode
        
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'wgan':
            self.loss = None
        else:
            raise ValueError(f"不支援的 GAN 模式: {gan_mode}")
        
        print(f"✅ GAN 損失初始化: {gan_mode}")
    
    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """
        Args:
            prediction: (B, 1, H, W) 判別器輸出
            target_is_real: True=真實標籤, False=假標籤
        """
        if self.gan_mode == 'wgan':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        else:
            # 建立全 1 或全 0 的目標張量
            target_tensor = torch.tensor(1.0 if target_is_real else 0.0).to(prediction.device)
            target = target_tensor.expand_as(prediction)
            
            loss = self.loss(prediction, target)
        
        return loss

# ---------------------------------------------------------
# 3. 感知損失 (Perceptual Loss) - 使用 VGG16
# ---------------------------------------------------------
class PerceptualLoss(nn.Module):
    """
    感知損失：比較生成圖與原圖在 VGG16 特徵空間中的差異
    這能讓生成的紋理更自然，避免 L1 Loss 造成的模糊感。
    """
    def __init__(self, layer_weights: dict = None, device: str = 'cuda'):
        super(PerceptualLoss, self).__init__()
        
        # 預設層權重 (淺層捕捉紋理，深層捕捉結構)
        if layer_weights is None:
            layer_weights = {
                'relu1_2': 1.0,
                'relu2_2': 1.0,
                'relu3_3': 1.0,
                'relu4_3': 1.0
            }
        self.layer_weights = layer_weights
        
        # [修正] 使用新版 weights 參數載入 VGG16
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT).features.to(device).eval()
        
        # 凍結參數
        for param in vgg.parameters():
            param.requires_grad = False
        
        # 切分 VGG 模型以提取特定層的輸出
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])   # -> relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])  # -> relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16]) # -> relu3_3
        self.slice4 = nn.Sequential(*list(vgg.children())[16:23])# -> relu4_3
        
        self.loss_fn = nn.L1Loss()
        
        # ImageNet 正規化參數 (用於預處理輸入)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
        print(f"✅ 感知損失初始化完成 (VGG16)")
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x, y: 輸入影像 (B, 3, H, W)，範圍 [-1, 1]
        """
        # 1. 確保 3 通道 (如果是灰階圖，複製成 3 層)
        if x.size(1) == 1: x = x.repeat(1, 3, 1, 1)
        if y.size(1) == 1: y = y.repeat(1, 3, 1, 1)
        
        # 2. 正規化到 VGG 喜歡的範圍
        # 先從 [-1, 1] 轉到 [0, 1]
        x = (x + 1) / 2
        y = (y + 1) / 2
        # 再做 ImageNet Normalize
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        
        # 3. 計算各層特徵損失
        loss = 0.0
        
        x_relu1_2 = self.slice1(x)
        y_relu1_2 = self.slice1(y)
        loss += self.layer_weights.get('relu1_2', 0) * self.loss_fn(x_relu1_2, y_relu1_2)
        
        x_relu2_2 = self.slice2(x_relu1_2)
        y_relu2_2 = self.slice2(y_relu1_2)
        loss += self.layer_weights.get('relu2_2', 0) * self.loss_fn(x_relu2_2, y_relu2_2)
        
        x_relu3_3 = self.slice3(x_relu2_2)
        y_relu3_3 = self.slice3(y_relu2_2)
        loss += self.layer_weights.get('relu3_3', 0) * self.loss_fn(x_relu3_3, y_relu3_3)
        
        x_relu4_3 = self.slice4(x_relu3_3)
        y_relu4_3 = self.slice4(y_relu3_3)
        loss += self.layer_weights.get('relu4_3', 0) * self.loss_fn(x_relu4_3, y_relu4_3)
        
        return loss

# ---------------------------------------------------------
# 4. 分類損失 (Classification Loss)
# ---------------------------------------------------------
class ClassificationLoss(nn.Module):
    """
    確保生成的反事實影像能被分類器判定為目標類別
    (例如：把 Normal 改成 Cardiomegaly，分類器必須說它是 Cardiomegaly)
    """
    def __init__(self, classifier: nn.Module):
        super(ClassificationLoss, self).__init__()
        self.classifier = classifier
        self.classifier.eval() # 永遠保持評估模式，不要更新它
        
        # 凍結分類器參數 (我們只訓練 Generator，不訓練 Classifier)
        for param in self.classifier.parameters():
            param.requires_grad = False
        
        self.loss_fn = nn.CrossEntropyLoss()
        print(f"✅ 分類損失初始化完成")
    
    def forward(self, generated: torch.Tensor, target_label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            generated: (B, 3, H, W) 生成圖
            target_label: (B,) 目標類別索引 (LongTensor)
        """
        # 確保不需要計算 Classifier 的梯度
        with torch.no_grad():
            self.classifier.eval()
            
        # 計算 Logits (因為需要反向傳播給 Generator，這裡不需要 no_grad)
        # 注意: 我們需要 Generator 的梯度，但不需要 Classifier 的梯度
        # 但因為 Classifier 已經設為 requires_grad=False，PyTorch 會自動處理
        logits = self.classifier(generated)
        
        loss = self.loss_fn(logits, target_label)
        return loss

# ---------------------------------------------------------
# 5. 全變分損失 (TV Loss)
# ---------------------------------------------------------
class TotalVariationLoss(nn.Module):
    """鼓勵生成的影像平滑，減少噪點"""
    def __init__(self):
        super(TotalVariationLoss, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        
        count_h = x.size(1) * x.size(2) * x.size(3)
        
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2).sum()
        
        return (h_tv + w_tv) / count_h

# ---------------------------------------------------------
# 6. CounterfactualLoss (總損失包裝器)
# ---------------------------------------------------------
class CounterfactualLoss(nn.Module):
    """整合所有損失函數的 Wrapper"""
    
    def __init__(self,
                 classifier: nn.Module,
                 lambda_adv: float = 1.0,
                 lambda_cls: float = 10.0, # 分類權重通常要大，確保改圖有效
                 lambda_l1: float = 100.0, # L1 權重最大，確保不改背景
                 lambda_perceptual: float = 10.0,
                 lambda_tv: float = 0.1,
                 device: str = 'cuda'):
        
        super(CounterfactualLoss, self).__init__()
        
        self.weights = {
            'adv': lambda_adv,
            'cls': lambda_cls,
            'l1': lambda_l1,
            'perceptual': lambda_perceptual,
            'tv': lambda_tv
        }
        
        self.gan_loss = GANLoss(gan_mode='lsgan') # LSGAN 通常比 Vanilla 穩定
        self.cls_loss = ClassificationLoss(classifier)
        self.perceptual_loss = PerceptualLoss(device=device)
        self.tv_loss = TotalVariationLoss()
        self.l1_loss = nn.L1Loss()
        
        print(f"✅ 反事實總損失初始化完成 (Weights: {self.weights})")
    
    def forward(self,
                generated: torch.Tensor,
                original: torch.Tensor,
                target_label: torch.Tensor,
                disc_output: torch.Tensor = None) -> dict:
        
        losses = {}
        
        # 1. GAN Loss (騙過判別器)
        if disc_output is not None:
            losses['adv'] = self.weights['adv'] * self.gan_loss(disc_output, target_is_real=True)
        else:
            losses['adv'] = torch.tensor(0.0).to(generated.device)
            
        # 2. Classification Loss (騙過分類器)
        losses['cls'] = self.weights['cls'] * self.cls_loss(generated, target_label)
        
        # 3. L1 Loss (像素相似度)
        losses['l1'] = self.weights['l1'] * self.l1_loss(generated, original)
        
        # 4. Perceptual Loss (紋理相似度)
        losses['perceptual'] = self.weights['perceptual'] * self.perceptual_loss(generated, original)
        
        # 5. TV Loss (平滑度)
        losses['tv'] = self.weights['tv'] * self.tv_loss(generated)
        
        # 計算總合
        losses['total'] = sum(losses.values())
        
        return losses

# ---------------------------------------------------------
# 7. 主測試程式 (修正版：加入 requires_grad)
# ---------------------------------------------------------
def main():
    try:
        script_path = Path(__file__).resolve()
        project_root = script_path.parents[2]
    except NameError:
        project_root = Path.cwd()
        
    LOGS_DIR = project_root / "results" / "logs"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sys.stdout = DualLogger(LOGS_DIR / f"losses_test_log_{timestamp}.txt", sys.stdout)

    print("="*60)
    print(f"🚀 Losses 模組測試啟動 - {timestamp}")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"測試裝置: {device}")
    
    # 建立假分類器 (ResNet18)
    print("\n[Init] 建立假分類器...")
    dummy_classifier = models.resnet18(weights=None)
    dummy_classifier.fc = nn.Linear(512, 2) # 2 classes
    dummy_classifier.to(device)
    
    # 初始化總損失
    print("\n[Init] 初始化 CounterfactualLoss...")
    try:
        criterion = CounterfactualLoss(classifier=dummy_classifier, device=device)
    except Exception as e:
        print(f"❌ 初始化失敗: {e}")
        return

    # 模擬資料
    B = 2
    # [修正點 1] 設定 requires_grad=True，模擬這是由 Generator 生成的圖片
    generated = torch.randn(B, 3, 256, 256).to(device)
    generated.requires_grad_(True) 

    original = torch.randn(B, 3, 256, 256).to(device)  # 原圖不需要梯度
    target_label = torch.tensor([1, 0]).long().to(device) # 目標標籤
    
    # [修正點 2] 設定 requires_grad=True，模擬這是由 Discriminator 輸出的結果
    disc_output = torch.randn(B, 1, 30, 30).to(device) 
    disc_output.requires_grad_(True)
    
    # 計算損失
    print("\n[Test] 計算 Loss...")
    loss_dict = criterion(generated, original, target_label, disc_output)
    
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")
        
    print("\n✅ Backward Pass 測試 (檢查梯度)...")
    try:
        # 清空之前的梯度 (如果是真實訓練迴圈需要這步，這裡非必須但好習慣)
        if generated.grad is not None: generated.grad.zero_()
        
        # 反向傳播
        loss_dict['total'].backward()
        
        # 檢查 generated 是否真的有收到梯度
        if generated.grad is not None:
            print(f"  梯度反向傳播成功！Generated Grad Mean: {generated.grad.abs().mean().item():.6f}")
        else:
            print("  ⚠️ 警告: Generated 沒有收到梯度！")
            
    except Exception as e:
        print(f"❌ 反向傳播失敗: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("🎉 Losses 模組測試完成！")
    print("="*60)

if __name__ == "__main__":
    main()