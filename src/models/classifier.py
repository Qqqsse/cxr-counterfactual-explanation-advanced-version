"""
ResNet18 分類器模組 (v2.1 - 修復參數名稱錯誤與消除警告)
用於胸腔 X 光影像二元分類 (Normal vs Cardiomegaly)
"""

import sys
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights # 新增這行以消除警告
import datetime
from pathlib import Path
from typing import Optional

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
# 2. 核心模型：CXRClassifier
# ---------------------------------------------------------
class CXRClassifier(nn.Module):
    """基於 ResNet18 的胸腔 X 光分類器"""
    
    def __init__(self, 
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout_rate: float = 0.3):
        """
        初始化分類器
        """
        super(CXRClassifier, self).__init__()
        
        # --- [修正] 使用新版 weights 參數來消除 UserWarning ---
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None
            
        self.backbone = models.resnet18(weights=weights)
        
        # 取得特徵維度 (ResNet18 的 fc 輸入通常是 512)
        num_features = self.backbone.fc.in_features
        
        # 替換全連接層 (Classifier Head)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
        
        print(f"✅ 分類器初始化完成")
        print(f"   - 骨幹網路: ResNet18")
        print(f"   - 預訓練權重: {'IMAGENET1K_V1' if pretrained else 'None'}")
        print(f"   - 特徵維度: {num_features}")
        print(f"   - 類別數: {num_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特徵向量 (跳過最後一層)"""
        modules = list(self.backbone.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
        
        features = feature_extractor(x)
        features = torch.flatten(features, 1)
        return features
    
    def freeze_backbone(self):
        """凍結骨幹網路"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
        print("✅ 骨幹網路已凍結")
    
    def unfreeze_backbone(self):
        """解凍所有參數"""
        for param in self.parameters():
            param.requires_grad = True
        print("✅ 所有層已解凍")
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ---------------------------------------------------------
# 3. 輔助工具：早停、優化器、Checkpoint
# ---------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.monitor_op = lambda x, y: x < (y - min_delta)
        else:
            self.monitor_op = lambda x, y: x > (y + min_delta)
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.monitor_op(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n⚠️ 早停觸發！已連續 {self.patience} 個 epoch 無改善")
                return True
        return False

def create_optimizer(model: nn.Module, optimizer_name: str = 'adam', learning_rate: float = 1e-4, weight_decay: float = 1e-5) -> torch.optim.Optimizer:
    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支援的優化器: {optimizer_name}")
    
    print(f"✅ 優化器建立: {optimizer_name.upper()}, LR: {learning_rate}, WD: {weight_decay}")
    return optimizer

def create_scheduler(optimizer: torch.optim.Optimizer, scheduler_name: str = 'reduce_on_plateau', **kwargs):
    if scheduler_name.lower() == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True, **kwargs
        )
    elif scheduler_name.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, **kwargs)
    elif scheduler_name.lower() == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, **kwargs)
    else:
        raise ValueError(f"不支援的調度器: {scheduler_name}")
    
    print(f"✅ 學習率調度器: {scheduler_name}")
    return scheduler

def save_checkpoint(model, optimizer, epoch, val_loss, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }
    torch.save(checkpoint, save_path)
    print(f"💾 模型存檔: {save_path}")

def load_checkpoint(model, optimizer, checkpoint_path, device):
    print(f"📖 載入檢查點: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    val_loss = checkpoint.get('val_loss', float('inf'))
    return epoch, val_loss

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
    sys.stdout = DualLogger(LOGS_DIR / f"classifier_test_log_{timestamp}.txt", sys.stdout)

    print("="*60)
    print(f"🚀 Classifier 模型測試啟動 - {timestamp}")
    print("="*60)

    # 1. 測試裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ 測試裝置: {device}")
    if device.type == 'cuda':
        print(f"   顯卡型號: {torch.cuda.get_device_name(0)}")

    # 2. 建立模型
    print("\n🏗️ 建立模型 (ResNet18)...")
    model = CXRClassifier(num_classes=2, pretrained=True).to(device)
    
    # 3. 測試前向傳播
    dummy_input = torch.randn(4, 3, 512, 512).to(device)
    
    print(f"\n⚡ 執行前向傳播測試...")
    print(f"   輸入形狀: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"   輸出形狀: {output.shape} (預期: [4, 2])")
    # print(f"   輸出數值:\n{output.cpu().numpy()}")

    # 4. 測試特徵提取
    print(f"\n🔍 測試特徵提取...")
    with torch.no_grad():
        features = model.get_features(dummy_input)
    print(f"   特徵形狀: {features.shape} (預期: [4, 512])")

    # 5. 測試凍結機制
    print("\n🧊 測試凍結機制...")
    model.freeze_backbone()
    print(f"   凍結後參數: {model.get_num_params():,}")
    
    model.unfreeze_backbone()
    print(f"   解凍後參數: {model.get_num_params():,}")

    # 6. 測試優化器 (這裡修正了參數名稱)
    try:
        # [修正] 將 lr 改為 learning_rate 以符合函式定義
        optimizer = create_optimizer(model, 'adam', learning_rate=1e-3)
        print("✅ 優化器建立成功")
    except Exception as e:
        print(f"❌ 優化器建立失敗: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("🎉 Classifier 模組測試完成！")
    print("="*60)

if __name__ == "__main__":
    main()