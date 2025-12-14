"""
cGAN 判別器模組 (v2.0 - 整合自動記錄與記憶體優化)
使用 PatchGAN 架構判斷影像真實性
"""

import sys
import torch
import torch.nn as nn
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
# 2. PatchGAN 判別器 (標準版)
# ---------------------------------------------------------
class PatchGANDiscriminator(nn.Module):
    """PatchGAN 判別器（輸出為 N x N 的特徵圖，而非單一數值）"""
    
    def __init__(self, 
                 in_channels: int = 3, 
                 num_classes: int = 2, 
                 base_features: int = 64):
        """
        Args:
            in_channels: 影像通道數 (RGB=3)
            num_classes: 條件類別數 (用於 cGAN)
            base_features: 網路寬度
        """
        super(PatchGANDiscriminator, self).__init__()
        
        self.num_classes = num_classes
        
        # PatchGAN 架構
        # 輸入通道 = 影像通道 + 類別通道 (直接串接)
        input_channels = in_channels + num_classes
        
        self.model = nn.Sequential(
            # Layer 1: (B, C+N, H, W) -> (B, 64, H/2, W/2)
            # 第一層通常不加 BatchNorm
            nn.Conv2d(input_channels, base_features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: (B, 64, H/2, W/2) -> (B, 128, H/4, W/4)
            nn.Conv2d(base_features, base_features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: (B, 128, H/4, W/4) -> (B, 256, H/8, W/8)
            nn.Conv2d(base_features * 2, base_features * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: (B, 256, H/8, W/8) -> (B, 512, H/16, W/16)
            nn.Conv2d(base_features * 4, base_features * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output Layer: (B, 512, ...) -> (B, 1, 30, 30) for 512 input
            # 這裡輸出的是 logits (還沒經過 Sigmoid)，Loss Function 會處理
            nn.Conv2d(base_features * 8, 1, kernel_size=4, stride=1, padding=1)
        )
        
        print(f"✅ PatchGAN 判別器初始化完成")
        print(f"   - 實際輸入通道: {input_channels} ({in_channels} img + {num_classes} cls)")
        print(f"   - Receptive Field: 70x70")
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 影像 (B, C, H, W)
            label: 標籤 (B, num_classes)
        """
        B, C, H, W = x.size()
        
        # [優化] 空間複製 (Spatially Replicate)
        # 將 (B, num_classes) -> (B, num_classes, H, W)
        label_map = label.view(B, self.num_classes, 1, 1).expand(B, self.num_classes, H, W)
        
        # 串接影像與標籤
        x_with_label = torch.cat([x, label_map], dim=1) # (B, C + num_classes, H, W)
        
        return self.model(x_with_label)

# ---------------------------------------------------------
# 3. Spectral Norm 判別器 (推薦使用：更穩定)
# ---------------------------------------------------------
class SpectralNormDiscriminator(nn.Module):
    """
    帶 Spectral Normalization 的判別器
    SN 能限制判別器的 Lipschitz 常數，防止梯度爆炸，對 GAN 訓練穩定性極有幫助。
    """
    
    def __init__(self, 
                 in_channels: int = 3, 
                 num_classes: int = 2, 
                 base_features: int = 64):
        super(SpectralNormDiscriminator, self).__init__()
        
        self.num_classes = num_classes
        input_channels = in_channels + num_classes
        
        self.model = nn.Sequential(
            # Layer 1
            nn.utils.spectral_norm(
                nn.Conv2d(input_channels, base_features, 4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2
            nn.utils.spectral_norm(
                nn.Conv2d(base_features, base_features * 2, 4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3
            nn.utils.spectral_norm(
                nn.Conv2d(base_features * 2, base_features * 4, 4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4
            nn.utils.spectral_norm(
                nn.Conv2d(base_features * 4, base_features * 8, 4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output
            nn.utils.spectral_norm(
                nn.Conv2d(base_features * 8, 1, 4, stride=1, padding=1)
            )
        )
        
        print(f"✅ Spectral Norm 判別器初始化完成 (Stable Mode)")
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        label_map = label.view(B, self.num_classes, 1, 1).expand(B, self.num_classes, H, W)
        x_with_label = torch.cat([x, label_map], dim=1)
        return self.model(x_with_label)

# ---------------------------------------------------------
# 4. 輔助函數
# ---------------------------------------------------------
def initialize_weights(model):
    """初始化模型權重（DCGAN 標準：Normal(0, 0.02)）"""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

# ---------------------------------------------------------
# 5. 主測試程式
# ---------------------------------------------------------
def test_discriminator():
    """測試腳本"""
    # --- Log 設定 ---
    try:
        script_path = Path(__file__).resolve()
        project_root = script_path.parents[2]
    except NameError:
        project_root = Path.cwd()
        
    LOGS_DIR = project_root / "results" / "logs"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sys.stdout = DualLogger(LOGS_DIR / f"discriminator_test_log_{timestamp}.txt", sys.stdout)

    print("="*60)
    print(f"🚀 Discriminator 模組測試啟動 - {timestamp}")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"測試裝置: {device}")
    
    # 測試輸入
    batch_size = 2
    # 模擬 512x512 影像
    x = torch.randn(batch_size, 3, 512, 512).to(device)
    # 模擬 One-hot 標籤
    label = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).to(device)
    
    print("\n[Test 1] Standard PatchGAN...")
    discriminator = PatchGANDiscriminator(
        in_channels=3,
        num_classes=2,
        base_features=64
    ).to(device)
    
    initialize_weights(discriminator)
    
    with torch.no_grad():
        output = discriminator(x, label)
        
    print(f"  輸入形狀: {x.shape}")
    print(f"  輸出形狀: {output.shape} (預期為 [B, 1, 30, 30] 左右)")
    print(f"  輸出範圍: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # 計算參數量
    num_params = sum(p.numel() for p in discriminator.parameters())
    print(f"  總參數量: {num_params:,}")

    print("\n[Test 2] Spectral Norm Discriminator (推薦)...")
    sn_discriminator = SpectralNormDiscriminator(
        in_channels=3,
        num_classes=2,
        base_features=64
    ).to(device)
    
    with torch.no_grad():
        sn_output = sn_discriminator(x, label)
    
    print(f"  輸出形狀: {sn_output.shape}")
    print(f"  輸出範圍: [{sn_output.min().item():.3f}, {sn_output.max().item():.3f}]")

    print("\n" + "="*60)
    print("🎉 Discriminator 模組測試完成！")
    print("="*60)

if __name__ == "__main__":
    test_discriminator()