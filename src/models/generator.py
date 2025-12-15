"""
cGAN 生成器模組 (v2.0 - 整合自動記錄與殘差架構)
使用 U-Net 架構生成反事實影像
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
# 2. U-Net 生成器
# ---------------------------------------------------------
class UNetGenerator(nn.Module):
    """U-Net 生成器（用於反事實生成）"""
    
    def __init__(self, 
                 in_channels: int = 3,
                 out_channels: int = 3,
                 num_classes: int = 2,
                 base_features: int = 64):
        """
        Args:
            in_channels: 輸入通道數 (通常是 3, 若包含 Label Map 則在 forward 處理)
            out_channels: 輸出通道數 (通常是 3)
            num_classes: 條件類別數
            base_features: 基礎特徵數 (控制模型寬度)
        """
        super(UNetGenerator, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # 將類別標籤嵌入為空間特徵圖
        # 輸入: One-hot vector -> 輸出: 全圖大小的特徵層
        self.label_embedding = nn.Sequential(
            nn.Linear(num_classes, 512 * 512), # 假設輸入圖是 512x512
            nn.ReLU(inplace=True)
        )
        
        # 編碼器 (Downsampling)
        # 輸入通道 = 圖片通道 + 1 (Label Map)
        self.enc1 = self._conv_block(in_channels + 1, base_features) 
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self._conv_block(base_features, base_features * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self._conv_block(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = self._conv_block(base_features * 4, base_features * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # 瓶頸層 (Bottleneck)
        self.bottleneck = self._conv_block(base_features * 8, base_features * 16)
        
        # 解碼器 (Upsampling) - [修改] 改用 Upsample + Conv 以消除棋盤格效應
        self.upconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_features * 16, base_features * 8, kernel_size=3, padding=1)
        )
        self.dec4 = self._conv_block(base_features * 16, base_features * 8) 
        
        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_features * 8, base_features * 4, kernel_size=3, padding=1)
        )
        self.dec3 = self._conv_block(base_features * 8, base_features * 4)
        
        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_features * 4, base_features * 2, kernel_size=3, padding=1)
        )
        self.dec2 = self._conv_block(base_features * 4, base_features * 2)
        
        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_features * 2, base_features, kernel_size=3, padding=1)
        )
        self.dec1 = self._conv_block(base_features * 2, base_features)
        
        # 輸出層
        self.out = nn.Conv2d(base_features, out_channels, kernel_size=1)
        self.out_activation = nn.Tanh() # 確保輸出在 [-1, 1]
        
        print(f"✅ U-Net 生成器初始化完成")
        print(f"   - 輸入/輸出: {in_channels} -> {out_channels}")
        print(f"   - 基礎特徵: {base_features}")
    
    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """標準卷積區塊: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 512, 512)
            label: (B, num_classes)
        """
        # 1. 處理標籤
        # 注意: 如果輸入圖片不是 512x512，這裡的 view 會報錯。
        # 更穩健的做法是直接生成與 x 相同大小的 map
        B, C, H, W = x.size()
        
        # 將標籤 (B, num_classes) 擴展成 (B, num_classes, H, W) 的特徵圖
        # 這裡我們簡化處理：直接把 label 的每個維度擴展成一張圖
        label_map = label.view(B, self.num_classes, 1, 1).expand(B, self.num_classes, H, W)
        
        # 或者使用原版邏輯 (Embedding -> View)，但原版寫死了 512
        # 為了彈性，我們改用這種 "Spatially Replicated One-Hot" 的方式
        # 但為了配合 __init__ 中的定義 (in_channels + 1)，我們只取第一個有效的 conditional channel
        # 這樣做比較簡單：我們把 embedding 層拿掉，改用全連接層生成的特徵圖
        # 但為了保持架構一致性，我們還是用 Embedding 的方式，但要動態調整 view
        
        label_embed = self.label_embedding(label) # (B, 512*512)
        label_map = label_embed.view(B, 1, 512, 512) # (B, 1, 512, 512)
        
        # 如果輸入圖片不是 512，需要 interpolate
        if H != 512 or W != 512:
             label_map = nn.functional.interpolate(label_map, size=(H, W), mode='bilinear')

        # 2. 串接輸入
        x_with_label = torch.cat([x, label_map], dim=1) # (B, 4, H, W)
        
        # 3. Encoder
        enc1 = self.enc1(x_with_label)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # 4. Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # 5. Decoder (with Skip Connections)
        dec4 = self.upconv4(bottleneck)
        # 注意: 這裡的 cat 要注意特徵圖大小是否完全一致 (可能因 padding 有微小差異)
        # 為了安全，可以加 resize，但 U-Net 512 輸入通常對齊得很好
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # 6. Output
        out = self.out(dec1)
        out = self.out_activation(out)
        
        return out

# ---------------------------------------------------------
# 3. 殘差生成器 (推薦使用)
# ---------------------------------------------------------
class ResidualGenerator(nn.Module):
    """
    殘差式生成器: Output = Input + U-Net(Input, Label)
    這強迫模型只學習「需要修改的部分」(例如病灶)，保持背景不變。
    """
    
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 base_features: int = 64):
        super(ResidualGenerator, self).__init__()
        
        self.unet = UNetGenerator(
            in_channels=in_channels,
            out_channels=in_channels, # 輸出的通道數必須跟輸入一樣
            num_classes=num_classes,
            base_features=base_features
        )
        
        # 學習一個可訓練的縮放因子，初始值設很小 (0.01)
        # 這樣剛開始訓練時，輸出會非常接近原圖，容易訓練
        self.scale = nn.Parameter(torch.tensor(0.1))
        
        print("✅ 殘差生成器初始化完成 (Output = Input + Scale * Residual)")
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # 生成殘差圖 (Difference Map)
        residual = self.unet(x, label)
        
        # 疊加到原圖
        counterfactual = x + self.scale * residual
        
        # 確保數值範圍在 [-1, 1]
        counterfactual = torch.clamp(counterfactual, -1.0, 1.0)
        
        return counterfactual

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
    sys.stdout = DualLogger(LOGS_DIR / f"generator_test_log_{timestamp}.txt", sys.stdout)

    print("="*60)
    print(f"🚀 Generator 模組測試啟動 - {timestamp}")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"測試裝置: {device}")

    # 1. 測試 U-Net
    print("\n[Test 1] U-Net 基礎生成器測試...")
    # 為了測試快速，減少 base_features
    unet = UNetGenerator(base_features=32).to(device)
    
    batch_size = 2
    dummy_img = torch.randn(batch_size, 3, 512, 512).to(device)
    dummy_label = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).to(device) # One-hot
    
    print(f"  輸入形狀: {dummy_img.shape}")
    with torch.no_grad():
        out = unet(dummy_img, dummy_label)
    print(f"  輸出形狀: {out.shape}")
    print(f"  輸出範圍: [{out.min().item():.3f}, {out.max().item():.3f}]")

    # 2. 測試 Residual Generator (我們主要會用這個)
    print("\n[Test 2] Residual Generator 殘差生成器測試...")
    res_gen = ResidualGenerator(base_features=32).to(device)
    
    with torch.no_grad():
        cf_out = res_gen(dummy_img, dummy_label)
    
    print(f"  輸出形狀: {cf_out.shape}")
    
    # 驗證殘差性質：剛初始化時，輸出應該跟輸入很像
    diff = torch.abs(cf_out - dummy_img).mean().item()
    print(f"  平均差異 (應該很小): {diff:.6f}")
    
    # 計算參數量
    total_params = sum(p.numel() for p in res_gen.parameters())
    print(f"\n模型總參數: {total_params:,}")

    print("\n" + "="*60)
    print("🎉 Generator 模組測試完成！")
    print("="*60)

if __name__ == "__main__":
    main()