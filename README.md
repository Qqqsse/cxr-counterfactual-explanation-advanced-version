# CXR Counterfactual Explanation (Advanced)

胸腔 X 光 (CXR) 反事實解釋研究專案。  
本專案使用 cGAN 與 STN + cGAN 生成反事實影像，目標是分析分類模型決策時依賴的影像線索，並提升醫療影像 AI 的可解釋性。

## 1. 研究目標

- 建立 CXR 二元分類任務（Cardiomegaly vs Non-Cardiomegaly）的反事實生成流程
- 比較純紋理導向的 cGAN 與具幾何控制能力的 STN + cGAN
- 量化反事實品質與分類反轉能力（SSIM、L1、分類成功率等）
- 透過可視化結果輔助臨床可解釋性分析

## 2. 專案現況

### 已完成
- Phase 1：cGAN 訓練流程 (`notebooks/03_cgan_training.ipynb`)
- Phase 2：STN 核心模組與整合生成器
  - `src/models/stn.py`
  - `src/models/stn_generator.py`
- 反事實分析流程 (`notebooks/04_counterfactual_analysis.ipynb`)
- STN 整合測試腳本 (`test_stn_integration.py`)
- Phase 2 文件
  - `docs/PHASE2_ARCHITECTURE_ANALYSIS.md`
  - `docs/PHASE2_IMPLEMENTATION_SUMMARY.md`
  - `docs/PHASE2_USAGE_GUIDE.md`

### 待補強
- 分類器訓練流程文件化（目前以既有權重載入為主）
- 專案根目錄下 `results/` 與 `models/` 產物的統一保存與版本化
- 部分 notebook 路徑命名一致性整理

## 3. 環境需求與安裝

### 環境需求
- Python 3.9+
- PyTorch 2.5.1 + CUDA 11.8
- NVIDIA GPU（建議 12GB+ VRAM；專案配置以 16GB VRAM 測試）

### 安裝方式（Conda）
```bash
conda env create -f environment.yml
conda activate cxr_cf
```

### 安裝方式（pip）
```bash
pip install -r requirements.txt
```

## 4. 完整專案結構

> 以下為目前 repo 可掃描到的實際結構（到 2-3 層）。

```text
cxr-counterfactual-explanation-advanced-version/
├─ README.md
├─ requirements.txt
├─ environment.yml
├─ .gitignore
├─ test_stn_integration.py
├─ data/
│  └─ labels/
│     └─ binary_labels_cardiomegaly.csv
├─ docs/
│  ├─ PHASE2_ARCHITECTURE_ANALYSIS.md
│  ├─ PHASE2_IMPLEMENTATION_SUMMARY.md
│  └─ PHASE2_USAGE_GUIDE.md
├─ notebooks/
│  ├─ 03_cgan_training.ipynb
│  ├─ 04_counterfactual_analysis.ipynb
│  └─ 05_phase2_stn_training.ipynb
└─ src/
   ├─ data/
   │  ├─ label_extraction.py
   │  ├─ preprocessing.py
   │  └─ dataset.py
   ├─ models/
   │  ├─ classifier.py
   │  ├─ discriminator.py
   │  ├─ generator.py
   │  ├─ losses.py
   │  ├─ stn.py
   │  └─ stn_generator.py
   ├─ utils/
   │  └─ metrics.py
   └─ visualization/
      └─ plotting.py
```

## 5. 實驗流程

### Step 1. 標籤抽取與資料前處理
1. 使用 `src/data/label_extraction.py` 從 IU X-ray metadata/reports 產生二元標籤
2. 使用 `src/data/preprocessing.py` 進行：
   - CLAHE
   - resize
   - normalization
   - train/val/test split
3. 由 `src/data/dataset.py` 建立 PyTorch Dataset/DataLoader

### Step 2. 分類器準備
- 使用 `src/models/classifier.py` 的分類器架構與載入工具
- notebook 訓練/推論流程會載入既有分類器權重作為教師或評估器

### Step 3. Phase 1 - cGAN 反事實訓練
- Notebook：`notebooks/03_cgan_training.ipynb`
- 主要組件：
  - Generator：`src/models/generator.py`
  - Discriminator：`src/models/discriminator.py`
  - Losses：`src/models/losses.py`
- 典型輸出：
  - `models/generator/best_cgan.pth`
  - `results/metrics/cgan_history.csv`
  - `results/figures/cgan_training.png`

### Step 4. Phase 2 - STN + cGAN 訓練
- Notebook：`notebooks/05_phase2_stn_training.ipynb`
- 主要組件：
  - STN：`src/models/stn.py`
  - STN + Residual Generator：`src/models/stn_generator.py`
- 典型輸出：
  - `models/generator/best_stn_cgan.pth`
  - `results/metrics/stn_cgan_history.csv`
  - `results/figures/stn_cgan_training.png`

### Step 5. 反事實分析與可視化
- Notebook：`notebooks/04_counterfactual_analysis.ipynb`
- 指標工具：`src/utils/metrics.py`
- 視覺化工具：`src/visualization/plotting.py`
- 典型輸出：
  - `results/metrics/counterfactual_results.csv`
  - `results/metrics/final_report.json`
  - `results/figures/counterfactuals/*.png`
  - `results/figures/gradcam/*.png`

## 6. 實驗結果（依 notebook 已儲存輸出整理）

> 下列數值來自 notebook 內已保存的執行輸出；實際結果會因資料版本、權重、隨機種子與訓練中斷狀態而異。

### Phase 1 / 分析流程可見結果
- `notebooks/04_counterfactual_analysis.ipynb` 可見：
  - Counterfactual success rate: **85.24%**
  - Mean SSIM: **0.9926**
  - Mean L1: **0.0048**

### 訓練 notebook 可見結果
- `notebooks/03_cgan_training.ipynb` 已儲存輸出中可見驗證分類成功率曾達約 **83.33%**
- `notebooks/05_phase2_stn_training.ipynb` 已儲存輸出中可見驗證分類成功率約 **83.70%**

### 注意事項
- `notebooks/05_phase2_stn_training.ipynb` 含 `KeyboardInterrupt` 紀錄，表示該次訓練曾中斷。
- 建議以固定 seed + 完整重跑後，再更新最終報告數值。

## 7. 常用執行順序

```bash
# 1) 準備環境
conda activate cxr_cf

# 2) 執行資料前處理（依需求）
python src/data/label_extraction.py
python src/data/preprocessing.py

# 3) 依序執行 notebook
# - 03_cgan_training.ipynb
# - 05_phase2_stn_training.ipynb
# - 04_counterfactual_analysis.ipynb
```

## 8. STN 模組快速驗證

```bash
python test_stn_integration.py
```

此測試會檢查：
- 模組導入
- 模型初始化
- 前向傳播
- 梯度反傳
- 與資料集介接
- 參數量統計

## 9. 研究限制與後續方向

- 需要補齊完整可重現的結果檔案管理（weights、csv、figures）
- 可進一步做 STN 變換類型對照（Affine / TPS / Hybrid）
- 可補充更完整的可解釋性比較（Grad-CAM 前後差異、臨床區域量化）
- 可增加跨資料集驗證以降低 domain bias