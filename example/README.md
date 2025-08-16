# DotsOCR 使用範例

這個資料夾包含了各種使用 DotsOCR 的完整範例，涵蓋從基礎使用到進階應用的各種場景。

## 📁 檔案說明

### 基礎使用
- **`quick_start.py`** - 最簡單的開始範例，5分鐘上手 DotsOCR
- **`basic_usage.py`** - 基本使用方法和配置
- **`simple_image_parser.py`** - 單張圖片解析範例

### 高效處理
- **`batch_processing.py`** - 批次處理多個檔案
- **`vllm_server_usage.py`** - 使用 vLLM 伺服器進行高效推理
- **`hf_transformers_usage.py`** - 使用 HuggingFace Transformers

### 進階功能
- **`advanced_parsing.py`** - 進階解析功能和自定義配置
- **`grounding_ocr_example.py`** - 精確定位 OCR 範例
- **`pdf_parsing_enhanced.py`** - **🆕 增強版 PDF 解析** - 支援混合中英文、表格、圖片、公式等複雜內容
- **`multilingual_example.py`** - 多語言文檔處理範例

### 工具程式
- **`run_all_examples.py`** - 一鍵運行所有範例

## 🚀 快速開始

如果您是第一次使用 DotsOCR，建議從這個順序開始：

1. **`quick_start.py`** - 了解基本概念
2. **`basic_usage.py`** - 學習詳細配置
3. **`pdf_parsing_enhanced.py`** - 嘗試複雜文檔解析 (🆕 增強版)
4. **`batch_processing.py`** - 批次處理檔案

## 🆕 增強版 PDF 解析功能

最新的 `pdf_parsing_enhanced.py` 提供了強大的混合內容處理能力：

### ✨ 主要特色
- **混合語言識別**：自動識別和分離中文、英文、混合語言內容
- **內容類型分析**：智能識別表格、圖片、公式、標題等不同類型
- **結構化輸出**：按內容類型分別保存，便於後續處理
- **高解析度處理**：支援 300 DPI 高品質解析
- **批量處理**：支援多檔案批次處理

### 🎯 解析模式
1. **基本解析** - 標準 OCR 處理
2. **複雜內容解析** - 增強的混合語言和多媒體處理
3. **批量解析** - 處理多個檔案
4. **自定義解析** - 可調整 DPI、語言偏好等參數

### 📊 輸出格式
- 中文內容 → `*_chinese.md`
- 英文內容 → `*_english.md`
- 混合語言內容 → `*_mixed_language.md`
- 表格內容 → `*_tables.html`
- 公式內容 → `*_formulas.md`
- 圖片資訊 → `*_images.json`
- 標題內容 → `*_headers.md`
- 分析報告 → `*_content_report.txt`

## 💡 使用建議

### 處理策略
- **文字文檔**：使用 150-200 DPI，選擇基本解析模式
- **學術論文**：使用 300 DPI，選擇複雜內容解析，重點處理公式和表格
- **商業報告**：使用 200-300 DPI，重點處理表格和圖表
- **混合語言文檔**：使用複雜內容解析模式，自動分離不同語言

### 效能優化
- **DPI 設定**：150 (快速) / 200 (平衡) / 300 (高品質)
- **線程數**：根據 CPU 核心數調整 (建議 4-8)
- **記憶體管理**：大檔案建議分批處理

## 🛠 環境要求

確保您已安裝以下依賴：

```bash
# 基本依賴
pip install torch transformers pillow pdf2image

# 可選依賴 (用於 vLLM)
pip install vllm

# PDF 處理依賴
pip install pymupdf  # 或 pip install fitz
```

## 📖 詳細使用說明

每個範例檔案都包含：
- ✅ 完整的程式碼註解
- ✅ 錯誤處理機制
- ✅ 性能優化建議
- ✅ 實際使用範例

## 🚨 注意事項

1. **模型載入**：首次使用需要下載約 3.5GB 的模型檔案
2. **GPU 記憶體**：建議至少 8GB GPU 記憶體
3. **處理時間**：複雜文檔解析時間較長，請耐心等待
4. **檔案格式**：支援 PDF、PNG、JPG、JPEG 等格式

## 🔧 故障排除

### 常見問題
- **記憶體不足**：降低 DPI 或使用 CPU 模式
- **模型載入失敗**：檢查網路連接和磁碟空間
- **PDF 解析失敗**：確認已安裝 `pdf2image` 和 `poppler`
- **中文顯示問題**：確認終端支援 UTF-8 編碼

### 獲取幫助
- 查看程式碼註解
- 運行 `run_all_examples.py` 進行測試
- 檢查輸出目錄中的日誌檔案

## 📈 更新日誌

### v2.0 (最新)
- 🆕 增強版 PDF 解析功能
- 🆕 混合語言自動識別
- 🆕 結構化內容提取
- 🆕 批量處理支援
- ⚡ 效能優化和錯誤處理改進

### v1.0
- ✅ 基本 OCR 功能
- ✅ 多語言支援
- ✅ vLLM 整合
- ✅ 批次處理

---

Happy coding! 🎉

## 使用前準備

確保您已經：

1. 安裝並設定好 dots.ocr 環境
2. 下載模型權重到 `./weights/DotsOCR`
3. 啟動 vLLM 伺服器（對於需要的範例）

```bash
# 啟動 conda 環境
conda activate dots_ocr

# 如果使用 vLLM 伺服器，請先啟動
CUDA_VISIBLE_DEVICES=0 vllm serve ./weights/DotsOCR --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --chat-template-content-format string --served-model-name model --trust-remote-code
```

## 快速開始

```python
from dots_ocr.parser import DotsOCRParser

# 創建解析器
parser = DotsOCRParser()

# 解析圖片
results = parser.parse_file("your_image.jpg")
print(results)
```

查看各個範例檔案以了解更詳細的使用方法。