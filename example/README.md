# DotsOCR 使用範例

這個資料夾包含了使用 dots.ocr 的各種範例程式。

## 範例概述

1. **basic_usage.py** - 基本使用方法
2. **simple_image_parser.py** - 簡單圖片解析
3. **batch_processing.py** - 批次處理多個檔案
4. **vllm_server_usage.py** - 使用 vLLM 伺服器
5. **hf_transformers_usage.py** - 使用 HuggingFace Transformers
6. **advanced_parsing.py** - 高級解析功能
7. **grounding_ocr_example.py** - 指定區域 OCR
8. **pdf_parsing_example.py** - PDF 解析範例
9. **multilingual_example.py** - 多語言文件解析

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