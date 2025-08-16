#!/usr/bin/env python3
"""
基本使用範例 - 展示 dots.ocr 的基本功能

這個範例展示如何：
1. 初始化 DotsOCRParser
2. 解析單一圖片
3. 獲取解析結果
"""

import os
import sys
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dots_ocr.parser import DotsOCRParser


def main():
    """基本使用示例"""
    print("=== DotsOCR 基本使用範例 ===")
    
    # 1. 創建解析器實例
    # 使用 vLLM 伺服器（推薦）
    parser = DotsOCRParser(
        ip='localhost',           # vLLM 伺服器 IP
        port=8000,               # vLLM 伺服器端口
        model_name='model',      # 模型名稱
        use_hf=False,           # 使用 vLLM 而非 HuggingFace
        output_dir="./output"    # 輸出目錄
    )
    
    # 2. 準備輸入圖片
    # 這裡使用 demo 中的範例圖片
    image_path = "../demo/demo_image1.jpg"
    
    if not os.path.exists(image_path):
        print(f"錯誤：找不到範例圖片 {image_path}")
        print("請確認您在正確的目錄下執行此腳本，並且 demo 資料夾存在")
        return
    
    print(f"正在解析圖片：{image_path}")
    
    try:
        # 3. 解析圖片
        # 使用預設的完整版面解析模式
        results = parser.parse_file(
            input_path=image_path,
            prompt_mode="prompt_layout_all_en"  # 完整版面解析（包含檢測和識別）
        )
        
        # 4. 顯示結果
        print(f"\n解析完成！共處理 {len(results)} 個頁面")
        
        for i, result in enumerate(results):
            print(f"\n--- 第 {i+1} 頁結果 ---")
            print(f"檔案路徑: {result['file_path']}")
            print(f"輸入尺寸: {result['input_width']} x {result['input_height']}")
            
            # 檢查是否有版面資訊
            if 'layout_info_path' in result:
                print(f"版面資訊 JSON: {result['layout_info_path']}")
                print(f"版面標註圖片: {result['layout_image_path']}")
            
            # 檢查是否有 Markdown 輸出
            if 'md_content_path' in result:
                print(f"Markdown 內容: {result['md_content_path']}")
                print(f"Markdown 內容 (無頁首頁尾): {result['md_content_nohf_path']}")
        
        print(f"\n所有結果已保存到：{parser.output_dir}")
        
    except Exception as e:
        print(f"解析過程中發生錯誤：{e}")
        print("請檢查：")
        print("1. vLLM 伺服器是否已啟動")
        print("2. 模型權重是否已下載")
        print("3. 網路連接是否正常")


if __name__ == "__main__":
    main()