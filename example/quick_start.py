#!/usr/bin/env python3
"""
快速開始範例 - 最簡單的 dots.ocr 使用方式

這個腳本展示如何用最少的程式碼開始使用 dots.ocr
"""

import os
import sys
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def quick_start():
    """快速開始示例"""
    print("=== DotsOCR 快速開始 ===")
    
    # 最簡單的使用方式
    try:
        from dots_ocr.parser import DotsOCRParser
        
        # 1. 創建解析器（使用預設設置）
        parser = DotsOCRParser()
        
        # 2. 解析圖片
        image_path = "../demo/demo_image1.jpg"
        
        if not os.path.exists(image_path):
            print(f"找不到範例圖片：{image_path}")
            print("請確認 demo 資料夾存在並包含範例圖片")
            return
        
        print(f"正在解析：{image_path}")
        results = parser.parse_file(image_path)
        
        # 3. 查看結果
        if results:
            result = results[0]
            print(f"✓ 解析成功！")
            print(f"檔案：{result['file_path']}")
            print(f"處理尺寸：{result['input_width']} x {result['input_height']}")
            
            # 如果有版面資訊，顯示統計
            if 'layout_info_path' in result:
                import json
                with open(result['layout_info_path'], 'r', encoding='utf-8') as f:
                    layout_data = json.load(f)
                
                print(f"檢測到 {len(layout_data)} 個版面元素")
                
                # 統計元素類型
                categories = {}
                for item in layout_data:
                    cat = item.get('category', 'Unknown')
                    categories[cat] = categories.get(cat, 0) + 1
                
                print("元素類型統計：")
                for cat, count in categories.items():
                    print(f"  {cat}: {count}")
            
            # 如果有文字內容，顯示預覽
            if 'md_content_path' in result:
                with open(result['md_content_path'], 'r', encoding='utf-8') as f:
                    text_content = f.read()
                
                print(f"\\n提取的文字內容（前200字符）：")
                print("-" * 40)
                print(text_content[:200] + "..." if len(text_content) > 200 else text_content)
                print("-" * 40)
            
            print(f"\\n完整結果已保存到：{parser.output_dir}")
            
        else:
            print("✗ 解析失敗")
            
    except Exception as e:
        print(f"發生錯誤：{e}")
        print("\\n可能的解決方法：")
        print("1. 確認已啟動 vLLM 伺服器")
        print("2. 檢查模型權重是否已下載")
        print("3. 確認在正確的 conda 環境中")

def show_next_steps():
    """顯示後續步驟"""
    print("\\n=== 後續步驟 ===")
    print("✅ 成功！現在您可以：")
    print()
    print("1. 查看其他範例：")
    print("   python basic_usage.py           # 基本使用方法")
    print("   python simple_image_parser.py   # 簡單圖片解析")
    print("   python batch_processing.py      # 批次處理")
    print()
    print("2. 處理您自己的檔案：")
    print("   parser = DotsOCRParser()")
    print("   results = parser.parse_file('your_image.jpg')")
    print()
    print("3. 不同的解析模式：")
    print("   prompt_layout_all_en    # 完整解析（預設）")
    print("   prompt_layout_only_en   # 只檢測版面")
    print("   prompt_ocr              # 只提取文字")
    print("   prompt_grounding_ocr    # 指定區域OCR")
    print()
    print("4. 調整設置：")
    print("   parser = DotsOCRParser(")
    print("       use_hf=True,        # 使用 HuggingFace 而非 vLLM")
    print("       dpi=300,            # 更高解析度")
    print("       num_thread=8        # 更多線程")
    print("   )")
    print()
    print("📖 詳細說明請查看 README.md 或各個範例檔案")

if __name__ == "__main__":
    quick_start()
    show_next_steps()