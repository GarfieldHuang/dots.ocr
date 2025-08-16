#!/usr/bin/env python3
"""
簡單圖片解析器 - 最簡單的圖片 OCR 使用方式

這個範例展示如何：
1. 快速初始化解析器
2. 解析圖片並獲取文字內容
3. 處理不同的輸出格式
"""

import os
import sys
import json
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dots_ocr.parser import DotsOCRParser


class SimpleImageParser:
    """簡化的圖片解析器"""
    
    def __init__(self, use_hf=False):
        """
        初始化解析器
        
        Args:
            use_hf (bool): 是否使用 HuggingFace 模型，False 則使用 vLLM
        """
        self.parser = DotsOCRParser(use_hf=use_hf)
    
    def extract_text_only(self, image_path):
        """
        只提取文字內容，不包含版面資訊
        
        Args:
            image_path (str): 圖片路徑
            
        Returns:
            str: 提取的文字內容
        """
        results = self.parser.parse_file(
            input_path=image_path,
            prompt_mode="prompt_ocr"  # 只提取文字
        )
        
        if results and 'md_content_path' in results[0]:
            with open(results[0]['md_content_path'], 'r', encoding='utf-8') as f:
                return f.read().strip()
        return ""
    
    def extract_with_layout(self, image_path):
        """
        提取文字和版面資訊
        
        Args:
            image_path (str): 圖片路徑
            
        Returns:
            tuple: (文字內容, 版面資訊)
        """
        results = self.parser.parse_file(
            input_path=image_path,
            prompt_mode="prompt_layout_all_en"
        )
        
        text_content = ""
        layout_info = None
        
        if results:
            result = results[0]
            
            # 讀取文字內容
            if 'md_content_path' in result:
                with open(result['md_content_path'], 'r', encoding='utf-8') as f:
                    text_content = f.read().strip()
            
            # 讀取版面資訊
            if 'layout_info_path' in result:
                with open(result['layout_info_path'], 'r', encoding='utf-8') as f:
                    layout_info = json.load(f)
        
        return text_content, layout_info
    
    def detect_layout_only(self, image_path):
        """
        只檢測版面，不識別文字
        
        Args:
            image_path (str): 圖片路徑
            
        Returns:
            list: 版面檢測結果
        """
        results = self.parser.parse_file(
            input_path=image_path,
            prompt_mode="prompt_layout_only_en"
        )
        
        if results and 'layout_info_path' in results[0]:
            with open(results[0]['layout_info_path'], 'r', encoding='utf-8') as f:
                return json.load(f)
        return []


def main():
    """簡單圖片解析示例"""
    print("=== 簡單圖片解析器範例 ===")
    
    # 使用範例圖片
    image_path = "../demo/demo_image1.jpg"
    
    if not os.path.exists(image_path):
        print(f"錯誤：找不到範例圖片 {image_path}")
        return
    
    # 創建簡單解析器
    parser = SimpleImageParser(use_hf=False)  # 使用 vLLM
    
    print(f"正在解析圖片：{image_path}")
    
    try:
        # 1. 只提取文字
        print("\n--- 1. 只提取文字 ---")
        text = parser.extract_text_only(image_path)
        print("提取的文字內容：")
        print(text[:500] + "..." if len(text) > 500 else text)
        
        # 2. 提取文字和版面
        print("\n--- 2. 提取文字和版面資訊 ---")
        text_content, layout_info = parser.extract_with_layout(image_path)
        
        print("文字內容長度：", len(text_content))
        
        if layout_info:
            print(f"檢測到 {len(layout_info)} 個版面元素：")
            for i, item in enumerate(layout_info[:5]):  # 只顯示前5個
                category = item.get('category', 'Unknown')
                bbox = item.get('bbox', [])
                print(f"  {i+1}. 類型: {category}, 位置: {bbox}")
            
            if len(layout_info) > 5:
                print(f"  ... 還有 {len(layout_info) - 5} 個元素")
        
        # 3. 只檢測版面
        print("\n--- 3. 只檢測版面（不識別文字）---")
        layout_only = parser.detect_layout_only(image_path)
        
        if layout_only:
            print(f"檢測到 {len(layout_only)} 個版面區域：")
            category_count = {}
            for item in layout_only:
                category = item.get('category', 'Unknown')
                category_count[category] = category_count.get(category, 0) + 1
            
            for category, count in category_count.items():
                print(f"  {category}: {count} 個")
        
        print(f"\n解析完成！結果已保存到：{parser.parser.output_dir}")
        
    except Exception as e:
        print(f"解析過程中發生錯誤：{e}")
        print("請確認 vLLM 伺服器已啟動且模型已加載")


if __name__ == "__main__":
    main()