#!/usr/bin/env python3
"""
高級解析功能範例 - 展示 dots.ocr 的進階特性

這個範例展示如何：
1. 自訂解析參數
2. 處理不同解析度的圖片
3. 提取特定類型的內容（表格、公式等）
4. 後處理和格式化結果
5. 處理多頁文件
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dots_ocr.parser import DotsOCRParser
from dots_ocr.utils.prompts import dict_promptmode_to_prompt
from dots_ocr.utils.image_utils import fetch_image
from dots_ocr.utils.layout_utils import post_process_output


class AdvancedParser:
    """高級解析器"""
    
    def __init__(self, use_hf=False):
        """初始化高級解析器"""
        self.parser = DotsOCRParser(
            use_hf=use_hf,
            output_dir="./advanced_output"
        )
    
    def parse_with_custom_resolution(self, image_path: str, min_pixels: int = None, max_pixels: int = None):
        """
        使用自訂解析度解析圖片
        
        Args:
            image_path (str): 圖片路徑
            min_pixels (int): 最小像素數
            max_pixels (int): 最大像素數
        """
        print(f"正在使用自訂解析度解析：{image_path}")
        
        # 更新解析器參數
        self.parser.min_pixels = min_pixels
        self.parser.max_pixels = max_pixels
        
        # 獲取原始圖片資訊
        image = fetch_image(image_path)
        original_size = image.size
        original_pixels = original_size[0] * original_size[1]
        
        print(f"原始尺寸：{original_size[0]} x {original_size[1]} ({original_pixels:,} 像素)")
        
        if min_pixels:
            print(f"最小像素限制：{min_pixels:,}")
        if max_pixels:
            print(f"最大像素限制：{max_pixels:,}")
        
        results = self.parser.parse_file(
            input_path=image_path,
            prompt_mode="prompt_layout_all_en"
        )
        
        if results:
            result = results[0]
            processed_size = (result['input_width'], result['input_height'])
            processed_pixels = processed_size[0] * processed_size[1]
            
            print(f"處理後尺寸：{processed_size[0]} x {processed_size[1]} ({processed_pixels:,} 像素)")
            print(f"縮放比例：{(processed_pixels / original_pixels):.2f}")
        
        return results
    
    def extract_tables_only(self, image_path: str) -> List[Dict]:
        """
        只提取表格
        
        Args:
            image_path (str): 圖片路徑
            
        Returns:
            List[Dict]: 表格列表
        """
        print(f"正在提取表格：{image_path}")
        
        results = self.parser.parse_file(
            input_path=image_path,
            prompt_mode="prompt_layout_all_en"
        )
        
        tables = []
        if results and 'layout_info_path' in results[0]:
            with open(results[0]['layout_info_path'], 'r', encoding='utf-8') as f:
                layout_data = json.load(f)
            
            # 篩選表格
            for item in layout_data:
                if item.get('category') == 'Table':
                    tables.append(item)
        
        print(f"找到 {len(tables)} 個表格")
        return tables
    
    def extract_formulas_only(self, image_path: str) -> List[Dict]:
        """
        只提取公式
        
        Args:
            image_path (str): 圖片路徑
            
        Returns:
            List[Dict]: 公式列表
        """
        print(f"正在提取公式：{image_path}")
        
        results = self.parser.parse_file(
            input_path=image_path,
            prompt_mode="prompt_layout_all_en"
        )
        
        formulas = []
        if results and 'layout_info_path' in results[0]:
            with open(results[0]['layout_info_path'], 'r', encoding='utf-8') as f:
                layout_data = json.load(f)
            
            # 篩選公式
            for item in layout_data:
                if item.get('category') == 'Formula':
                    formulas.append(item)
        
        print(f"找到 {len(formulas)} 個公式")
        return formulas
    
    def extract_structured_content(self, image_path: str) -> Dict[str, List]:
        """
        提取結構化內容（按類型分類）
        
        Args:
            image_path (str): 圖片路徑
            
        Returns:
            Dict[str, List]: 按類型分類的內容
        """
        print(f"正在提取結構化內容：{image_path}")
        
        results = self.parser.parse_file(
            input_path=image_path,
            prompt_mode="prompt_layout_all_en"
        )
        
        structured_content = {
            'Title': [],
            'Text': [],
            'Table': [],
            'Formula': [],
            'List-item': [],
            'Caption': [],
            'Section-header': [],
            'Picture': [],
            'Footnote': [],
            'Page-header': [],
            'Page-footer': []
        }
        
        if results and 'layout_info_path' in results[0]:
            with open(results[0]['layout_info_path'], 'r', encoding='utf-8') as f:
                layout_data = json.load(f)
            
            # 按類型分類
            for item in layout_data:
                category = item.get('category', 'Unknown')
                if category in structured_content:
                    structured_content[category].append(item)
        
        # 列印統計
        print("內容統計：")
        for category, items in structured_content.items():
            if items:
                print(f"  {category}: {len(items)} 個")
        
        return structured_content
    
    def convert_tables_to_html(self, tables: List[Dict]) -> List[str]:
        """
        將表格轉換為 HTML 格式
        
        Args:
            tables (List[Dict]): 表格列表
            
        Returns:
            List[str]: HTML 表格列表
        """
        html_tables = []
        
        for i, table in enumerate(tables):
            text = table.get('text', '')
            
            # 如果已經是 HTML 格式，直接使用
            if text.strip().startswith('<table'):
                html_tables.append(text)
            else:
                # 嘗試轉換為 HTML
                html_table = f"""
                <table>
                <caption>Table {i+1}</caption>
                <tbody>
                {text}
                </tbody>
                </table>
                """
                html_tables.append(html_table)
        
        return html_tables
    
    def extract_formulas_latex(self, formulas: List[Dict]) -> List[str]:
        """
        提取 LaTeX 格式的公式
        
        Args:
            formulas (List[Dict]): 公式列表
            
        Returns:
            List[str]: LaTeX 公式列表
        """
        latex_formulas = []
        
        for formula in formulas:
            text = formula.get('text', '')
            
            # 清理 LaTeX 格式
            if text.strip():
                # 移除可能的包裝標記
                clean_text = text.strip()
                if clean_text.startswith('$') and clean_text.endswith('$'):
                    clean_text = clean_text[1:-1]
                elif clean_text.startswith('$$') and clean_text.endswith('$$'):
                    clean_text = clean_text[2:-2]
                
                latex_formulas.append(clean_text)
        
        return latex_formulas
    
    def analyze_reading_order(self, layout_data: List[Dict]) -> List[Dict]:
        """
        分析閱讀順序
        
        Args:
            layout_data (List[Dict]): 版面資料
            
        Returns:
            List[Dict]: 按閱讀順序排序的元素
        """
        print("分析閱讀順序...")
        
        # 按位置排序（從上到下，從左到右）
        def sort_key(item):
            bbox = item.get('bbox', [0, 0, 0, 0])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                # 主要按 y 座標排序，次要按 x 座標排序
                return (y1, x1)
            return (0, 0)
        
        sorted_data = sorted(layout_data, key=sort_key)
        
        # 添加閱讀順序索引
        for i, item in enumerate(sorted_data):
            item['reading_order'] = i + 1
        
        return sorted_data
    
    def save_results_multiple_formats(self, results: List[Dict], base_name: str):
        """
        保存多種格式的結果
        
        Args:
            results (List[Dict]): 解析結果
            base_name (str): 基礎檔名
        """
        if not results:
            return
        
        result = results[0]
        
        # 讀取版面資料
        if 'layout_info_path' not in result:
            return
        
        with open(result['layout_info_path'], 'r', encoding='utf-8') as f:
            layout_data = json.load(f)
        
        # 1. 保存 JSON 格式（已有）
        print(f"✓ JSON 格式已保存：{result['layout_info_path']}")
        
        # 2. 保存 CSV 格式
        csv_path = f"{base_name}.csv"
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("category,bbox,text\n")
            for item in layout_data:
                category = item.get('category', '')
                bbox = str(item.get('bbox', []))
                text = item.get('text', '').replace('"', '""')  # 轉義雙引號
                f.write(f'"{category}","{bbox}","{text}"\n')
        
        print(f"✓ CSV 格式已保存：{csv_path}")
        
        # 3. 保存結構化文字格式
        txt_path = f"{base_name}_structured.txt"
        structured = self.extract_structured_content(result['file_path'])
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            for category, items in structured.items():
                if items:
                    f.write(f"\n=== {category} ===\n")
                    for i, item in enumerate(items, 1):
                        text = item.get('text', '')
                        if text.strip():
                            f.write(f"{i}. {text}\n")
        
        print(f"✓ 結構化文字格式已保存：{txt_path}")


def main():
    """高級解析功能示例"""
    print("=== 高級解析功能範例 ===")
    
    # 創建高級解析器
    parser = AdvancedParser(use_hf=False)
    
    image_path = "../demo/demo_image1.jpg"
    if not os.path.exists(image_path):
        print(f"找不到測試圖片：{image_path}")
        return
    
    try:
        # 1. 自訂解析度解析
        print("\n--- 1. 自訂解析度解析 ---")
        
        # 高解析度處理
        high_res_results = parser.parse_with_custom_resolution(
            image_path,
            min_pixels=1000000,  # 1M 像素
            max_pixels=10000000  # 10M 像素
        )
        
        # 2. 提取特定內容類型
        print("\n--- 2. 提取特定內容類型 ---")
        
        # 提取表格
        tables = parser.extract_tables_only(image_path)
        if tables:
            html_tables = parser.convert_tables_to_html(tables)
            print(f"轉換了 {len(html_tables)} 個 HTML 表格")
        
        # 提取公式
        formulas = parser.extract_formulas_only(image_path)
        if formulas:
            latex_formulas = parser.extract_formulas_latex(formulas)
            print(f"提取了 {len(latex_formulas)} 個 LaTeX 公式")
        
        # 3. 結構化內容提取
        print("\n--- 3. 結構化內容提取 ---")
        structured_content = parser.extract_structured_content(image_path)
        
        # 4. 閱讀順序分析
        print("\n--- 4. 閱讀順序分析 ---")
        if high_res_results and 'layout_info_path' in high_res_results[0]:
            with open(high_res_results[0]['layout_info_path'], 'r', encoding='utf-8') as f:
                layout_data = json.load(f)
            
            ordered_data = parser.analyze_reading_order(layout_data)
            print(f"按閱讀順序排序了 {len(ordered_data)} 個元素")
        
        # 5. 多格式保存
        print("\n--- 5. 多格式保存 ---")
        if high_res_results:
            parser.save_results_multiple_formats(
                high_res_results,
                "./advanced_output/demo_image1_advanced"
            )
        
        print(f"\n所有結果已保存到：{parser.parser.output_dir}")
        
    except Exception as e:
        print(f"處理過程中發生錯誤：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()