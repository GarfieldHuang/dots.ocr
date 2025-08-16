#!/usr/bin/env python3
"""
指定區域 OCR 範例 - 展示 grounding OCR 功能

這個範例展示如何：
1. 使用座標指定 OCR 區域
2. 批次處理多個區域
3. 互動式選擇區域
4. 結合版面檢測和區域 OCR
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageDraw

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dots_ocr.parser import DotsOCRParser
from dots_ocr.utils.image_utils import fetch_image


class GroundingOCRProcessor:
    """指定區域 OCR 處理器"""
    
    def __init__(self, use_hf=False):
        """初始化處理器"""
        self.parser = DotsOCRParser(
            use_hf=use_hf,
            output_dir="./grounding_output"
        )
    
    def ocr_region(self, image_path: str, bbox: List[int]) -> str:
        """
        對指定區域進行 OCR
        
        Args:
            image_path (str): 圖片路徑
            bbox (List[int]): 邊界框 [x1, y1, x2, y2]
            
        Returns:
            str: OCR 結果
        """
        print(f"正在對區域 {bbox} 進行 OCR...")
        
        results = self.parser.parse_file(
            input_path=image_path,
            prompt_mode="prompt_grounding_ocr",
            bbox=bbox
        )
        
        if results and 'md_content_path' in results[0]:
            with open(results[0]['md_content_path'], 'r', encoding='utf-8') as f:
                return f.read().strip()
        
        return ""
    
    def ocr_multiple_regions(self, image_path: str, bboxes: List[List[int]]) -> List[Tuple[List[int], str]]:
        """
        對多個區域進行 OCR
        
        Args:
            image_path (str): 圖片路徑
            bboxes (List[List[int]]): 邊界框列表
            
        Returns:
            List[Tuple[List[int], str]]: 區域和對應文字的列表
        """
        results = []
        
        for i, bbox in enumerate(bboxes, 1):
            print(f"處理區域 {i}/{len(bboxes)}: {bbox}")
            
            try:
                text = self.ocr_region(image_path, bbox)
                results.append((bbox, text))
                print(f"✓ 提取文字：{text[:50]}...")
                
            except Exception as e:
                print(f"✗ 區域 {bbox} 處理失敗：{e}")
                results.append((bbox, ""))
        
        return results
    
    def detect_then_ocr(self, image_path: str, target_categories: List[str] = None) -> List[Tuple[List[int], str, str]]:
        """
        先檢測版面，再對指定類型的區域進行 OCR
        
        Args:
            image_path (str): 圖片路徑
            target_categories (List[str]): 目標類別列表
            
        Returns:
            List[Tuple[List[int], str, str]]: (bbox, category, text) 的列表
        """
        if target_categories is None:
            target_categories = ['Text', 'Title', 'Table', 'Formula']
        
        print(f"正在檢測版面並OCR類別：{target_categories}")
        
        # 1. 先進行版面檢測
        detection_results = self.parser.parse_file(
            input_path=image_path,
            prompt_mode="prompt_layout_only_en"
        )
        
        if not detection_results or 'layout_info_path' not in detection_results[0]:
            print("版面檢測失敗")
            return []
        
        # 2. 讀取檢測結果
        with open(detection_results[0]['layout_info_path'], 'r', encoding='utf-8') as f:
            layout_data = json.load(f)
        
        # 3. 篩選目標類別
        target_regions = []
        for item in layout_data:
            category = item.get('category', '')
            bbox = item.get('bbox', [])
            
            if category in target_categories and len(bbox) >= 4:
                target_regions.append((bbox, category))
        
        print(f"找到 {len(target_regions)} 個目標區域")
        
        # 4. 對每個區域進行 OCR
        results = []
        for i, (bbox, category) in enumerate(target_regions, 1):
            print(f"OCR 區域 {i}/{len(target_regions)} ({category}): {bbox}")
            
            try:
                text = self.ocr_region(image_path, bbox)
                results.append((bbox, category, text))
                print(f"✓ {category} 文字：{text[:50]}...")
                
            except Exception as e:
                print(f"✗ 區域 OCR 失敗：{e}")
                results.append((bbox, category, ""))
        
        return results
    
    def visualize_regions(self, image_path: str, regions: List[Tuple], output_path: str = None):
        """
        視覺化標註區域
        
        Args:
            image_path (str): 圖片路徑
            regions (List[Tuple]): 區域列表 (bbox, text) 或 (bbox, category, text)
            output_path (str): 輸出路徑
        """
        # 載入圖片
        image = fetch_image(image_path)
        draw = ImageDraw.Draw(image)
        
        # 定義顏色
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, region in enumerate(regions):
            if len(region) == 2:
                bbox, text = region
                label = f"Region {i+1}"
            elif len(region) == 3:
                bbox, category, text = region
                label = f"{category} {i+1}"
            else:
                continue
            
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                color = colors[i % len(colors)]
                
                # 繪製邊界框
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # 繪製標籤
                draw.text((x1, y1-20), label, fill=color)
        
        # 保存結果
        if output_path is None:
            output_path = f"./grounding_output/visualized_{os.path.basename(image_path)}"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        print(f"✓ 視覺化結果已保存：{output_path}")
        
        return output_path
    
    def save_grounding_results(self, image_path: str, results: List[Tuple], output_path: str = None):
        """
        保存 grounding OCR 結果
        
        Args:
            image_path (str): 圖片路徑
            results (List[Tuple]): 結果列表
            output_path (str): 輸出路徑
        """
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"./grounding_output/{base_name}_grounding_results.json"
        
        # 準備保存的資料
        save_data = {
            "source_image": image_path,
            "total_regions": len(results),
            "results": []
        }
        
        for i, result in enumerate(results):
            if len(result) == 2:
                bbox, text = result
                region_data = {
                    "region_id": i + 1,
                    "bbox": bbox,
                    "text": text
                }
            elif len(result) == 3:
                bbox, category, text = result
                region_data = {
                    "region_id": i + 1,
                    "bbox": bbox,
                    "category": category,
                    "text": text
                }
            else:
                continue
            
            save_data["results"].append(region_data)
        
        # 保存 JSON
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 結果已保存：{output_path}")
        
        # 同時保存 TXT 格式
        txt_path = output_path.replace('.json', '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Grounding OCR Results for: {image_path}\n")
            f.write("="*50 + "\n\n")
            
            for i, result in enumerate(results):
                if len(result) == 2:
                    bbox, text = result
                    f.write(f"Region {i+1}:\n")
                    f.write(f"  Bbox: {bbox}\n")
                    f.write(f"  Text: {text}\n\n")
                elif len(result) == 3:
                    bbox, category, text = result
                    f.write(f"Region {i+1} ({category}):\n")
                    f.write(f"  Bbox: {bbox}\n")
                    f.write(f"  Text: {text}\n\n")
        
        print(f"✓ 文字結果已保存：{txt_path}")


def main():
    """指定區域 OCR 示例"""
    print("=== 指定區域 OCR 範例 ===")
    
    # 創建處理器
    processor = GroundingOCRProcessor(use_hf=False)
    
    image_path = "../demo/demo_image1.jpg"
    if not os.path.exists(image_path):
        print(f"找不到測試圖片：{image_path}")
        return
    
    try:
        # 1. 單個區域 OCR
        print("\n--- 1. 單個區域 OCR ---")
        
        # 假設我們要 OCR 圖片中央區域
        image = fetch_image(image_path)
        width, height = image.size
        
        center_bbox = [
            width // 4,      # x1
            height // 4,     # y1  
            3 * width // 4,  # x2
            3 * height // 4  # y2
        ]
        
        center_text = processor.ocr_region(image_path, center_bbox)
        print(f"中央區域文字：{center_text[:100]}...")
        
        # 2. 多個區域 OCR
        print("\n--- 2. 多個區域 OCR ---")
        
        # 定義多個測試區域
        test_regions = [
            [0, 0, width//2, height//2],           # 左上
            [width//2, 0, width, height//2],       # 右上
            [0, height//2, width//2, height],      # 左下
            [width//2, height//2, width, height],  # 右下
        ]
        
        multi_results = processor.ocr_multiple_regions(image_path, test_regions)
        
        print(f"處理了 {len(multi_results)} 個區域：")
        for i, (bbox, text) in enumerate(multi_results, 1):
            print(f"  區域 {i}: {len(text)} 字符")
        
        # 3. 檢測後 OCR
        print("\n--- 3. 版面檢測 + 區域 OCR ---")
        
        detect_ocr_results = processor.detect_then_ocr(
            image_path,
            target_categories=['Text', 'Title', 'Table']
        )
        
        print(f"檢測並OCR了 {len(detect_ocr_results)} 個區域")
        
        # 4. 視覺化結果
        print("\n--- 4. 視覺化結果 ---")
        
        # 視覺化多區域結果
        vis_path1 = processor.visualize_regions(
            image_path,
            multi_results,
            "./grounding_output/multi_regions_visualization.jpg"
        )
        
        # 視覺化檢測+OCR結果
        vis_path2 = processor.visualize_regions(
            image_path,
            detect_ocr_results,
            "./grounding_output/detect_ocr_visualization.jpg"
        )
        
        # 5. 保存結果
        print("\n--- 5. 保存結果 ---")
        
        processor.save_grounding_results(
            image_path,
            multi_results,
            "./grounding_output/multi_regions_results.json"
        )
        
        processor.save_grounding_results(
            image_path,
            detect_ocr_results,
            "./grounding_output/detect_ocr_results.json"
        )
        
        # 6. 實用技巧展示
        print("\n--- 6. 實用技巧 ---")
        
        print("技巧 1：只 OCR 標題和文字")
        title_text_results = processor.detect_then_ocr(
            image_path,
            target_categories=['Title', 'Text', 'Section-header']
        )
        print(f"提取了 {len(title_text_results)} 個標題和文字區域")
        
        print("\n技巧 2：只處理表格")
        table_results = processor.detect_then_ocr(
            image_path,
            target_categories=['Table']
        )
        print(f"提取了 {len(table_results)} 個表格")
        
        print(f"\n所有結果已保存到：{processor.parser.output_dir}")
        
    except Exception as e:
        print(f"處理過程中發生錯誤：{e}")
        import traceback
        traceback.print_exc()


def show_grounding_tips():
    """顯示 grounding OCR 使用技巧"""
    print("\n=== Grounding OCR 使用技巧 ===")
    
    print("1. 座標格式：")
    print("   bbox = [x1, y1, x2, y2]  # 左上角和右下角座標")
    
    print("\n2. 獲取座標的方法：")
    print("   - 使用圖片編輯軟體測量")
    print("   - 先用版面檢測獲取候選區域")
    print("   - 使用互動式標註工具")
    
    print("\n3. 常用區域模式：")
    print("   - 圖片四等分：適合處理簡單版面")
    print("   - 條帶掃描：適合表格或列表")
    print("   - 精確定位：適合已知位置的特定內容")
    
    print("\n4. 最佳實踐：")
    print("   - 區域不要太小（至少包含幾個字符）")
    print("   - 避免跨越多個版面元素")
    print("   - 考慮文字的邊距")
    
    print("\n5. 故障排除：")
    print("   - 座標超出圖片範圍 → 檢查座標計算")
    print("   - OCR 結果為空 → 檢查區域是否包含文字")
    print("   - 結果不準確 → 調整區域邊界")


if __name__ == "__main__":
    main()
    show_grounding_tips()