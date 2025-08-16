#!/usr/bin/env python3
"""
多語言文件解析範例 - 展示 dots.ocr 的多語言支援

這個範例展示如何：
1. 處理不同語言的文件
2. 語言檢測和識別
3. 混合語言文件處理
4. 特殊字符處理
5. 多語言結果格式化
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dots_ocr.parser import DotsOCRParser
from dots_ocr.utils.image_utils import fetch_image


class MultilingualProcessor:
    """多語言處理器"""
    
    def __init__(self, use_hf=False):
        """初始化多語言處理器"""
        self.parser = DotsOCRParser(
            use_hf=use_hf,
            output_dir="./multilingual_output"
        )
        
        # 語言檢測規則（簡單的字符範圍檢測）
        self.language_patterns = {
            'chinese_simplified': r'[\u4e00-\u9fff]',
            'chinese_traditional': r'[\u4e00-\u9fff]',  # 需要更精確的檢測
            'japanese': r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]',
            'korean': r'[\uac00-\ud7af]',
            'arabic': r'[\u0600-\u06ff]',
            'hebrew': r'[\u0590-\u05ff]',
            'thai': r'[\u0e00-\u0e7f]',
            'vietnamese': r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]',
            'cyrillic': r'[\u0400-\u04ff]',  # 俄語等
            'devanagari': r'[\u0900-\u097f]',  # 印地語等
            'latin': r'[a-zA-Z]',
            'digits': r'[0-9]',
            'punctuation': r'[.,;:!?()"\'-]'
        }
    
    def detect_languages(self, text: str) -> Dict[str, float]:
        """
        檢測文字中的語言
        
        Args:
            text (str): 待檢測文字
            
        Returns:
            Dict[str, float]: 語言及其比例
        """
        if not text.strip():
            return {}
        
        total_chars = len(text)
        language_counts = {}
        
        for lang, pattern in self.language_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            count = len(matches)
            if count > 0:
                language_counts[lang] = count / total_chars
        
        return language_counts
    
    def analyze_multilingual_content(self, results: List[Dict]) -> Dict:
        """
        分析多語言內容
        
        Args:
            results (List[Dict]): 解析結果
            
        Returns:
            Dict: 多語言分析結果
        """
        if not results or 'layout_info_path' not in results[0]:
            return {}
        
        # 讀取版面資訊
        with open(results[0]['layout_info_path'], 'r', encoding='utf-8') as f:
            layout_data = json.load(f)
        
        analysis = {
            'total_elements': len(layout_data),
            'elements_by_language': {},
            'language_statistics': {},
            'mixed_language_elements': [],
            'dominant_languages': []
        }
        
        all_detected_languages = Counter()
        
        # 分析每個版面元素
        for i, element in enumerate(layout_data):
            text = element.get('text', '')
            category = element.get('category', 'Unknown')
            
            if text.strip():
                # 檢測語言
                languages = self.detect_languages(text)
                
                element_info = {
                    'element_id': i,
                    'category': category,
                    'text_length': len(text),
                    'detected_languages': languages,
                    'bbox': element.get('bbox', [])
                }
                
                # 統計語言使用
                for lang, ratio in languages.items():
                    all_detected_languages[lang] += ratio
                    
                    if lang not in analysis['elements_by_language']:
                        analysis['elements_by_language'][lang] = []
                    analysis['elements_by_language'][lang].append(element_info)
                
                # 檢測混合語言元素
                if len(languages) > 2:  # 超過2種語言（不包括數字和標點）
                    filtered_langs = {k: v for k, v in languages.items() 
                                    if k not in ['digits', 'punctuation']}
                    if len(filtered_langs) > 1:
                        analysis['mixed_language_elements'].append(element_info)
        
        # 計算語言統計
        total_language_score = sum(all_detected_languages.values())
        if total_language_score > 0:
            for lang, score in all_detected_languages.items():
                analysis['language_statistics'][lang] = {
                    'ratio': score / total_language_score,
                    'element_count': len(analysis['elements_by_language'].get(lang, []))
                }
        
        # 確定主要語言
        main_languages = sorted(
            analysis['language_statistics'].items(),
            key=lambda x: x[1]['ratio'],
            reverse=True
        )
        
        # 過濾掉數字和標點，取前3種語言
        analysis['dominant_languages'] = [
            lang for lang, _ in main_languages
            if lang not in ['digits', 'punctuation']
        ][:3]
        
        return analysis
    
    def process_multilingual_document(self, image_path: str) -> Dict:
        """
        處理多語言文件
        
        Args:
            image_path (str): 圖片路徑
            
        Returns:
            Dict: 處理結果
        """
        print(f"正在處理多語言文件：{image_path}")
        
        # 解析文件
        results = self.parser.parse_file(
            input_path=image_path,
            prompt_mode="prompt_layout_all_en"
        )
        
        if not results:
            return {'error': '解析失敗'}
        
        # 多語言分析
        multilingual_analysis = self.analyze_multilingual_content(results)
        
        # 組合結果
        processing_result = {
            'file_path': image_path,
            'parsing_results': results[0],
            'multilingual_analysis': multilingual_analysis
        }
        
        return processing_result
    
    def create_language_report(self, analysis: Dict, output_path: str = None):
        """
        創建語言分析報告
        
        Args:
            analysis (Dict): 多語言分析結果
            output_path (str): 輸出路徑
        """
        if output_path is None:
            output_path = "./multilingual_output/language_report.txt"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("多語言文件分析報告\\n")
            f.write("=" * 40 + "\\n\\n")
            
            # 基本統計
            f.write(f"版面元素總數：{analysis.get('total_elements', 0)}\\n")
            f.write(f"混合語言元素：{len(analysis.get('mixed_language_elements', []))}\\n")
            f.write(f"主要語言：{', '.join(analysis.get('dominant_languages', []))}\\n\\n")
            
            # 語言統計
            f.write("語言統計：\\n")
            f.write("-" * 20 + "\\n")
            
            lang_stats = analysis.get('language_statistics', {})
            for lang, stats in sorted(lang_stats.items(), key=lambda x: x[1]['ratio'], reverse=True):
                if lang not in ['digits', 'punctuation']:
                    f.write(f"{lang:20s}: {stats['ratio']*100:5.1f}% ({stats['element_count']} 個元素)\\n")
            
            # 混合語言元素詳情
            if analysis.get('mixed_language_elements'):
                f.write("\\n混合語言元素：\\n")
                f.write("-" * 20 + "\\n")
                
                for element in analysis['mixed_language_elements']:
                    f.write(f"元素 {element['element_id']} ({element['category']}):\\n")
                    for lang, ratio in element['detected_languages'].items():
                        if lang not in ['digits', 'punctuation'] and ratio > 0.1:
                            f.write(f"  {lang}: {ratio*100:.1f}%\\n")
                    f.write("\\n")
        
        print(f"✓ 語言分析報告已保存：{output_path}")
    
    def extract_by_language(self, analysis: Dict, target_language: str) -> List[Dict]:
        """
        提取特定語言的內容
        
        Args:
            analysis (Dict): 多語言分析結果
            target_language (str): 目標語言
            
        Returns:
            List[Dict]: 該語言的元素列表
        """
        elements_by_lang = analysis.get('elements_by_language', {})
        return elements_by_lang.get(target_language, [])
    
    def save_multilingual_results(self, processing_result: Dict, base_name: str):
        """
        保存多語言處理結果
        
        Args:
            processing_result (Dict): 處理結果
            base_name (str): 基礎檔名
        """
        analysis = processing_result.get('multilingual_analysis', {})
        
        # 1. 保存完整分析結果
        analysis_path = f"./multilingual_output/{base_name}_analysis.json"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(processing_result, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 完整分析結果已保存：{analysis_path}")
        
        # 2. 保存語言報告
        self.create_language_report(analysis, f"./multilingual_output/{base_name}_language_report.txt")
        
        # 3. 按語言分類保存內容
        elements_by_lang = analysis.get('elements_by_language', {})
        
        for lang, elements in elements_by_lang.items():
            if lang not in ['digits', 'punctuation'] and elements:
                lang_file = f"./multilingual_output/{base_name}_{lang}.txt"
                
                with open(lang_file, 'w', encoding='utf-8') as f:
                    f.write(f"{lang.upper()} 內容\\n")
                    f.write("=" * 30 + "\\n\\n")
                    
                    for i, element in enumerate(elements, 1):
                        f.write(f"{i}. [{element['category']}] (元素 {element['element_id']})\\n")
                        
                        # 從原始版面資料中獲取文字
                        if 'parsing_results' in processing_result:
                            layout_path = processing_result['parsing_results'].get('layout_info_path')
                            if layout_path and os.path.exists(layout_path):
                                with open(layout_path, 'r', encoding='utf-8') as layout_f:
                                    layout_data = json.load(layout_f)
                                    if element['element_id'] < len(layout_data):
                                        text = layout_data[element['element_id']].get('text', '')
                                        f.write(f"{text}\\n\\n")
                
                print(f"✓ {lang} 內容已保存：{lang_file}")


def create_sample_multilingual_content():
    """創建多語言範例內容（如果沒有現成的多語言圖片）"""
    print("\\n=== 多語言範例文字 ===")
    
    samples = {
        '英文': "Hello, this is a sample English text.",
        '中文': "你好，這是一個中文範例文字。",
        '日文': "こんにちは、これは日本語のサンプルテキストです。",
        '韓文': "안녕하세요, 이것은 한국어 샘플 텍스트입니다.",
        '阿拉伯文': "مرحبا، هذا نص عربي نموذجي.",
        '俄文': "Привет, это образец русского текста.",
        '泰文': "สวัสดี นี่คือข้อความตัวอย่างภาษาไทย",
        '越南文': "Xin chào, đây là văn bản tiếng Việt mẫu."
    }
    
    processor = MultilingualProcessor()
    
    for lang, text in samples.items():
        detected = processor.detect_languages(text)
        print(f"{lang:10s}: {text}")
        print(f"{'':10s}  檢測結果: {detected}")
        print()


def main():
    """多語言文件解析示例"""
    print("=== 多語言文件解析範例 ===")
    
    # 創建多語言處理器
    processor = MultilingualProcessor(use_hf=False)
    
    # 測試圖片
    test_images = [
        "../demo/demo_image1.jpg",
        # 可以添加更多多語言圖片
    ]
    
    # 過濾存在的圖片
    existing_images = [img for img in test_images if os.path.exists(img)]
    
    if not existing_images:
        print("沒有找到測試圖片，顯示語言檢測範例...")
        create_sample_multilingual_content()
        return
    
    try:
        for i, image_path in enumerate(existing_images, 1):
            print(f"\\n--- 處理圖片 {i}: {image_path} ---")
            
            # 處理多語言文件
            result = processor.process_multilingual_document(image_path)
            
            if 'error' in result:
                print(f"✗ 處理失敗：{result['error']}")
                continue
            
            analysis = result.get('multilingual_analysis', {})
            
            # 顯示基本分析結果
            print(f"版面元素：{analysis.get('total_elements', 0)} 個")
            print(f"主要語言：{', '.join(analysis.get('dominant_languages', []))}")
            print(f"混合語言元素：{len(analysis.get('mixed_language_elements', []))} 個")
            
            # 顯示語言統計
            lang_stats = analysis.get('language_statistics', {})
            if lang_stats:
                print("\\n語言分布：")
                for lang, stats in sorted(lang_stats.items(), key=lambda x: x[1]['ratio'], reverse=True):
                    if lang not in ['digits', 'punctuation'] and stats['ratio'] > 0.05:
                        print(f"  {lang}: {stats['ratio']*100:.1f}%")
            
            # 提取特定語言內容
            if 'chinese_simplified' in analysis.get('elements_by_language', {}):
                chinese_elements = processor.extract_by_language(analysis, 'chinese_simplified')
                print(f"\\n中文元素：{len(chinese_elements)} 個")
            
            # 保存結果
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            processor.save_multilingual_results(result, base_name)
        
        print(f"\\n所有結果已保存到：{processor.parser.output_dir}")
        
        # 顯示語言檢測範例
        create_sample_multilingual_content()
        
    except Exception as e:
        print(f"處理過程中發生錯誤：{e}")
        import traceback
        traceback.print_exc()


def show_multilingual_tips():
    """顯示多語言處理建議"""
    print("\\n=== 多語言處理建議 ===")
    
    print("1. 語言檢測限制：")
    print("   - 基於字符範圍的簡單檢測")
    print("   - 中文繁簡體需要額外區分")
    print("   - 短文本檢測準確性較低")
    
    print("\\n2. 處理建議：")
    print("   - 預先了解文件主要語言")
    print("   - 注意文字方向（RTL語言）")
    print("   - 考慮字體和編碼問題")
    
    print("\\n3. 輸出格式：")
    print("   - 確保 UTF-8 編碼")
    print("   - 保留原始字符")
    print("   - 分語言保存結果")
    
    print("\\n4. 性能優化：")
    print("   - 複雜文檔建議分頁處理")
    print("   - 使用合適的 DPI 設置")
    print("   - 考慮文字密度影響")


if __name__ == "__main__":
    main()
    show_multilingual_tips()