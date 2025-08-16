#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增強版 PDF 解析範例 - 基於 demo/demo_gradio.py 的高階 API

本範例使用 DotsOCRParser 的高階 API，與 demo 中的實現保持一致，
能夠解析包含混合中英文、表格、圖片、公式等複雜內容的 PDF 文檔。

特色功能：
- 使用經過驗證的高階 API
- 自動混合語言識別
- 結構化內容提取
- 批量處理支援
- 與 demo 保持一致的解析品質
"""

import os
import sys
import json
import tempfile
import uuid
import shutil
import re
from typing import Dict, List, Any, Optional
from pathlib import Path

# 添加專案根目錄到 Python 路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dots_ocr.parser import DotsOCRParser
from dots_ocr.utils.doc_utils import load_images_from_pdf
from PIL import Image


class EnhancedPDFProcessor:
    """增強版 PDF 處理器，基於 demo 的高階 API"""
    
    def __init__(self, use_hf: bool = False, num_threads: int = 4, dpi: int = 200):
        """
        初始化 PDF 處理器
        
        Args:
            use_hf (bool): 是否使用 HuggingFace 模型
            num_threads (int): 處理線程數
            dpi (int): PDF 渲染 DPI
        """
        print(f"🔧 初始化增強版 PDF 處理器 (DPI: {dpi}, 線程: {num_threads})")
        
        self.parser = DotsOCRParser(
            use_hf=use_hf,
            num_thread=num_threads,
            dpi=dpi,
            output_dir="./pdf_output"
        )
        
        self.language_patterns = {
            'chinese': r'[\u4e00-\u9fff]',
            'english': r'[a-zA-Z]',
            'numbers': r'[0-9]',
            'punctuation': r'[.,;:!?()"\'-]'
        }
    
    def create_temp_session_dir(self) -> tuple:
        """創建臨時會話目錄"""
        session_id = uuid.uuid4().hex[:8]
        temp_dir = os.path.join(tempfile.gettempdir(), f"dots_ocr_enhanced_{session_id}")
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir, session_id
    
    def parse_pdf_with_api(self, pdf_path: str, prompt_mode: str = "prompt_layout_all_en") -> Dict[str, Any]:
        """
        使用高階 API 解析 PDF（與 demo 一致）
        
        Args:
            pdf_path (str): PDF 檔案路徑
            prompt_mode (str): 解析模式
            
        Returns:
            Dict[str, Any]: 解析結果
        """
        if not os.path.exists(pdf_path):
            return {'success': False, 'error': 'PDF 檔案不存在'}
        
        print(f"📄 使用高階 API 解析 PDF：{pdf_path}")
        
        # 使用 parser 的輸出目錄而不是臨時目錄
        filename = f"enhanced_{uuid.uuid4().hex[:8]}"
        save_dir = os.path.join(self.parser.output_dir, filename)
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"📁 解析結果將保存到：{save_dir}")
        
        try:
            # 使用 DotsOCRParser 的高階 API（與 demo/demo_gradio.py 一致）
            results = self.parser.parse_pdf(
                input_path=pdf_path,
                filename=filename,
                prompt_mode=prompt_mode,
                save_dir=save_dir
            )
            
            if not results:
                return {'success': False, 'error': '解析器未返回結果'}
            
            # 處理多頁結果
            parsed_results = []
            all_md_content = []
            all_cells_data = []
            actual_files = []  # 記錄實際生成的檔案
            
            for i, result in enumerate(results):
                page_result = {
                    'page_no': result.get('page_no', i),
                    'layout_image': None,
                    'cells_data': None,
                    'md_content': None,
                    'filtered': False,
                    'file_path': pdf_path
                }
                
                # 讀取版面圖片
                if 'layout_image_path' in result and os.path.exists(result['layout_image_path']):
                    page_result['layout_image'] = Image.open(result['layout_image_path'])
                    actual_files.append(('layout_image', result['layout_image_path']))
                    print(f"✓ 找到版面圖片：{result['layout_image_path']}")
                
                # 讀取 JSON 資料
                if 'layout_info_path' in result and os.path.exists(result['layout_info_path']):
                    with open(result['layout_info_path'], 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        page_result['cells_data'] = json_data
                        
                        # 檢查是否為 filtered 頁面（包含字串資料而非字典列表）
                        if result.get('filtered', False):
                            # filtered 頁面的 JSON 檔案包含原始字串回應，跳過結構化分析
                            print(f"⚠️ 第 {i} 頁解析失敗（filtered=True），跳過結構化分析")
                        else:
                            # 正常頁面包含字典列表，可以進行結構化分析
                            if isinstance(json_data, list):
                                all_cells_data.extend(json_data)
                            else:
                                print(f"⚠️ 第 {i} 頁 JSON 格式異常，預期為列表但得到 {type(json_data)}")
                    actual_files.append(('layout_json', result['layout_info_path']))
                    print(f"✓ 找到佈局 JSON：{result['layout_info_path']}")
                
                # 讀取 Markdown 內容
                if 'md_content_path' in result and os.path.exists(result['md_content_path']):
                    with open(result['md_content_path'], 'r', encoding='utf-8') as f:
                        page_content = f.read()
                        page_result['md_content'] = page_content
                        all_md_content.append(page_content)
                    actual_files.append(('markdown', result['md_content_path']))
                    print(f"✓ 找到 Markdown：{result['md_content_path']}")
                
                page_result['filtered'] = result.get('filtered', False)
                parsed_results.append(page_result)
            
            combined_md = "\n\n---\n\n".join(all_md_content) if all_md_content else ""
            
            print(f"✅ 成功解析 {len(results)} 頁，共 {len(all_cells_data)} 個元素")
            print(f"📁 所有檔案已保存到：{save_dir}")
            
            # 顯示生成的檔案列表
            if actual_files:
                print(f"\n📋 生成的檔案列表：")
                for file_type, file_path in actual_files:
                    print(f"  {file_type}: {file_path}")
            
            return {
                'success': True,
                'parsed_results': parsed_results,
                'combined_md_content': combined_md,
                'combined_cells_data': all_cells_data,
                'temp_dir': save_dir,  # 使用實際的保存目錄
                'session_id': filename,
                'total_pages': len(results),
                'actual_files': actual_files
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def detect_content_languages(self, text: str) -> Dict[str, float]:
        """檢測文字內容的語言分布"""
        if not text.strip():
            return {}
        
        total_chars = len(text)
        language_counts = {}
        
        for lang, pattern in self.language_patterns.items():
            matches = re.findall(pattern, text)
            count = len(matches)
            if count > 0:
                language_counts[lang] = count / total_chars
        
        return language_counts
    
    def analyze_structured_content(self, cells_data: List[Dict]) -> Dict[str, Any]:
        """分析結構化內容"""
        analysis = {
            'total_elements': len(cells_data),
            'content_types': {},
            'language_stats': {
                'chinese_elements': 0,
                'english_elements': 0,
                'mixed_elements': 0,
                'total_text_elements': 0
            },
            'element_types': {
                'tables': 0,
                'images': 0,
                'formulas': 0,
                'text_blocks': 0,
                'titles': 0
            },
            'text_by_type': {
                'chinese_text': [],
                'english_text': [],
                'mixed_text': [],
                'tables': [],
                'formulas': [],
                'titles': []
            }
        }
        
        for element in cells_data:
                
            category = element.get('category', 'Unknown')
            text = element.get('text', '')
            bbox = element.get('bbox', [])
            
            # 統計內容類型
            analysis['content_types'][category] = analysis['content_types'].get(category, 0) + 1
            
            # 元素類型統計
            if category == 'Table':
                analysis['element_types']['tables'] += 1
                analysis['text_by_type']['tables'].append({
                    'text': text, 'bbox': bbox, 'category': category
                })
            elif category == 'Picture':
                analysis['element_types']['images'] += 1
            elif category == 'Formula':
                analysis['element_types']['formulas'] += 1
                analysis['text_by_type']['formulas'].append({
                    'text': text, 'bbox': bbox, 'category': category
                })
            elif category in ['Title', 'Section-header']:
                analysis['element_types']['titles'] += 1
                analysis['text_by_type']['titles'].append({
                    'text': text, 'bbox': bbox, 'category': category
                })
            elif category in ['Text', 'List-item', 'Caption']:
                analysis['element_types']['text_blocks'] += 1
                
                # 語言分析
                if text.strip():
                    analysis['language_stats']['total_text_elements'] += 1
                    languages = self.detect_content_languages(text)
                    chinese_ratio = languages.get('chinese', 0)
                    english_ratio = languages.get('english', 0)
                    
                    element_info = {
                        'text': text, 'bbox': bbox, 'category': category,
                        'chinese_ratio': chinese_ratio, 'english_ratio': english_ratio
                    }
                    
                    if chinese_ratio > 0.5 and english_ratio < 0.1:
                        analysis['language_stats']['chinese_elements'] += 1
                        analysis['text_by_type']['chinese_text'].append(element_info)
                    elif english_ratio > 0.5 and chinese_ratio < 0.1:
                        analysis['language_stats']['english_elements'] += 1
                        analysis['text_by_type']['english_text'].append(element_info)
                    elif chinese_ratio > 0.1 and english_ratio > 0.1:
                        analysis['language_stats']['mixed_elements'] += 1
                        analysis['text_by_type']['mixed_text'].append(element_info)
                    else:
                        # 默認歸類為英文
                        analysis['language_stats']['english_elements'] += 1
                        analysis['text_by_type']['english_text'].append(element_info)
        
        return analysis
    
    def save_structured_results(self, analysis: Dict[str, Any], session_id: str, output_dir: str) -> str:
        """保存結構化結果到不同檔案"""
        structured_dir = os.path.join(output_dir, f"structured_{session_id}")
        os.makedirs(structured_dir, exist_ok=True)
        
        # 保存中文內容
        if analysis['text_by_type']['chinese_text']:
            chinese_file = os.path.join(structured_dir, "chinese_content.md")
            with open(chinese_file, 'w', encoding='utf-8') as f:
                f.write("# 中文內容\n\n")
                for i, item in enumerate(analysis['text_by_type']['chinese_text'], 1):
                    f.write(f"## 內容 {i} - {item['category']}\n\n")
                    f.write(f"**位置**: {item['bbox']}\n\n")
                    f.write(f"{item['text']}\n\n")
            print(f"✓ 中文內容已保存：{chinese_file}")
        
        # 保存英文內容
        if analysis['text_by_type']['english_text']:
            english_file = os.path.join(structured_dir, "english_content.md")
            with open(english_file, 'w', encoding='utf-8') as f:
                f.write("# English Content\n\n")
                for i, item in enumerate(analysis['text_by_type']['english_text'], 1):
                    f.write(f"## Content {i} - {item['category']}\n\n")
                    f.write(f"**Position**: {item['bbox']}\n\n")
                    f.write(f"{item['text']}\n\n")
            print(f"✓ 英文內容已保存：{english_file}")
        
        # 保存混合語言內容
        if analysis['text_by_type']['mixed_text']:
            mixed_file = os.path.join(structured_dir, "mixed_language_content.md")
            with open(mixed_file, 'w', encoding='utf-8') as f:
                f.write("# 中英文混合內容 / Mixed Language Content\n\n")
                for i, item in enumerate(analysis['text_by_type']['mixed_text'], 1):
                    chinese_pct = item.get('chinese_ratio', 0) * 100
                    english_pct = item.get('english_ratio', 0) * 100
                    f.write(f"## 混合內容 {i} - {item['category']}\n\n")
                    f.write(f"**語言分布**: 中文 {chinese_pct:.1f}%, 英文 {english_pct:.1f}%\n\n")
                    f.write(f"**位置**: {item['bbox']}\n\n")
                    f.write(f"{item['text']}\n\n")
            print(f"✓ 混合語言內容已保存：{mixed_file}")
        
        # 保存表格內容
        if analysis['text_by_type']['tables']:
            table_file = os.path.join(structured_dir, "tables.html")
            with open(table_file, 'w', encoding='utf-8') as f:
                f.write("<html><head><meta charset='utf-8'><title>表格內容</title></head><body>\n")
                f.write("<h1>表格內容</h1>\n")
                for i, item in enumerate(analysis['text_by_type']['tables'], 1):
                    f.write(f"<h2>表格 {i}</h2>\n")
                    f.write(f"<p><strong>位置</strong>: {item['bbox']}</p>\n")
                    f.write(f"<div style='border: 1px solid #ccc; padding: 10px; margin: 10px 0;'>\n")
                    f.write(f"{item['text'].replace(chr(10), '<br>')}\n")
                    f.write("</div>\n")
                f.write("</body></html>")
            print(f"✓ 表格內容已保存：{table_file}")
        
        # 保存分析報告
        report_file = os.path.join(structured_dir, "analysis_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("PDF 內容分析報告\n")
            f.write("=" * 30 + "\n\n")
            
            f.write("## 總體統計\n")
            f.write(f"總元素數：{analysis['total_elements']}\n")
            f.write(f"文字元素數：{analysis['language_stats']['total_text_elements']}\n")
            f.write(f"表格數：{analysis['element_types']['tables']}\n")
            f.write(f"圖片數：{analysis['element_types']['images']}\n")
            f.write(f"公式數：{analysis['element_types']['formulas']}\n")
            f.write(f"標題數：{analysis['element_types']['titles']}\n\n")
            
            f.write("## 語言分布\n")
            f.write(f"中文元素：{analysis['language_stats']['chinese_elements']}\n")
            f.write(f"英文元素：{analysis['language_stats']['english_elements']}\n")
            f.write(f"混合語言元素：{analysis['language_stats']['mixed_elements']}\n\n")
            
            f.write("## 元素類型分布\n")
            for content_type, count in analysis['content_types'].items():
                f.write(f"{content_type}: {count}\n")
        
        print(f"✓ 分析報告已保存：{report_file}")
        print(f"✅ 所有結構化內容已保存到：{structured_dir}")
        
        return structured_dir
    
    def process_pdf_enhanced(self, pdf_path: str, prompt_mode: str = "prompt_layout_all_en") -> Dict[str, Any]:
        """
        增強版 PDF 處理（完整流程）
        
        Args:
            pdf_path (str): PDF 檔案路徑
            prompt_mode (str): 解析模式
            
        Returns:
            Dict[str, Any]: 完整處理結果
        """
        print(f"🚀 啟動增強版 PDF 處理：{os.path.basename(pdf_path)}")
        
        # 第一步：使用高階 API 解析
        parse_result = self.parse_pdf_with_api(pdf_path, prompt_mode)
        
        if not parse_result['success']:
            return parse_result
        
        # 第二步：結構化內容分析
        print("📊 進行結構化內容分析...")
        analysis = self.analyze_structured_content(parse_result['combined_cells_data'])
        
        # 第三步：保存結構化結果
        print("💾 保存結構化結果...")
        structured_dir = self.save_structured_results(
            analysis, 
            parse_result['session_id'], 
            parse_result['temp_dir']
        )
        
        # 第四步：生成摘要
        summary = {
            'file_name': os.path.basename(pdf_path),
            'total_pages': parse_result['total_pages'],
            'total_elements': analysis['total_elements'],
            'chinese_elements': analysis['language_stats']['chinese_elements'],
            'english_elements': analysis['language_stats']['english_elements'],
            'mixed_elements': analysis['language_stats']['mixed_elements'],
            'tables': analysis['element_types']['tables'],
            'images': analysis['element_types']['images'],
            'formulas': analysis['element_types']['formulas'],
            'structured_output_dir': structured_dir
        }
        
        print(f"\n📋 處理摘要：")
        print(f"  檔案：{summary['file_name']}")
        print(f"  頁數：{summary['total_pages']}")
        print(f"  總元素：{summary['total_elements']}")
        print(f"  中文元素：{summary['chinese_elements']}")
        print(f"  英文元素：{summary['english_elements']}")
        print(f"  混合語言元素：{summary['mixed_elements']}")
        print(f"  表格：{summary['tables']}")
        print(f"  圖片：{summary['images']}")
        print(f"  公式：{summary['formulas']}")
        
        return {
            'success': True,
            'summary': summary,
            'analysis': analysis,
            'parse_result': parse_result,
            'structured_dir': structured_dir
        }


def demo_enhanced_pdf_processing():
    """演示增強版 PDF 處理功能"""
    print("🎯 增強版 PDF 處理演示")
    print("基於 demo/demo_gradio.py 的高階 API")
    print("=" * 50)
    
    # 創建處理器
    processor = EnhancedPDFProcessor(
        use_hf=False,  # 使用 vLLM 以獲得更好的性能
        num_threads=4,
        dpi=200  # 與 demo 一致的 DPI 設定
    )
    
    # 測試檔案列表
    test_files = [
        "../demo/demo_pdf1.pdf",
        "110120240613G01.pdf"
    ]
    
    print("\n📄 可用的測試檔案：")
    available_files = []
    for i, file_path in enumerate(test_files, 1):
        if os.path.exists(file_path):
            print(f"  {i}. {file_path} ✅")
            available_files.append(file_path)
        else:
            print(f"  {i}. {file_path} ❌")
    
    if not available_files:
        print("\n❌ 沒有找到可用的測試檔案")
        return
    
    # 用戶選擇
    choice = input(f"\n請選擇檔案 (1-{len(available_files)}) 或輸入自定義路徑: ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(available_files):
        pdf_path = available_files[int(choice) - 1]
    else:
        pdf_path = choice
        if not os.path.exists(pdf_path):
            print(f"❌ 檔案不存在：{pdf_path}")
            return
    
    print("\n🔧 解析模式：")
    print("1. prompt_layout_all_en - 完整版面分析")
    print("2. prompt_layout_only_en - 僅版面檢測")
    print("3. prompt_ocr - 僅文字提取")
    
    mode_choice = input("請選擇解析模式 (1-3, 預設 1): ").strip()
    mode_map = {
        "1": "prompt_layout_all_en",
        "2": "prompt_layout_only_en", 
        "3": "prompt_ocr",
        "": "prompt_layout_all_en"
    }
    prompt_mode = mode_map.get(mode_choice, "prompt_layout_all_en")
    
    print(f"\n🚀 開始處理 {pdf_path}...")
    print(f"📋 解析模式：{prompt_mode}")
    
    try:
        # 執行增強處理
        result = processor.process_pdf_enhanced(pdf_path, prompt_mode)
        
        if result['success']:
            print(f"\n🎉 處理完成！")
            print(f"📁 結構化輸出目錄：{result['structured_dir']}")
            
            # 顯示所有生成的檔案
            if 'actual_files' in result['parse_result']:
                print(f"\n📋 解析生成的檔案：")
                for file_type, file_path in result['parse_result']['actual_files']:
                    print(f"  {file_type}: {file_path}")
            
            # 檢查並顯示結構化檔案
            structured_dir = result['structured_dir']
            if os.path.exists(structured_dir):
                print(f"\n📂 結構化檔案：")
                for file in os.listdir(structured_dir):
                    file_path = os.path.join(structured_dir, file)
                    if os.path.isfile(file_path):
                        file_size = os.path.getsize(file_path)
                        print(f"  {file} ({file_size} bytes)")
            
            # 詢問是否查看結果
            view_choice = input("\n是否查看處理結果？ (y/n): ").strip().lower()
            if view_choice == 'y':
                # 顯示 Markdown 內容預覽
                md_content = result['parse_result']['combined_md_content']
                if md_content:
                    print("\n📝 Markdown 內容預覽（前500字符）：")
                    print("-" * 50)
                    print(md_content[:500])
                    if len(md_content) > 500:
                        print("...(更多內容請查看輸出檔案)")
                        
                # 詢問是否打開檔案所在目錄
                open_choice = input("\n是否在檔案管理器中打開輸出目錄？ (y/n): ").strip().lower()
                if open_choice == 'y':
                    output_dir = os.path.abspath(result['structured_dir'])
                    print(f"📁 輸出目錄位置：{output_dir}")
                    # 嘗試打開檔案管理器（Linux）
                    try:
                        os.system(f"xdg-open '{output_dir}' 2>/dev/null || nautilus '{output_dir}' 2>/dev/null || echo '請手動打開目錄：{output_dir}'")
                    except:
                        print(f"請手動打開目錄：{output_dir}")
        else:
            print(f"❌ 處理失敗：{result['error']}")
            
    except Exception as e:
        print(f"❌ 處理過程中發生錯誤：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 設置工作目錄
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 創建輸出目錄
    os.makedirs("./pdf_output", exist_ok=True)
    
    try:
        demo_enhanced_pdf_processing()
        
        print("\n" + "=" * 60)
        print("💡 使用提示：")
        print("1. 本範例使用與 demo 相同的高階 API")
        print("2. 解析品質與 demo/demo_gradio.py 保持一致")
        print("3. 自動進行混合語言內容分析")
        print("4. 結構化輸出便於後續處理")
        print("5. 支援表格、圖片、公式的專門處理")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠ 用戶中斷操作")
    except Exception as e:
        print(f"\n❌ 發生錯誤：{e}")
        import traceback
        traceback.print_exc()