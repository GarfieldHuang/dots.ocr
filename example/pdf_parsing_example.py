#!/usr/bin/env python3
"""
PDF 解析範例 - 展示如何處理 PDF 文件

這個範例展示如何：
1. 解析單頁和多頁 PDF
2. 批次處理 PDF 文件
3. 自訂 PDF 處理參數
4. 合併多頁結果
5. 處理大型 PDF 文件
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dots_ocr.parser import DotsOCRParser
from dots_ocr.utils.doc_utils import load_images_from_pdf


class PDFProcessor:
    """PDF 處理器"""
    
    def __init__(self, use_hf=False, num_threads=8, dpi=200):
        """
        初始化 PDF 處理器
        
        Args:
            use_hf (bool): 是否使用 HuggingFace 模型
            num_threads (int): 處理線程數
            dpi (int): PDF 渲染 DPI
        """
        self.parser = DotsOCRParser(
            use_hf=use_hf,
            num_thread=num_threads,
            dpi=dpi,
            output_dir="./pdf_output"
        )
        self.processing_stats = {
            'total_files': 0,
            'total_pages': 0,
            'successful_pages': 0,
            'failed_pages': 0,
            'total_time': 0
        }
    
    def get_pdf_info(self, pdf_path: str) -> Dict[str, Any]:
        """
        獲取 PDF 基本資訊
        
        Args:
            pdf_path (str): PDF 檔案路徑
            
        Returns:
            Dict: PDF 資訊
        """
        try:
            images = load_images_from_pdf(pdf_path, dpi=self.parser.dpi)
            
            info = {
                'file_path': pdf_path,
                'file_size': os.path.getsize(pdf_path),
                'page_count': len(images),
                'dpi': self.parser.dpi,
                'estimated_processing_time': len(images) * 10  # 估算每頁10秒
            }
            
            # 獲取第一頁尺寸資訊
            if images:
                first_page = images[0]
                info.update({
                    'page_width': first_page.width,
                    'page_height': first_page.height,
                    'page_pixels': first_page.width * first_page.height
                })
            
            return info
            
        except Exception as e:
            return {
                'file_path': pdf_path,
                'error': str(e),
                'page_count': 0
            }
    
    def parse_pdf_basic(self, pdf_path: str, prompt_mode: str = "prompt_layout_all_en") -> List[Dict]:
        """
        基本 PDF 解析
        
        Args:
            pdf_path (str): PDF 檔案路徑
            prompt_mode (str): 解析模式
            
        Returns:
            List[Dict]: 解析結果
        """
        print(f"正在解析 PDF：{pdf_path}")
        
        # 獲取 PDF 資訊
        pdf_info = self.get_pdf_info(pdf_path)
        print(f"PDF 資訊：{pdf_info['page_count']} 頁，檔案大小 {pdf_info['file_size']/1024/1024:.1f} MB")
        
        if 'error' in pdf_info:
            print(f"✗ PDF 載入失敗：{pdf_info['error']}")
            return []
        
        # 開始解析
        start_time = time.time()
        
        try:
            results = self.parser.parse_file(
                input_path=pdf_path,
                prompt_mode=prompt_mode
            )
            
            processing_time = time.time() - start_time
            
            # 更新統計
            self.processing_stats['total_files'] += 1
            self.processing_stats['total_pages'] += pdf_info['page_count']
            self.processing_stats['successful_pages'] += len(results)
            self.processing_stats['total_time'] += processing_time
            
            print(f"✓ 解析完成，耗時 {processing_time:.2f} 秒")
            print(f"處理速度：{pdf_info['page_count']/processing_time:.2f} 頁/秒")
            
            return results
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.processing_stats['total_files'] += 1
            self.processing_stats['total_pages'] += pdf_info['page_count']
            self.processing_stats['failed_pages'] += pdf_info['page_count']
            self.processing_stats['total_time'] += processing_time
            
            print(f"✗ 解析失敗：{e}")
            return []
    
    def parse_pdf_pages_range(self, pdf_path: str, start_page: int, end_page: int, 
                             prompt_mode: str = "prompt_layout_all_en") -> List[Dict]:
        """
        解析 PDF 指定頁面範圍
        
        Args:
            pdf_path (str): PDF 檔案路徑
            start_page (int): 起始頁面（從 0 開始）
            end_page (int): 結束頁面（不包含）
            prompt_mode (str): 解析模式
            
        Returns:
            List[Dict]: 解析結果
        """
        print(f"正在解析 PDF 第 {start_page+1}-{end_page} 頁：{pdf_path}")
        
        try:
            # 載入指定頁面
            all_images = load_images_from_pdf(pdf_path, dpi=self.parser.dpi)
            selected_images = all_images[start_page:end_page]
            
            print(f"載入了 {len(selected_images)} 頁圖片")
            
            # 創建臨時處理器來處理選定頁面
            # 這裡我們需要手動處理每一頁
            results = []
            
            for i, image in enumerate(selected_images):
                page_no = start_page + i
                print(f"處理第 {page_no+1} 頁...")
                
                result = self.parser._parse_single_image(
                    origin_image=image,
                    prompt_mode=prompt_mode,
                    save_dir=os.path.join(self.parser.output_dir, f"pdf_pages_{start_page+1}_{end_page}"),
                    save_name=f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{page_no}",
                    source="pdf",
                    page_idx=page_no
                )
                
                result['file_path'] = pdf_path
                results.append(result)
            
            print(f"✓ 完成指定頁面解析：{len(results)} 頁")
            return results
            
        except Exception as e:
            print(f"✗ 指定頁面解析失敗：{e}")
            return []
    
    def merge_pdf_results(self, results: List[Dict], output_path: str = None) -> Dict[str, Any]:
        """
        合併 PDF 多頁結果
        
        Args:
            results (List[Dict]): 各頁解析結果
            output_path (str): 輸出路徑
            
        Returns:
            Dict: 合併結果
        """
        if not results:
            return {}
        
        print(f"正在合併 {len(results)} 頁結果...")
        
        # 收集所有版面資訊
        all_layout_data = []
        all_text_content = []
        
        for result in results:
            page_no = result.get('page_no', 0)
            
            # 讀取版面資訊
            if 'layout_info_path' in result:
                try:
                    with open(result['layout_info_path'], 'r', encoding='utf-8') as f:
                        page_layout = json.load(f)
                    
                    # 為每個元素添加頁面資訊
                    for item in page_layout:
                        item['page_number'] = page_no + 1
                    
                    all_layout_data.extend(page_layout)
                    
                except Exception as e:
                    print(f"⚠ 讀取第 {page_no+1} 頁版面資訊失敗：{e}")
            
            # 讀取文字內容
            if 'md_content_path' in result:
                try:
                    with open(result['md_content_path'], 'r', encoding='utf-8') as f:
                        page_text = f.read()
                    
                    all_text_content.append(f"\\n\\n--- 第 {page_no+1} 頁 ---\\n\\n{page_text}")
                    
                except Exception as e:
                    print(f"⚠ 讀取第 {page_no+1} 頁文字內容失敗：{e}")
        
        # 統計資訊
        merged_info = {
            'total_pages': len(results),
            'total_layout_elements': len(all_layout_data),
            'total_text_length': sum(len(text) for text in all_text_content),
            'pages_info': []
        }
        
        # 統計每頁資訊
        for result in results:
            page_info = {
                'page_number': result.get('page_no', 0) + 1,
                'input_width': result.get('input_width', 0),
                'input_height': result.get('input_height', 0),
                'has_layout': 'layout_info_path' in result,
                'has_text': 'md_content_path' in result
            }
            merged_info['pages_info'].append(page_info)
        
        # 統計版面元素類型
        category_stats = {}
        for item in all_layout_data:
            category = item.get('category', 'Unknown')
            category_stats[category] = category_stats.get(category, 0) + 1
        
        merged_info['category_statistics'] = category_stats
        
        # 保存合併結果
        if output_path is None:
            base_name = results[0].get('file_path', 'merged')
            base_name = os.path.splitext(os.path.basename(base_name))[0]
            output_path = f"./pdf_output/{base_name}_merged"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存合併的版面資訊
        layout_path = f"{output_path}_layout.json"
        with open(layout_path, 'w', encoding='utf-8') as f:
            json.dump(all_layout_data, f, ensure_ascii=False, indent=2)
        
        # 保存合併的文字內容
        text_path = f"{output_path}_text.md"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write("\\n\\n".join(all_text_content))
        
        # 保存統計資訊
        info_path = f"{output_path}_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(merged_info, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 合併結果已保存：")
        print(f"  版面資訊：{layout_path}")
        print(f"  文字內容：{text_path}")
        print(f"  統計資訊：{info_path}")
        
        print(f"\\n合併統計：")
        print(f"  總頁數：{merged_info['total_pages']}")
        print(f"  版面元素：{merged_info['total_layout_elements']} 個")
        print(f"  文字長度：{merged_info['total_text_length']} 字符")
        print(f"  元素類型：{len(category_stats)} 種")
        
        return merged_info
    
    def batch_process_pdfs(self, pdf_paths: List[str], prompt_mode: str = "prompt_layout_all_en") -> List[Dict]:
        """
        批次處理多個 PDF
        
        Args:
            pdf_paths (List[str]): PDF 檔案路徑列表
            prompt_mode (str): 解析模式
            
        Returns:
            List[Dict]: 批次處理結果
        """
        print(f"開始批次處理 {len(pdf_paths)} 個 PDF 檔案...")
        
        batch_results = []
        
        for i, pdf_path in enumerate(pdf_paths, 1):
            print(f"\\n處理 PDF {i}/{len(pdf_paths)}: {pdf_path}")
            
            if not os.path.exists(pdf_path):
                print(f"✗ 檔案不存在：{pdf_path}")
                continue
            
            try:
                results = self.parse_pdf_basic(pdf_path, prompt_mode)
                
                batch_result = {
                    'file_path': pdf_path,
                    'status': 'success' if results else 'failed',
                    'page_results': results,
                    'page_count': len(results)
                }
                
                # 合併頁面結果
                if results:
                    merged_info = self.merge_pdf_results(results)
                    batch_result['merged_info'] = merged_info
                
                batch_results.append(batch_result)
                
            except Exception as e:
                print(f"✗ 處理失敗：{e}")
                batch_results.append({
                    'file_path': pdf_path,
                    'status': 'error',
                    'error': str(e),
                    'page_results': [],
                    'page_count': 0
                })
        
        self.print_batch_statistics()
        return batch_results
    
    def print_batch_statistics(self):
        """列印批次處理統計"""
        stats = self.processing_stats
        
        print(f"\\n=== 批次處理統計 ===")
        print(f"總檔案數：{stats['total_files']}")
        print(f"總頁數：{stats['total_pages']}")
        print(f"成功頁數：{stats['successful_pages']}")
        print(f"失敗頁數：{stats['failed_pages']}")
        print(f"成功率：{stats['successful_pages']/max(stats['total_pages'], 1)*100:.1f}%")
        print(f"總耗時：{stats['total_time']:.2f} 秒")
        
        if stats['total_pages'] > 0:
            print(f"平均速度：{stats['total_pages']/stats['total_time']:.2f} 頁/秒")


def main():
    """PDF 解析示例"""
    print("=== PDF 解析範例 ===")
    
    # 創建 PDF 處理器
    processor = PDFProcessor(
        use_hf=False,
        num_threads=4,  # 根據您的硬體調整
        dpi=200
    )
    
    # 測試 PDF 檔案
    test_pdf = "../demo/demo_pdf1.pdf"
    
    if not os.path.exists(test_pdf):
        print(f"找不到測試 PDF：{test_pdf}")
        print("請確認 demo 資料夾中有 PDF 檔案")
        return
    
    try:
        # 1. 獲取 PDF 資訊
        print("\\n--- 1. PDF 資訊 ---")
        pdf_info = processor.get_pdf_info(test_pdf)
        print(f"PDF 資訊：{json.dumps(pdf_info, indent=2, ensure_ascii=False)}")
        
        # 2. 基本 PDF 解析
        print("\\n--- 2. 基本 PDF 解析 ---")
        results = processor.parse_pdf_basic(test_pdf, "prompt_layout_all_en")
        
        if results:
            print(f"✓ 成功解析 {len(results)} 頁")
            
            # 3. 合併多頁結果
            print("\\n--- 3. 合併多頁結果 ---")
            merged_info = processor.merge_pdf_results(results)
            
            # 4. 指定頁面解析（如果有多頁）
            if len(results) > 1:
                print("\\n--- 4. 指定頁面解析 ---")
                
                # 只解析前兩頁
                range_results = processor.parse_pdf_pages_range(
                    test_pdf, 
                    start_page=0, 
                    end_page=min(2, len(results)),
                    prompt_mode="prompt_ocr"  # 只提取文字，速度更快
                )
                
                print(f"指定頁面解析完成：{len(range_results)} 頁")
        
        # 5. 不同解析模式測試
        print("\\n--- 5. 不同解析模式測試 ---")
        
        test_modes = [
            ("prompt_layout_only_en", "只檢測版面"),
            ("prompt_ocr", "只提取文字"),
        ]
        
        for mode, description in test_modes:
            print(f"\\n測試 {description} 模式...")
            mode_results = processor.parse_pdf_basic(test_pdf, mode)
            print(f"結果：{len(mode_results)} 頁")
        
        # 6. 批次處理（如果有多個 PDF）
        demo_dir = "../demo"
        if os.path.exists(demo_dir):
            pdf_files = [os.path.join(demo_dir, f) for f in os.listdir(demo_dir) if f.endswith('.pdf')]
            
            if len(pdf_files) > 1:
                print(f"\\n--- 6. 批次處理 {len(pdf_files)} 個 PDF ---")
                batch_results = processor.batch_process_pdfs(
                    pdf_files[:2],  # 只處理前兩個，以節省時間
                    prompt_mode="prompt_ocr"
                )
                print(f"批次處理完成：{len(batch_results)} 個檔案")
        
        print(f"\\n所有結果已保存到：{processor.parser.output_dir}")
        
    except Exception as e:
        print(f"處理過程中發生錯誤：{e}")
        import traceback
        traceback.print_exc()


def show_pdf_optimization_tips():
    """顯示 PDF 處理優化建議"""
    print("\\n=== PDF 處理優化建議 ===")
    
    print("1. DPI 設置：")
    print("   - 150 DPI：快速處理，適合文字文件")
    print("   - 200 DPI：平衡品質和速度（推薦）")
    print("   - 300 DPI：高品質，適合複雜文件")
    
    print("\\n2. 線程數設置：")
    print("   - CPU 核心數的 1-2 倍")
    print("   - 考慮記憶體限制")
    print("   - vLLM 模式建議 4-8 線程")
    
    print("\\n3. 記憶體優化：")
    print("   - 分批處理大文件")
    print("   - 及時清理臨時檔案")
    print("   - 監控 GPU 記憶體使用")
    
    print("\\n4. 處理策略：")
    print("   - 先用 layout_only 模式預覽")
    print("   - 根據需求選擇合適的解析模式")
    print("   - 對重要頁面使用高品質設置")


if __name__ == "__main__":
    main()
    show_pdf_optimization_tips()