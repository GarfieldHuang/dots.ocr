#!/usr/bin/env python3
"""
批次處理範例 - 處理多個檔案和資料夾

這個範例展示如何：
1. 批次處理多個圖片檔案
2. 處理整個資料夾
3. 處理混合的檔案類型（圖片和PDF）
4. 自訂輸出格式
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
from dots_ocr.utils.consts import image_extensions


class BatchProcessor:
    """批次處理器"""
    
    def __init__(self, use_hf=False, num_threads=4):
        """
        初始化批次處理器
        
        Args:
            use_hf (bool): 是否使用 HuggingFace 模型
            num_threads (int): 處理 PDF 時的線程數
        """
        self.parser = DotsOCRParser(
            use_hf=use_hf,
            num_thread=num_threads,
            output_dir="./batch_output"
        )
        self.results = []
    
    def process_single_file(self, file_path: str, prompt_mode: str = "prompt_layout_all_en") -> Dict[str, Any]:
        """
        處理單一檔案
        
        Args:
            file_path (str): 檔案路徑
            prompt_mode (str): 處理模式
            
        Returns:
            Dict: 處理結果
        """
        print(f"正在處理：{file_path}")
        
        start_time = time.time()
        try:
            results = self.parser.parse_file(
                input_path=file_path,
                prompt_mode=prompt_mode
            )
            
            processing_time = time.time() - start_time
            
            return {
                "file_path": file_path,
                "status": "success",
                "processing_time": processing_time,
                "page_count": len(results),
                "results": results
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "file_path": file_path,
                "status": "error",
                "processing_time": processing_time,
                "error": str(e),
                "results": []
            }
    
    def process_file_list(self, file_paths: List[str], prompt_mode: str = "prompt_layout_all_en") -> List[Dict[str, Any]]:
        """
        處理檔案列表
        
        Args:
            file_paths (List[str]): 檔案路徑列表
            prompt_mode (str): 處理模式
            
        Returns:
            List[Dict]: 批次處理結果
        """
        print(f"開始批次處理 {len(file_paths)} 個檔案...")
        
        batch_results = []
        total_start_time = time.time()
        
        for i, file_path in enumerate(file_paths, 1):
            print(f"\n進度：{i}/{len(file_paths)}")
            result = self.process_single_file(file_path, prompt_mode)
            batch_results.append(result)
            
            # 顯示進度資訊
            if result["status"] == "success":
                print(f"✓ 成功處理，耗時 {result['processing_time']:.2f} 秒")
            else:
                print(f"✗ 處理失敗：{result['error']}")
        
        total_time = time.time() - total_start_time
        print(f"\n批次處理完成！總耗時：{total_time:.2f} 秒")
        
        # 統計結果
        successful = sum(1 for r in batch_results if r["status"] == "success")
        failed = len(batch_results) - successful
        total_pages = sum(r["page_count"] for r in batch_results if r["status"] == "success")
        
        print(f"成功：{successful} 個檔案")
        print(f"失敗：{failed} 個檔案") 
        print(f"總頁數：{total_pages} 頁")
        
        self.results.extend(batch_results)
        return batch_results
    
    def process_directory(self, directory_path: str, prompt_mode: str = "prompt_layout_all_en", 
                         recursive: bool = True) -> List[Dict[str, Any]]:
        """
        處理整個資料夾
        
        Args:
            directory_path (str): 資料夾路徑
            prompt_mode (str): 處理模式
            recursive (bool): 是否遞歸處理子資料夾
            
        Returns:
            List[Dict]: 批次處理結果
        """
        print(f"掃描資料夾：{directory_path}")
        
        # 支援的檔案格式
        supported_extensions = set(image_extensions + ['.pdf'])
        
        file_paths = []
        
        if recursive:
            # 遞歸掃描
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in supported_extensions):
                        file_paths.append(os.path.join(root, file))
        else:
            # 只掃描當前目錄
            for file in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file)
                if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in supported_extensions):
                    file_paths.append(file_path)
        
        print(f"找到 {len(file_paths)} 個支援的檔案")
        
        if not file_paths:
            print("沒有找到支援的檔案！")
            return []
        
        # 按檔案類型分類
        images = [f for f in file_paths if any(f.lower().endswith(ext) for ext in image_extensions)]
        pdfs = [f for f in file_paths if f.lower().endswith('.pdf')]
        
        print(f"圖片檔案：{len(images)} 個")
        print(f"PDF 檔案：{len(pdfs)} 個")
        
        return self.process_file_list(file_paths, prompt_mode)
    
    def save_batch_summary(self, output_path: str = "./batch_summary.json"):
        """
        保存批次處理摘要
        
        Args:
            output_path (str): 輸出檔案路徑
        """
        summary = {
            "total_files": len(self.results),
            "successful_files": sum(1 for r in self.results if r["status"] == "success"),
            "failed_files": sum(1 for r in self.results if r["status"] == "error"),
            "total_pages": sum(r["page_count"] for r in self.results if r["status"] == "success"),
            "total_processing_time": sum(r["processing_time"] for r in self.results),
            "results": self.results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"批次處理摘要已保存到：{output_path}")


def main():
    """批次處理示例"""
    print("=== 批次處理範例 ===")
    
    # 創建批次處理器
    processor = BatchProcessor(use_hf=False, num_threads=4)
    
    # 範例1：處理多個檔案
    print("\n--- 範例1：處理多個檔案 ---")
    file_list = [
        "../demo/demo_image1.jpg",
        "../demo/demo_pdf1.pdf"
    ]
    
    # 過濾存在的檔案
    existing_files = [f for f in file_list if os.path.exists(f)]
    
    if existing_files:
        batch_results = processor.process_file_list(
            existing_files,
            prompt_mode="prompt_layout_all_en"
        )
        
        print(f"\n處理了 {len(batch_results)} 個檔案")
    else:
        print("沒有找到範例檔案，跳過檔案列表處理")
    
    # 範例2：處理資料夾（如果 demo 資料夾存在）
    demo_dir = "../demo"
    if os.path.exists(demo_dir):
        print(f"\n--- 範例2：處理 {demo_dir} 資料夾 ---")
        directory_results = processor.process_directory(
            demo_dir,
            prompt_mode="prompt_ocr",  # 只提取文字，加快處理速度
            recursive=False
        )
        
        print(f"\n從資料夾處理了 {len(directory_results)} 個檔案")
    
    # 範例3：不同的處理模式
    if existing_files:
        print("\n--- 範例3：不同的處理模式 ---")
        
        # 只檢測版面
        layout_only_results = processor.process_file_list(
            existing_files[:1],  # 只處理第一個檔案
            prompt_mode="prompt_layout_only_en"
        )
        
        print("版面檢測模式處理完成")
    
    # 保存批次處理摘要
    processor.save_batch_summary("./batch_processing_summary.json")
    
    print(f"\n所有結果已保存到：{processor.parser.output_dir}")


if __name__ == "__main__":
    main()