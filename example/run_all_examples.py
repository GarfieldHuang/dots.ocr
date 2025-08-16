#!/usr/bin/env python3
"""
範例執行器 - 運行所有 dots.ocr 範例

這個腳本會依序執行所有範例程式，展示 dots.ocr 的各種功能。
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_requirements():
    """檢查運行要求"""
    print("=== 檢查運行要求 ===")
    
    # 檢查 conda 環境
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env != 'dots_ocr':
        print("⚠ 警告：建議在 dots_ocr conda 環境中運行")
        print("請執行：conda activate dots_ocr")
    else:
        print("✓ conda 環境正確")
    
    # 檢查必要檔案
    required_files = [
        "../demo/demo_image1.jpg",
        "../demo/demo_pdf1.pdf"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"⚠ 找不到範例檔案：{missing_files}")
        print("某些範例可能無法執行")
    else:
        print("✓ 範例檔案齊全")
    
    # 檢查模型權重
    model_path = "./weights/DotsOCR"
    if not os.path.exists(model_path):
        print(f"⚠ 找不到模型權重：{model_path}")
        print("請先下載模型權重：python tools/download_model.py")
        return False
    else:
        print("✓ 模型權重存在")
    
    return True


def check_vllm_server():
    """檢查 vLLM 伺服器狀態"""
    print("\\n=== 檢查 vLLM 伺服器 ===")
    
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=3)
        if response.status_code == 200:
            print("✓ vLLM 伺服器正在運行")
            return True
    except:
        pass
    
    print("✗ vLLM 伺服器未運行")
    print("\\n請在另一個終端啟動 vLLM 伺服器：")
    print("conda activate dots_ocr")
    print("export hf_model_path=./weights/DotsOCR")
    print("export PYTHONPATH=$(dirname \"$hf_model_path\"):$PYTHONPATH")
    print("CUDA_VISIBLE_DEVICES=0 vllm serve ${hf_model_path} --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --chat-template-content-format string --served-model-name model --trust-remote-code")
    
    return False


def run_example(script_name, description, use_vllm=True):
    """
    運行單個範例
    
    Args:
        script_name (str): 腳本檔名
        description (str): 範例描述
        use_vllm (bool): 是否需要 vLLM 伺服器
    """
    print(f"\\n{'='*60}")
    print(f"執行範例：{description}")
    print(f"腳本：{script_name}")
    print('='*60)
    
    if use_vllm and not check_vllm_server():
        print(f"跳過 {script_name}（需要 vLLM 伺服器）")
        return False
    
    try:
        start_time = time.time()
        
        # 執行腳本
        result = subprocess.run([
            sys.executable, script_name
        ], capture_output=True, text=True, timeout=300)  # 5分鐘超時
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ {description} 執行成功（耗時 {execution_time:.2f} 秒）")
            
            # 顯示輸出的最後幾行
            output_lines = result.stdout.strip().split('\\n')
            if len(output_lines) > 5:
                print("輸出摘要：")
                for line in output_lines[-5:]:
                    print(f"  {line}")
            
            return True
        else:
            print(f"✗ {description} 執行失敗")
            print("錯誤輸出：")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ {description} 執行超時")
        return False
    except Exception as e:
        print(f"✗ {description} 執行異常：{e}")
        return False


def main():
    """主函數"""
    print("=== DotsOCR 範例執行器 ===")
    print(f"當前目錄：{os.getcwd()}")
    
    # 檢查基本要求
    if not check_requirements():
        print("\\n請解決上述問題後重新運行")
        return
    
    # 定義範例列表
    examples = [
        ("basic_usage.py", "基本使用範例", True),
        ("simple_image_parser.py", "簡單圖片解析", True),
        ("vllm_server_usage.py", "vLLM 伺服器使用", True),
        ("hf_transformers_usage.py", "HuggingFace Transformers 使用", False),
        ("grounding_ocr_example.py", "指定區域 OCR", True),
        ("advanced_parsing.py", "高級解析功能", True),
        ("pdf_parsing_example.py", "PDF 解析", True),
        ("multilingual_example.py", "多語言文件解析", True),
        ("batch_processing.py", "批次處理", True),
    ]
    
    # 詢問用戶要運行哪些範例
    print(f"\\n找到 {len(examples)} 個範例：")
    for i, (script, desc, needs_vllm) in enumerate(examples, 1):
        vllm_status = " [需要 vLLM]" if needs_vllm else " [獨立運行]"
        print(f"  {i}. {desc}{vllm_status}")
    
    print("\\n選擇運行模式：")
    print("1. 運行所有範例")
    print("2. 運行單個範例")
    print("3. 運行不需要 vLLM 的範例")
    print("4. 只運行基本範例")
    
    try:
        choice = input("\\n請選擇 (1-4): ").strip()
        
        if choice == "1":
            # 運行所有範例
            selected_examples = examples
        elif choice == "2":
            # 運行單個範例
            try:
                idx = int(input(f"請選擇範例編號 (1-{len(examples)}): ")) - 1
                if 0 <= idx < len(examples):
                    selected_examples = [examples[idx]]
                else:
                    print("無效的範例編號")
                    return
            except ValueError:
                print("請輸入有效的數字")
                return
        elif choice == "3":
            # 只運行不需要 vLLM 的範例
            selected_examples = [(s, d, v) for s, d, v in examples if not v]
        elif choice == "4":
            # 只運行基本範例
            selected_examples = examples[:3]
        else:
            print("無效的選擇")
            return
        
        # 執行選定的範例
        print(f"\\n準備執行 {len(selected_examples)} 個範例...")
        
        success_count = 0
        total_start_time = time.time()
        
        for script_name, description, needs_vllm in selected_examples:
            if run_example(script_name, description, needs_vllm):
                success_count += 1
            
            # 短暫暫停，避免伺服器過載
            time.sleep(2)
        
        total_time = time.time() - total_start_time
        
        # 總結
        print(f"\\n{'='*60}")
        print("執行總結")
        print('='*60)
        print(f"總共執行：{len(selected_examples)} 個範例")
        print(f"成功：{success_count} 個")
        print(f"失敗：{len(selected_examples) - success_count} 個")
        print(f"總耗時：{total_time:.2f} 秒")
        
        if success_count == len(selected_examples):
            print("\\n🎉 所有範例執行成功！")
        else:
            print("\\n⚠ 部分範例執行失敗，請檢查錯誤訊息")
        
        print(f"\\n結果檔案保存在以下目錄：")
        output_dirs = [
            "./output",
            "./batch_output", 
            "./grounding_output",
            "./advanced_output",
            "./pdf_output",
            "./multilingual_output"
        ]
        
        for output_dir in output_dirs:
            if os.path.exists(output_dir):
                file_count = len(os.listdir(output_dir))
                print(f"  {output_dir}: {file_count} 個檔案")
        
    except KeyboardInterrupt:
        print("\\n程式被用戶中斷")
    except Exception as e:
        print(f"\\n程式執行異常：{e}")


if __name__ == "__main__":
    main()