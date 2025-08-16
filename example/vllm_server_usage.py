#!/usr/bin/env python3
"""
vLLM 伺服器使用範例 - 展示如何使用 vLLM API

這個範例展示如何：
1. 檢查和啟動 vLLM 伺服器
2. 使用 vLLM API 進行推理
3. 自訂 vLLM 參數
4. 處理 vLLM 伺服器錯誤
"""

import os
import sys
import time
import requests
import subprocess
from pathlib import Path
from PIL import Image

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dots_ocr.model.inference import inference_with_vllm
from dots_ocr.utils.prompts import dict_promptmode_to_prompt
from dots_ocr.utils.image_utils import fetch_image


class VLLMServerManager:
    """vLLM 伺服器管理器"""
    
    def __init__(self, ip="localhost", port=8000, model_name="model"):
        self.ip = ip
        self.port = port
        self.model_name = model_name
        self.base_url = f"http://{ip}:{port}"
    
    def is_server_running(self) -> bool:
        """檢查伺服器是否正在運行"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def wait_for_server(self, timeout=60) -> bool:
        """等待伺服器啟動"""
        print(f"等待 vLLM 伺服器啟動 ({self.base_url})...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_server_running():
                print("✓ vLLM 伺服器已就緒")
                return True
            time.sleep(2)
        
        print("✗ vLLM 伺服器啟動超時")
        return False
    
    def get_server_info(self) -> dict:
        """獲取伺服器資訊"""
        try:
            # 嘗試獲取模型資訊
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {}
    
    def test_inference(self, image_path: str, prompt: str) -> str:
        """測試推理功能"""
        try:
            image = fetch_image(image_path)
            response = inference_with_vllm(
                image=image,
                prompt=prompt,
                ip=self.ip,
                port=self.port,
                model_name=self.model_name,
                temperature=0.1,
                top_p=0.9
            )
            return response
        except Exception as e:
            return f"推理錯誤：{str(e)}"


def demonstrate_vllm_usage():
    """展示 vLLM 使用方法"""
    print("=== vLLM 伺服器使用範例 ===")
    
    # 1. 初始化伺服器管理器
    server = VLLMServerManager()
    
    # 2. 檢查伺服器狀態
    print("\n--- 1. 檢查伺服器狀態 ---")
    if server.is_server_running():
        print("✓ vLLM 伺服器正在運行")
        
        # 獲取伺服器資訊
        server_info = server.get_server_info()
        if server_info:
            print(f"伺服器資訊：{server_info}")
    else:
        print("✗ vLLM 伺服器未運行")
        print("\n請啟動 vLLM 伺服器：")
        print("conda activate dots_ocr")
        print("CUDA_VISIBLE_DEVICES=0 vllm serve ./weights/DotsOCR --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --chat-template-content-format string --served-model-name model --trust-remote-code")
        
        # 等待用戶啟動伺服器
        input("\n啟動伺服器後按 Enter 繼續...")
        
        if not server.wait_for_server():
            print("無法連接到 vLLM 伺服器，程式退出")
            return
    
    # 3. 基本推理測試
    print("\n--- 2. 基本推理測試 ---")
    
    image_path = "../demo/demo_image1.jpg"
    if not os.path.exists(image_path):
        print(f"找不到測試圖片：{image_path}")
        return
    
    # 使用簡單的 OCR 提示
    prompt = dict_promptmode_to_prompt["prompt_ocr"]
    
    print(f"正在測試圖片：{image_path}")
    print(f"使用提示：{prompt[:100]}...")
    
    result = server.test_inference(image_path, prompt)
    
    if result.startswith("推理錯誤"):
        print(f"✗ {result}")
    else:
        print("✓ 推理成功")
        print(f"結果長度：{len(result)} 字符")
        print(f"結果預覽：{result[:200]}...")
    
    # 4. 不同模式測試
    print("\n--- 3. 不同模式測試 ---")
    
    test_modes = [
        ("prompt_layout_only_en", "版面檢測"),
        ("prompt_layout_all_en", "完整解析"),
    ]
    
    for prompt_mode, description in test_modes:
        print(f"\n測試 {description} 模式...")
        prompt = dict_promptmode_to_prompt[prompt_mode]
        
        start_time = time.time()
        result = server.test_inference(image_path, prompt)
        processing_time = time.time() - start_time
        
        if result.startswith("推理錯誤"):
            print(f"✗ {result}")
        else:
            print(f"✓ 成功，耗時 {processing_time:.2f} 秒")
            print(f"結果長度：{len(result)} 字符")
    
    # 5. 參數調整測試
    print("\n--- 4. 參數調整測試 ---")
    
    print("測試不同的 temperature 參數...")
    
    temperatures = [0.0, 0.1, 0.5, 1.0]
    prompt = dict_promptmode_to_prompt["prompt_ocr"]
    
    for temp in temperatures:
        print(f"\nTemperature = {temp}")
        
        try:
            image = fetch_image(image_path)
            result = inference_with_vllm(
                image=image,
                prompt=prompt,
                ip=server.ip,
                port=server.port,
                model_name=server.model_name,
                temperature=temp,
                top_p=0.9
            )
            print(f"✓ 成功，結果長度：{len(result)}")
            
        except Exception as e:
            print(f"✗ 錯誤：{e}")


def show_vllm_server_commands():
    """顯示 vLLM 伺服器相關命令"""
    print("\n=== vLLM 伺服器相關命令 ===")
    
    print("\n1. 啟動伺服器：")
    print("conda activate dots_ocr")
    print("export hf_model_path=./weights/DotsOCR")
    print("export PYTHONPATH=$(dirname \"$hf_model_path\"):$PYTHONPATH")
    print("CUDA_VISIBLE_DEVICES=0 vllm serve ${hf_model_path} --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --chat-template-content-format string --served-model-name model --trust-remote-code")
    
    print("\n2. 檢查伺服器狀態：")
    print("curl http://localhost:8000/health")
    
    print("\n3. 查看可用模型：")
    print("curl http://localhost:8000/v1/models")
    
    print("\n4. 停止伺服器：")
    print("按 Ctrl+C 或關閉終端")
    
    print("\n5. 使用不同 GPU：")
    print("CUDA_VISIBLE_DEVICES=1 vllm serve ...")  # 使用 GPU 1
    
    print("\n6. 調整記憶體使用：")
    print("--gpu-memory-utilization 0.9  # 使用 90% GPU 記憶體")
    
    print("\n7. 多 GPU 部署：")
    print("--tensor-parallel-size 2  # 使用 2 個 GPU")


def main():
    """主函數"""
    try:
        demonstrate_vllm_usage()
        
        print("\n" + "="*50)
        show_vllm_server_commands()
        
    except KeyboardInterrupt:
        print("\n程式被用戶中斷")
    except Exception as e:
        print(f"\n程式執行錯誤：{e}")


if __name__ == "__main__":
    main()