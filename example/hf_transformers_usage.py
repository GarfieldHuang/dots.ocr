#!/usr/bin/env python3
"""
HuggingFace Transformers 使用範例 - 直接使用 Transformers 模型

這個範例展示如何：
1. 載入 HuggingFace 模型
2. 使用 Transformers 進行推理
3. 比較不同的注意力機制
4. 記憶體優化技巧
"""

import os
import sys
import time
import torch
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from transformers import AutoModelForCausalLM, AutoProcessor
    from qwen_vl_utils import process_vision_info
    from dots_ocr.utils.prompts import dict_promptmode_to_prompt
    from dots_ocr.utils.image_utils import fetch_image
except ImportError as e:
    print(f"導入錯誤：{e}")
    print("請確認已安裝相關依賴：pip install transformers qwen-vl-utils")
    sys.exit(1)


class HuggingFaceInference:
    """HuggingFace 推理器"""
    
    def __init__(self, model_path="./weights/DotsOCR", device="auto"):
        """
        初始化 HuggingFace 模型
        
        Args:
            model_path (str): 模型路徑
            device (str): 設備設置
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        
        print(f"正在載入模型：{model_path}")
        self._load_model()
    
    def _load_model(self):
        """載入模型和處理器"""
        try:
            # 檢查 CUDA 可用性
            if torch.cuda.is_available():
                print(f"✓ CUDA 可用，GPU 數量：{torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            else:
                print("⚠ CUDA 不可用，將使用 CPU（可能很慢）")
            
            # 載入模型
            start_time = time.time()
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                attn_implementation="flash_attention_2",  # 使用 Flash Attention
                torch_dtype=torch.bfloat16,  # 使用 bf16 以節省記憶體
                device_map=self.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True  # 降低 CPU 記憶體使用
            )
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            loading_time = time.time() - start_time
            print(f"✓ 模型載入完成，耗時 {loading_time:.2f} 秒")
            
            # 顯示模型資訊
            self._print_model_info()
            
        except Exception as e:
            print(f"✗ 模型載入失敗：{e}")
            raise
    
    def _print_model_info(self):
        """列印模型資訊"""
        if self.model is None:
            return
        
        # 計算參數數量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"模型參數：{total_params/1e9:.2f}B 總計，{trainable_params/1e9:.2f}B 可訓練")
        
        # 顯示記憶體使用
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU 記憶體使用：{memory_allocated:.2f} GB 已分配，{memory_reserved:.2f} GB 已保留")
    
    def inference(self, image_path: str, prompt: str, max_new_tokens: int = 24000, **kwargs) -> str:
        """
        執行推理
        
        Args:
            image_path (str): 圖片路徑
            prompt (str): 提示詞
            max_new_tokens (int): 最大生成標記數
            **kwargs: 其他生成參數
            
        Returns:
            str: 生成的回應
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("模型未載入")
        
        try:
            # 準備輸入
            image = fetch_image(image_path)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # 處理輸入
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # 移動到 GPU
            inputs = inputs.to(self.model.device)
            
            # 生成設置
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": kwargs.get("do_sample", True),
                "temperature": kwargs.get("temperature", 0.1),
                "top_p": kwargs.get("top_p", 0.9),
                "pad_token_id": self.processor.tokenizer.eos_token_id,
            }
            
            # 執行推理
            print(f"正在生成回應（最多 {max_new_tokens} 個標記）...")
            start_time = time.time()
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **generation_kwargs)
            
            inference_time = time.time() - start_time
            
            # 解碼結果
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # 計算生成速度
            generated_tokens = sum(len(ids) for ids in generated_ids_trimmed)
            tokens_per_second = generated_tokens / inference_time
            
            print(f"✓ 推理完成，耗時 {inference_time:.2f} 秒")
            print(f"生成 {generated_tokens} 個標記，速度 {tokens_per_second:.2f} tokens/s")
            
            return output_text
            
        except Exception as e:
            print(f"✗ 推理失敗：{e}")
            raise
    
    def clear_cache(self):
        """清理 GPU 快取"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✓ GPU 快取已清理")


def demonstrate_hf_usage():
    """展示 HuggingFace 使用方法"""
    print("=== HuggingFace Transformers 使用範例 ===")
    
    # 檢查模型路徑
    model_path = "./weights/DotsOCR"
    if not os.path.exists(model_path):
        print(f"✗ 找不到模型路徑：{model_path}")
        print("請確認已下載模型權重")
        return
    
    # 檢查測試圖片
    image_path = "../demo/demo_image1.jpg"
    if not os.path.exists(image_path):
        print(f"✗ 找不到測試圖片：{image_path}")
        return
    
    try:
        # 1. 初始化推理器
        print("\n--- 1. 載入模型 ---")
        inferencer = HuggingFaceInference(model_path)
        
        # 2. 基本推理測試
        print("\n--- 2. 基本推理測試 ---")
        
        prompt = dict_promptmode_to_prompt["prompt_ocr"]
        print(f"使用提示：OCR 文字提取")
        
        result = inferencer.inference(
            image_path=image_path,
            prompt=prompt,
            max_new_tokens=4000,
            temperature=0.1
        )
        
        print(f"提取的文字長度：{len(result)} 字符")
        print(f"文字預覽：{result[:300]}...")
        
        # 3. 版面解析測試
        print("\n--- 3. 版面解析測試 ---")
        
        prompt = dict_promptmode_to_prompt["prompt_layout_all_en"]
        print(f"使用提示：完整版面解析")
        
        result = inferencer.inference(
            image_path=image_path,
            prompt=prompt,
            max_new_tokens=16000,
            temperature=0.1
        )
        
        print(f"解析結果長度：{len(result)} 字符")
        
        # 嘗試解析 JSON
        try:
            import json
            parsed_result = json.loads(result)
            if isinstance(parsed_result, list):
                print(f"✓ 成功解析 JSON，檢測到 {len(parsed_result)} 個版面元素")
                
                # 統計版面元素類型
                categories = {}
                for item in parsed_result:
                    category = item.get('category', 'Unknown')
                    categories[category] = categories.get(category, 0) + 1
                
                print("版面元素統計：")
                for category, count in categories.items():
                    print(f"  {category}: {count}")
            
        except json.JSONDecodeError:
            print("⚠ 無法解析為 JSON，可能是文字格式回應")
            print(f"結果預覽：{result[:300]}...")
        
        # 4. 參數調整測試
        print("\n--- 4. 參數調整測試 ---")
        
        print("測試不同的生成參數...")
        
        test_configs = [
            {"temperature": 0.0, "description": "確定性生成"},
            {"temperature": 0.3, "description": "低隨機性"},
            {"temperature": 0.7, "description": "中等隨機性"},
        ]
        
        prompt = dict_promptmode_to_prompt["prompt_ocr"]
        
        for config in test_configs:
            print(f"\n{config['description']} (temperature={config['temperature']}):")
            
            try:
                result = inferencer.inference(
                    image_path=image_path,
                    prompt=prompt,
                    max_new_tokens=1000,
                    temperature=config['temperature']
                )
                print(f"✓ 成功，結果長度：{len(result)}")
                
            except Exception as e:
                print(f"✗ 失敗：{e}")
        
        # 5. 記憶體管理
        print("\n--- 5. 記憶體管理 ---")
        inferencer.clear_cache()
        
        print("\n✓ 所有測試完成")
        
    except Exception as e:
        print(f"✗ 程式執行錯誤：{e}")
        import traceback
        traceback.print_exc()


def show_memory_optimization_tips():
    """顯示記憶體優化建議"""
    print("\n=== 記憶體優化建議 ===")
    
    print("1. 使用量化：")
    print("   # 4-bit 量化（需要 bitsandbytes）")
    print("   load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16")
    
    print("\n2. 調整 batch size：")
    print("   # 減少 batch size 以節省記憶體")
    print("   max_new_tokens=1000  # 減少生成長度")
    
    print("\n3. 使用 CPU 卸載：")
    print("   # 將部分層卸載到 CPU")
    print("   device_map='balanced'")
    
    print("\n4. 清理快取：")
    print("   torch.cuda.empty_cache()")
    
    print("\n5. 監控記憶體：")
    print("   torch.cuda.memory_summary()")


def main():
    """主函數"""
    try:
        demonstrate_hf_usage()
        show_memory_optimization_tips()
        
    except KeyboardInterrupt:
        print("\n程式被用戶中斷")
    except Exception as e:
        print(f"\n程式執行錯誤：{e}")


if __name__ == "__main__":
    main()