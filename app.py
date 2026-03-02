#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XXG通义千问离线图片中文打标工具-离线版(Qwen3-VL) Ver.2.3
✅ 替换Qwen-vl-chat模型为Qwen3-VL | ✅ 文生图模型训练专用中文caption
✅ CUDA 11.8/12.1/12.4/13.x 全版本原生支持 | ✅ bitsandbytes>=0.44.0
✅ 100%中文caption | ✅ Qwen-Image、Seeddream等中文提示词LoRA训练专用
By 西小瓜 / Wechat:priest-mos
"""
import sys
import os
import time
import gc
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from config import Config


# ============ 核心修复: 猴子补丁注入HfFolder + is_offline_mode ============
def _inject_hf_compatibility():
    """动态注入HfFolder + is_offline_mode兼容层"""
    try:
        import warnings
        if 'huggingface_hub' not in sys.modules:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                __import__('huggingface_hub')

        import huggingface_hub

        # 注入is_offline_mode
        if not hasattr(huggingface_hub, 'is_offline_mode'):
            def is_offline_mode():
                import os
                return (
                        os.environ.get("HF_HUB_OFFLINE", "0") == "1" or
                        os.environ.get("HF_DATASETS_OFFLINE", "0") == "1" or
                        os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1" or
                        os.environ.get("HF_OFFLINE", "0") == "1"
                )

            huggingface_hub.is_offline_mode = is_offline_mode
            if hasattr(huggingface_hub, 'constants') and not hasattr(huggingface_hub.constants, 'is_offline_mode'):
                huggingface_hub.constants.is_offline_mode = is_offline_mode
            print("[MonkeyPatch] ✅ 已注入is_offline_mode兼容函数")

        # 注入HfFolder
        if not hasattr(huggingface_hub, 'HfFolder'):
            class HfFolder:
                _old_token_path = os.path.expanduser("~/.cache/huggingface/token")
                _new_token_path = os.path.expanduser("~/.cache/huggingface/hub/token")

                @staticmethod
                def get_token():
                    try:
                        from huggingface_hub import get_token
                        token = get_token()
                        if token:
                            return token
                        for path in [HfFolder._new_token_path, HfFolder._old_token_path]:
                            if os.path.exists(path):
                                with open(path, 'r') as f:
                                    return f.read().strip()
                        return None
                    except Exception:
                        return None

                @staticmethod
                def save_token(token: str):
                    try:
                        from huggingface_hub import login
                        login(token=token, add_to_git_credential=False)
                        os.makedirs(os.path.dirname(HfFolder._old_token_path), exist_ok=True)
                        with open(HfFolder._old_token_path, 'w') as f:
                            f.write(token)
                    except Exception:
                        pass

                @staticmethod
                def delete_token():
                    try:
                        for path in [HfFolder._new_token_path, HfFolder._old_token_path]:
                            if os.path.exists(path):
                                os.remove(path)
                        os.environ.pop("HF_TOKEN", None)
                    except Exception:
                        pass

            huggingface_hub.HfFolder = HfFolder
            print("[MonkeyPatch] ✅ 已注入HfFolder兼容类")

        has_is_offline = hasattr(huggingface_hub, 'is_offline_mode') and callable(huggingface_hub.is_offline_mode)
        has_hffolder = hasattr(huggingface_hub, 'HfFolder') and hasattr(huggingface_hub.HfFolder, 'get_token')

        if has_is_offline and has_hffolder:
            print(f"[MonkeyPatch] ✅ 兼容层注入成功 (huggingface_hub {huggingface_hub.__version__})")
            return True
        else:
            raise RuntimeError("注入不完整")

    except Exception as e:
        print(f"[MonkeyPatch] ⚠️  注入兼容层失败: {type(e).__name__}: {e}", file=sys.stderr)
        os.environ["HF_HUB_OFFLINE"] = "0"
        print("[MonkeyPatch] 💡 已设置HF_HUB_OFFLINE=0", file=sys.stderr)
        return False


_inject_hf_compatibility()
# ==============================================================================

# 设置环境变量
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# 修复NumPy 2.x兼容性
try:
    import numpy as np

    if np.__version__.startswith("2"):
        print(f"⚠️  检测到NumPy {np.__version__}，Qwen3-VL要求NumPy<2.0", file=sys.stderr)
        try:
            import subprocess

            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy<2.0", "--quiet"])
            print("✅ NumPy已降级，重启脚本生效", file=sys.stderr)
            sys.exit(0)
        except Exception as e:
            print(f"❌ 自动降级失败: {e}", file=sys.stderr)
            print("💡 请手动运行: pip install 'numpy<2.0' --upgrade", file=sys.stderr)
            sys.exit(1)
except ImportError:
    pass

# 导入torch
try:
    import torch
except ImportError as e:
    print(f"❌ 无法导入torch: {str(e)}", file=sys.stderr)
    print("💡 请先安装PyTorch 2.4.1", file=sys.stderr)
    sys.exit(1)

# 全局变量
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
processor = None
model_path = "./qwen3_vl_models"
global_use_4bit = False

# 导入其他依赖
try:
    from PIL import Image
    import gradio as gr
    from transformers import (
        Qwen3VLProcessor,
        Qwen3VLForConditionalGeneration,
        BitsAndBytesConfig
    )
    from tqdm import tqdm
    import psutil
    import platform
except ImportError as e:
    print(f"❌ 依赖导入失败: {str(e)}", file=sys.stderr)
    print("\n💡 解决方案:", file=sys.stderr)
    print("1. 确保已源码安装transformers (含Qwen3-VL支持)", file=sys.stderr)
    print("2. 验证安装: python -c 'from transformers import Qwen3VLProcessor; print(\"OK\")'", file=sys.stderr)
    print("3. 确保app.py开头包含猴子补丁注入代码", file=sys.stderr)
    sys.exit(1)

# ============ 核心修复: 强化中文特化Prompt (100%中文输出) ============
CAPTION_PROMPT = Config.get_caption_prompt()

# 中文主观词过滤（增强版）
SUBJECTIVE_WORDS = [
    "美丽", "可爱", "梦幻", "震撼", "惊艳", "优雅", "迷人", "漂亮", "帅气",
    "精致", "完美", "绝美", "超凡", "非凡", "令人", "非常", "极其", "特别",
    "好看", "动人", "倾城", "绝色", "清纯", "性感", "温柔", "甜美", "帅气",
    "英俊", "帅气", "可爱", "萌", "酷", "炫", "潮", "时尚", "高级", "质感"
]


def check_system_resources():
    """检查系统资源是否满足Qwen3-VL-8B要求"""
    print("🔍 系统资源检查 (Qwen3-VL-8B要求)...")

    disk = psutil.disk_usage(os.path.abspath("."))
    free_gb = disk.free / (1024 ** 3)
    print(f"💾 磁盘空间: {free_gb:.1f}GB 可用 (需要≥15GB)")
    if free_gb < 15:
        print(f"⚠️  警告: 磁盘空间不足! Qwen3-VL-8B模型约14GB")

    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    total_gb = mem.total / (1024 ** 3)
    print(f"🧠 系统内存: {available_gb:.1f}GB/{total_gb:.1f}GB 可用 (需要≥8GB)")
    if available_gb < 8:
        print(f"⚠️  警告: 可用内存不足8GB，处理大图时可能失败")

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        gpu_mem = props.total_memory / (1024 ** 3)
        print(f"🎮 GPU: {props.name} ({gpu_mem:.1f}GB 显存)")
        if gpu_mem < 6:
            print(f"⚠️  警告: GPU显存<6GB，必须启用4-bit量化")
        elif gpu_mem < 10:
            print(f"💡 建议: 启用4-bit量化获得更稳定体验")
    else:
        print("💻 使用CPU模式 (无GPU加速，速度较慢)")

    return {
        "disk_free_gb": free_gb,
        "mem_available_gb": available_gb,
        "gpu_available": device == "cuda",
        "gpu_mem_gb": torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if device == "cuda" else 0
    }


def smart_verify_qwen3_model(model_path: str):
    """智能验证Qwen3-VL模型文件完整性 (无special_tokens_map.json依赖)"""
    model_dir = Path(model_path)

    if not model_dir.exists() or not model_dir.is_dir():
        return False, "模型目录不存在或不是目录"

    # 移除special_tokens_map.json
    required_files = [
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json"
    ]
    missing = [f for f in required_files if not (model_dir / f).exists()]
    if missing:
        return False, f"缺失核心配置文件: {missing}"

    # 检查4分片权重
    weight_files = list(model_dir.glob("model-0000[1-4]-of-00004.safetensors")) + \
                   list(model_dir.glob("pytorch_model-0000[1-4]-of-00004.bin"))

    if not weight_files:
        if (model_dir / "model.safetensors.index.json").exists():
            return False, "检测到索引文件，但权重文件未完全下载"
        return False, "未找到模型权重文件 (需要model-0000X-of-00004.safetensors)"

    total_size = sum(f.stat().st_size for f in weight_files)
    if total_size < 12e9:
        return False, f"模型文件总大小过小 ({total_size / 1e9:.2f}GB)，可能下载不完整"

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
        vocab_size = len(tokenizer)
        if vocab_size < 150000:
            return False, f"Tokenizer词汇量过小 ({vocab_size})"
    except Exception as e:
        return False, f"Tokenizer验证失败: {str(e)}"

    return True, f"✅ Qwen3-VL模型验证成功! (4分片, {total_size / 1e9:.1f}GB)"


def load_qwen3_model(use_4bit: bool = False, use_cpu: bool = False):
    """加载Qwen3-VL-8B-Instruct模型"""
    global model, processor, device, global_use_4bit

    global_use_4bit = use_4bit

    if use_cpu:
        device = "cpu"
        print("⚠️  强制使用CPU模式 (无GPU加速)")

    if model is not None and processor is not None:
        print("✅ 模型已在内存中，跳过加载")
        return model, processor

    print(f"🚀 正在加载Qwen3-VL-8B-Instruct模型 (设备: {device.upper()})...")
    print(f"   模型路径: {os.path.abspath(model_path)}")

    if not os.path.exists(model_path):
        print(f"❌ 模型目录不存在: {model_path}")
        print("💡 请先下载模型: ./download_model.sh")
        raise FileNotFoundError(f"模型目录 {model_path} 不存在")

    print("🔍 智能验证模型文件...")
    model_valid, validation_msg = smart_verify_qwen3_model(model_path)
    if not model_valid:
        print(f"❌ 模型验证失败: {validation_msg}")
        print("💡 请重新下载完整模型: ./download_model.sh")
        raise ValueError("模型文件不完整或损坏")
    else:
        print(validation_msg)

    try:
        print("🔧 加载Qwen3VLProcessor...")
        processor = Qwen3VLProcessor.from_pretrained(
            model_path,
            trust_remote_code=False
        )
        print("✅ Processor加载成功!")

        quant_config = None
        if use_4bit and device == "cuda":
            print("⚡ 启用4-bit量化 (BitsAndBytes)...")
            try:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            except Exception as e:
                print(f"⚠️  4-bit量化初始化失败: {str(e)}")
                print("🔄 回退到标准加载...")
                use_4bit = False

        model_kwargs = {
            "trust_remote_code": False,
            "device_map": "auto" if device == "cuda" else "cpu",
            "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32
        }
        if quant_config:
            model_kwargs["quantization_config"] = quant_config

        print("🧠 加载Qwen3VLForConditionalGeneration...")
        start_time = time.time()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            **model_kwargs
        ).eval()
        load_time = time.time() - start_time
        print(f"✅ 模型加载成功! (耗时: {load_time:.1f}秒)")

        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return model, processor

    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()

        print("\n🛠️  详细故障排除:", file=sys.stderr)
        print("1. 模型下载: ./download_model.sh", file=sys.stderr)
        print("2. 依赖要求:", file=sys.stderr)
        print("   • PyTorch 2.4.1 + CUDA 12.4/12.1/13.0+", file=sys.stderr)
        print("   • bitsandbytes>=0.44.0", file=sys.stderr)
        print("   • transformers源码安装 (>=4.40.0)", file=sys.stderr)
        print("   • numpy<2.0", file=sys.stderr)
        sys.exit(1)


def _postprocess_caption(caption: str) -> str:
    """后处理：过滤主观词 + 格式标准化 + 强制中文"""
    # 移除系统/用户前缀
    if "assistant" in caption:
        caption = caption.split("assistant")[-1].strip()

    # 移除主观词
    for word in SUBJECTIVE_WORDS:
        caption = caption.replace(word, "")

    # 标准化标点 (英文逗号分隔)
    caption = caption.replace("，", ",").replace("、", ",").replace("。", "").replace("；", ",")

    # 移除多余空格
    caption = ",".join([part.strip() for part in caption.split(",") if part.strip()])

    # 截断至200字符
    #if len(caption) > 200:
    #    parts = caption.split(",")
    #    caption = ",".join(parts[:12]) + "..."

    # 确保以关键词结尾
    caption = caption.rstrip(",. ")

    return caption


# ✅ 核心修复: 严格遵循Qwen3-VL官方API + 强制中文输出
def generate_chinese_caption(image_path: str, max_new_tokens: int = 300):
    """使用Qwen3-VL生成100%中文训练专用caption"""
    global model, processor

    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")

        # 打开并验证图片
        image = Image.open(image_path).convert("RGB")
        image.verify()
        image = Image.open(image_path).convert("RGB")

        # ✅ 核心: messages中使用图像文件路径（字符串）
        messages = [
            {"role": "system", "content": CAPTION_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},  # ✅ 文件路径字符串
                    {"type": "text", "text": "生成文生图模型训练用中文caption，禁用所有英文描述，必须使用中文自然语句描述"}
                ]
            }
        ]

        # 处理输入
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # ✅ 核心: processor()中传入PIL Image对象
        inputs = processor(
            text=[text],
            images=[image],  # ✅ PIL Image对象
            return_tensors="pt",
            padding=True
        ).to(model.device)

        # 生成 (调整参数优化中文生成)
        start_time = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.55,  # ✅ 提高temperature增强创造性(中文)
                do_sample=True,
                top_p=0.7,
                top_k=20,
                repetition_penalty=1.2
            )
        gen_time = time.time() - start_time

        # 解码
        caption_raw = processor.decode(output[0], skip_special_tokens=True)
        caption_clean = _postprocess_caption(caption_raw)

        print(f"⏱️  生成耗时: {gen_time:.1f}秒 | 长度: {len(caption_clean)}字符")
        print(f"   描述: {caption_clean[:80]}...")
        return caption_clean

    except Exception as e:
        print(f"❌ 处理 {os.path.basename(image_path)} 时出错: {str(e)}")
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return None


def process_images(folder_path: str,trigger_word: str,use_4bit: bool = False, use_cpu: bool = False, progress=None):
    """批量处理图片文件夹"""
    if not folder_path or not folder_path.strip():
        return "❌ 错误: 请输入有效的文件夹路径"

    folder_path = folder_path.strip()
    if not os.path.isdir(folder_path):
        return f"❌ 错误: 路径 '{folder_path}' 不是有效文件夹"

    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

    image_files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f.lower())[1] in SUPPORTED_FORMATS and
           not f.lower().startswith('._')
    ]

    if not image_files:
        return f"⚠️ 警告: 在 '{folder_path}' 中未找到支持的图片文件"

    load_qwen3_model(use_4bit=use_4bit, use_cpu=use_cpu)

    results = {
        "total": len(image_files),
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "details": []
    }

    total = len(image_files)

    for i, filename in enumerate(image_files):
        if progress:
            progress(i / total, desc=f"处理中 ({i + 1}/{total}) - {filename}")

        image_path = os.path.join(folder_path, filename)
        txt_path = os.path.splitext(image_path)[0] + '.txt'

        if os.path.exists(txt_path):
            results["skipped"] += 1
            results["details"].append(f"⏭ 跳过: {filename} (已存在描述文件)")
            continue

        print(f"\n🖼️  处理: {filename}")
        caption = generate_chinese_caption(image_path)

        if caption and len(caption) > 30:
            try:
                if len(trigger_word.strip()) > 0:
                    caption = trigger_word.strip() + "," + caption
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(caption)
                results["success"] += 1
                preview = caption[:70] + "..." if len(caption) > 70 else caption
                results["details"].append(f"✅ 成功: {filename}\n   {preview}")
            except Exception as e:
                results["failed"] += 1
                results["details"].append(f"❌ 写入失败: {filename}\n   {str(e)}")
        else:
            results["failed"] += 1
            results["details"].append(f"❌ 生成失败: {filename}")

        if i % 3 == 0:
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    processed = max(1, results["total"] - results["skipped"])
    success_rate = results["success"] / processed * 100

    report = (
            f"🎉 批量处理完成!\n\n"
            f"📊 总计: {results['total']} 张图片\n"
            f"✅ 成功: {results['success']} ({success_rate:.1f}%)\n"
            f"❌ 失败: {results['failed']}\n"
            f"⏭ 跳过: {results['skipped']} (已存在)\n\n"
            f"📁 结果保存在: {folder_path}\n\n"
            f"📋 详细日志 (最近10条):\n" +
            "\n".join(results["details"][-10:])
    )

    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return report


def get_system_info():
    """获取系统信息"""
    try:
        gpu_info = "未检测到GPU"
        if device == "cuda":
            props = torch.cuda.get_device_properties(0)
            gpu_info = f"{props.name} ({props.total_memory / 1e9:.1f}GB)"

        mem = psutil.virtual_memory()
        disk = psutil.disk_usage(os.path.abspath("."))

        model_status = "✅ 已加载" if model is not None else "⏳ 未加载"
        quant_status = " (4-bit)" if global_use_4bit and model is not None else ""

        model_size = "未知"
        if model is not None:
            total_params = sum(p.numel() for p in model.parameters())
            model_size = f"{total_params / 1e9:.1f}B"

        transformers_version = "未知"
        try:
            import transformers
            transformers_version = transformers.__version__
        except:
            pass

        numpy_version = "未知"
        try:
            import numpy as np
            numpy_version = np.__version__
        except:
            pass

        import huggingface_hub
        patch_status = "✅ 已注入" if hasattr(huggingface_hub, 'HfFolder') else "❌ 未注入"

        return (
            f"**操作系统**: {platform.system()} {platform.release()}\n"
            f"**Python版本**: {platform.python_version()}\n"
            f"**PyTorch版本**: {torch.__version__} (CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'})\n"
            f"**Transformers版本**: {transformers_version}\n"
            f"**huggingface_hub版本**: {huggingface_hub.__version__} ({patch_status})\n"
            f"**NumPy版本**: {numpy_version}\n"
            f"**运行设备**: {device.upper()} ({gpu_info})\n"
            f"**系统内存**: {mem.total / 1e9:.1f}GB (可用: {mem.available / 1e9:.1f}GB)\n"
            f"**磁盘空间**: {disk.free / 1e9:.1f}GB 可用\n"
            f"**模型状态**: {model_status}{quant_status}\n"
            f"**模型大小**: Qwen3-VL-8B ({model_size})\n"
            f"**模型路径**: `{os.path.abspath(model_path)}`"
        )
    except Exception as e:
        return f"⚠️ 获取系统信息失败: {str(e)}"


def create_ui():
    """创建Gradio UI界面 (兼容Gradio 3.x/4.x)"""
    with gr.Blocks(title="XXG离线图片中文打标工具 Ver.2.3 (Qwen3-VL)") as demo:
        gr.Markdown("# 🖼️ Qwen3-VL 离线图片中文打标工具")
        gr.Markdown("### 100%中文caption生成 · 隐私安全 · 文生图训练专用")
        gr.Markdown("### By 西小瓜 / Wechat:priest-mos")

        with gr.Tabs():
            with gr.TabItem("🚀 批量打标"):
                with gr.Row():
                    with gr.Column(scale=3):
                        folder_input = gr.Textbox(
                            label="📁 图片文件夹路径",
                            placeholder="例如: C:/images 或 /home/user/photos",
                            value=os.path.join(os.path.expanduser("~"), "ai-toolkit/datasets/demo")
                        )
                        with gr.Row():
                             trigger_word= gr.Textbox(
                                label="默认触发词",
                                placeholder="不填写则会加入到标签中",
                                value=""
                            )

                        with gr.Row():
                            process_btn = gr.Button("🚀 开始生成中文caption", variant="primary")
                            stop_btn = gr.Button("🛑 停止", variant="stop")

                        with gr.Row():
                            use_4bit = gr.Checkbox(
                                label="启用4-bit量化 (低显存模式)",
                                value=False,
                                info="适用于6GB以下显存的GPU"
                            )
                            use_cpu = gr.Checkbox(
                                label="强制CPU模式",
                                value=False,
                                info="无GPU时使用"
                            )

                        output = gr.Textbox(label="📝 处理结果", lines=15, interactive=False)

                    with gr.Column(scale=2):
                        sys_info = gr.Markdown(label="🔧 系统信息")
                        refresh_btn = gr.Button("🔄 刷新系统信息", size="sm")
                        refresh_btn.click(fn=get_system_info, outputs=sys_info)

        demo.load(fn=get_system_info, outputs=sys_info)

        process_btn.click(
            fn=process_images,
            inputs=[folder_input,trigger_word, use_4bit, use_cpu],
            outputs=output,
            show_progress="full"
        )

        stop_btn.click(
            fn=lambda: "⏹️ 操作已停止",
            outputs=output
        )

        gr.Markdown("### 📝 使用指南")
        gr.Markdown("""
        #### **核心优势**
        - ✅ **100%中文caption**：非关键词堆砌，输出中文自然语句描述
        - ✅ **中文提示词文生图模型训练专用**：`一位年轻东亚女性,披散着乌黑波浪长发,佩戴宽檐编织草帽...`
        - ✅ **100%离线运行**：无任何网络请求

        #### **输出示例**
        ```
        一位年轻东亚女性,披散着乌黑波浪长发,佩戴宽檐编织草帽,身穿印有彩色鸟类图案的白色吊带连衣裙,身体微微侧向镜头,右手自然垂落握持物体,站在浅灰色墙面旁,头顶上方攀爬生长着茂密翠绿植物,阳光自前方柔和洒落照亮人物正面,整幅画面呈现清新明亮的日系户外摄影风格
        ```
        > ✅ 全中文描述 | ✅ 优化多种场景 | ✅ 包含构图/风格标签

        #### **操作步骤**
        1. 填写图片文件夹路径
        2. 低显存GPU：勾选"启用4-bit量化"
        3. 点击"🚀 开始生成中文caption"
        """)

        gr.Markdown(
            "<div style='text-align: center; margin-top: 20px; color: #888;'>"
            "© 2026 XXG离线图片中文打标工具（Qwen3-VL） | 100%中文caption · 隐私安全<br>"
            "核心技术: 强化中文prompt + 中文质量验证 + 离线模型"
            "</div>"
        )

    return demo


def main():
    """主函数"""
    global global_use_4bit

    parser = argparse.ArgumentParser(description='Qwen3-VL离线图片中文打标工具')
    parser.add_argument('--4bit', action='store_true', help='启用4-bit量化')
    parser.add_argument('--cpu', action='store_true', help='强制CPU模式')
    parser.add_argument('--port', type=int, default=9527, help='Web UI端口')
    parser.add_argument('--folder', type=str, help='直接处理文件夹')
    parser.add_argument('--trigger', type=str, help='默认触发词')
    args = parser.parse_args()

    global_use_4bit = args.__dict__['4bit']

    if args.__dict__['4bit']:
        print("⚡ 启动4-bit量化模式")
    if args.cpu:
        global device
        device = "cpu"
        print("💻 强制CPU模式")

    print("=" * 70)
    print("🖼️  XXG离线图片中文打标工具 Ver.2.3 (Qwen3-VL)")
    print("✅ 100%中文caption生成 | ✅ 本地模型化")
    print("=" * 70)

    check_system_resources()

    if args.folder:
        print(f"\n📁 直接处理文件夹: {args.folder}")
        result = process_images(args.folder,args.trigger, use_4bit=args.__dict__['4bit'], use_cpu=args.cpu)
        print("\n" + result)
        return

    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        show_error=True,
        quiet=True,
        theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 程序已安全退出")
    except Exception as e:
        print(f"❌ 严重错误: {str(e)}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)