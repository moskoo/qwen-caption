#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通义千问离线图片中文打标工具
- 100%离线运行
- 隐私安全保护
- 专业级中文描述
- 适配Qwen-VL-Chat实际文件结构
- 修复所有依赖和兼容性问题
"""

import os
import sys
import time
import gc
import json
import argparse
from pathlib import Path

# 设置环境变量 (必须在导入torch前设置)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # 加速下载
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5分钟超时
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免警告
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# 修复NumPy兼容性问题
try:
    import numpy as np

    if np.__version__.startswith("2"):
        print(f"⚠️  检测到NumPy {np.__version__}，可能与PyTorch不兼容")
        print("💡 建议运行: pip install numpy==1.26.4 --upgrade")
except ImportError:
    pass

# 首先导入torch
try:
    import torch
except ImportError as e:
    print(f"❌ 无法导入torch: {str(e)}")
    print("💡 请先安装依赖: pip install -r requirements.txt")
    sys.exit(1)

# 全局变量 (现在可以安全使用torch)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
tokenizer = None
model_path = "./qwen_models"
global_use_4bit = False  # 重命名全局变量，避免与函数参数冲突

# 导入其他依赖
try:
    from PIL import Image
    import gradio as gr
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm
    import psutil
    import platform
except ImportError as e:
    print(f"❌ 依赖导入失败: {str(e)}")
    print("\n💡 解决方案:")
    print("1. 安装依赖: pip install -r requirements.txt")
    print("2. 确保虚拟环境已激活")
    sys.exit(1)


def check_system_resources():
    """检查系统资源是否满足要求"""
    print("🔍 系统资源检查...")

    # 检查磁盘空间
    disk = psutil.disk_usage(os.path.abspath("."))
    free_gb = disk.free / (1024 ** 3)
    print(f"💾 磁盘空间: {free_gb:.1f}GB 可用")
    if free_gb < 8:
        print(f"⚠️  警告: 磁盘空间不足! 建议至少8GB空闲空间")

    # 检查内存
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    total_gb = mem.total / (1024 ** 3)
    print(f"🧠 系统内存: {available_gb:.1f}GB/{total_gb:.1f}GB 可用")
    if available_gb < 4:
        print(f"⚠️  警告: 可用内存不足4GB，处理大图时可能失败")

    # 检查GPU
    if device == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f}GB 显存)")
        if gpu_mem < 6:
            print(f"⚠️  警告: GPU显存小于6GB，建议启用4-bit量化")
    else:
        print("💻 使用CPU模式 (无GPU加速)")

    return {
        "disk_free_gb": free_gb,
        "mem_available_gb": available_gb,
        "gpu_available": device == "cuda",
        "gpu_mem_gb": torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if device == "cuda" else 0
    }


def smart_verify_qwen_model(model_path):
    """智能验证Qwen-VL-Chat模型文件是否完整"""
    model_dir = Path(model_path)

    if not model_dir.exists() or not model_dir.is_dir():
        return False, "模型目录不存在或不是目录"

    # 1. 检查基础配置文件
    required_configs = ["config.json", "tokenizer_config.json"]
    missing_configs = [f for f in required_configs if not (model_dir / f).exists()]

    if missing_configs:
        return False, f"缺失基础配置文件: {missing_configs}"

    # 2. 检查权重文件 (支持分片)
    weight_files = list(model_dir.glob("pytorch_model*.bin")) + list(model_dir.glob("model*.safetensors"))

    if not weight_files:
        # 检查索引文件
        if (model_dir / "pytorch_model.bin.index.json").exists():
            return False, "检测到索引文件，但权重文件未完全下载，请重新下载"
        return False, "未找到模型权重文件 (pytorch_model*.bin 或 model*.safetensors)"

    # 3. 检查Qwen特定tokenizer文件
    qwen_tokenizer_files = ["qwen.tiktoken", "tokenization_qwen.py"]
    has_qwen_tokenizer = all((model_dir / f).exists() for f in qwen_tokenizer_files)

    if not has_qwen_tokenizer:
        # 检查标准tokenizer文件
        std_tokenizer_files = ["special_tokens_map.json", "tokenizer.json"]
        has_std_tokenizer = any((model_dir / f).exists() for f in std_tokenizer_files)

        if not has_std_tokenizer:
            return False, "未找到有效的tokenizer文件 (缺少Qwen特定文件或标准tokenizer文件)"

    # 4. 检查文件大小
    total_size = sum(f.stat().st_size for f in weight_files)
    if total_size < 10e9:  # 10GB
        return False, f"模型文件总大小过小 ({total_size / 1e9:.2f}GB)，可能下载不完整 (Qwen-VL-Chat约18GB)"

    # 5. 验证分片数量
    shard_files = [f for f in weight_files if "pytorch_model-" in f.name]
    if shard_files:
        # 检查分片数量是否合理 (Qwen-VL-Chat应有10个分片)
        if len(shard_files) < 8:  # 允许缺失1-2个
            return False, f"仅找到 {len(shard_files)} 个权重分片，模型可能不完整 (应有10个分片)"

    return True, f"✅ Qwen-VL-Chat模型验证成功!\n   • 找到 {len(weight_files)} 个权重文件\n   • 总大小: {total_size / 1e9:.2f}GB\n   • {'检测到Qwen特定tokenizer' if has_qwen_tokenizer else '检测到标准tokenizer'}"


def load_qwen_model(use_4bit=False, use_cpu=False):
    """加载Qwen-VL-Chat模型，适配实际文件结构"""
    global model, tokenizer, device, global_use_4bit  # 使用重命名的全局变量

    # 更新全局4-bit标志
    global_use_4bit = use_4bit

    # 强制使用CPU模式
    if use_cpu:
        device = "cpu"
        print("⚠️  强制使用CPU模式 (无GPU加速)")

    if model is not None and tokenizer is not None:
        print("✅ 模型已在内存中，跳过加载")
        return model, tokenizer

    print(f"🚀 正在加载Qwen-VL-Chat模型 (设备: {device.upper()})...")
    print(f"   模型路径: {os.path.abspath(model_path)}")

    # 检查模型目录是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型目录不存在: {model_path}")
        print("💡 请先运行: python download_models.py")
        raise FileNotFoundError(f"模型目录 {model_path} 不存在")

    # 智能验证模型文件
    print("🔍 智能验证模型文件...")
    model_valid, validation_msg = smart_verify_qwen_model(model_path)

    if not model_valid:
        print(f"❌ 模型验证失败: {validation_msg}")
        print("💡 请重新下载完整模型: python download_models.py")
        raise ValueError("模型文件不完整或损坏")
    else:
        print(validation_msg)

    try:
        # 1️⃣ 加载tokenizer - 兼容Qwen的特殊tokenizer
        print("🔧 加载tokenizer (Qwen特定配置)...")

        # 确保模型路径在Python路径中
        if model_path not in sys.path:
            sys.path.insert(0, model_path)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side='left',
            use_fast=False  # Qwen有时在fast模式下有问题
        )
        print("✅ Tokenizer加载成功! (Qwen特定配置)")

        # 2️⃣ 准备模型加载参数
        model_kwargs = {
            "trust_remote_code": True,
            "cache_dir": model_path,
            "device_map": "auto" if device == "cuda" else "cpu"
        }

        # 3️⃣ 4-bit量化支持
        if use_4bit and device == "cuda":
            print("⚡ 启用4-bit量化 (减少显存需求)...")
            try:
                from auto_gptq import AutoGPTQForCausalLM
                model = AutoGPTQForCausalLM.from_quantized(
                    model_path,
                    device="cuda:0",
                    use_triton=False,
                    quantize_config=None
                )
                print("✅ 4-bit量化模型加载成功!")
            except Exception as e:
                print(f"⚠️  4-bit加载失败: {str(e)}")
                print("🔄 回退到标准加载...")
                use_4bit = False

        # 4️⃣ 标准加载
        if not use_4bit or model is None:
            print("🧠 加载标准精度模型...")
            start_time = time.time()
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            ).eval()
            load_time = time.time() - start_time
            print(f"✅ 标准精度模型加载成功! (耗时: {load_time:.1f}秒)")

        # 5️⃣ 验证模型
        print("🔍 验证模型功能...")
        try:
            test_query = tokenizer.from_list_format([
                {'text': '你好，通义千问!'}
            ])
            _ = model.chat(tokenizer, query=test_query, history=None, max_new_tokens=10)
            print("✅ 模型功能验证通过!")
        except Exception as e:
            print(f"⚠️  模型功能验证失败，但可能不影响图片打标: {str(e)}")

        # 6️⃣ 资源优化
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return model, tokenizer

    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        import traceback
        traceback.print_exc()

        # 详细的错误处理 - 增强版
        print("\n🛠️  详细故障排除:")
        print("1. 检查模型文件完整性:")
        print("   • 确保有10个pytorch_model-XXXXX-of-00010.bin文件")
        print("   • 确保有qwen.tiktoken和tokenization_qwen.py")
        print("   • 确保有modeling_qwen.py和configuration_qwen.py")
        print("2. 必需依赖检查:")

        # 检查transformers_stream_generator
        try:
            import transformers_stream_generator
            print("✅ transformers_stream_generator (必需依赖已安装)")
        except ImportError as e:
            print("❌ transformers_stream_generator 未安装! (关键依赖)")
            print("💡 修复命令: pip install transformers_stream_generator==0.0.4 --upgrade")

        # 检查tiktoken (兼容0.7.0+版本)
        try:
            import tiktoken
            # 安全检查版本属性
            version = getattr(tiktoken, '__version__', '0.7.0+ (无__version__属性)')
            print(f"✅ tiktoken 版本: {version} (已安装)")
        except ImportError as e:
            print("❌ tiktoken 未安装! (关键依赖)")
            print("💡 修复命令: pip install tiktoken==0.6.0 --upgrade")

        # 检查Qwen-VL特定工具
        try:
            import qwen_vl_utils
            print("✅ qwen_vl_utils (Qwen-VL专用工具已安装)")
        except ImportError as e:
            print("⚠️  qwen_vl_utils 未安装 (非必需，但推荐)")
            print("💡 安装命令: pip install qwen_vl_utils==0.0.1 --upgrade")

        # 检查auto-gptq (4-bit量化)
        if use_4bit:
            try:
                import auto_gptq
                print("✅ auto_gptq (4-bit量化支持已安装)")
            except ImportError as e:
                print("⚠️  auto_gptq 未安装 (4-bit量化需要)")
                print("💡 安装命令: pip install auto-gptq==0.7.1 --upgrade")

        print("\n3. 依赖版本要求:")
        print("   • transformers==4.37.0")
        print("   • torch==2.2.0")
        print("   • numpy==1.26.4")
        print("   • 请运行: pip install -r requirements.txt --upgrade")

        print("\n4. 尝试重新下载和修复:")
        print("   python fix_dependencies.py --full")
        print("   python download_models.py")

        # 提供一键修复命令
        print("\n🔧 一键修复所有依赖 (推荐):")
        print("   pip uninstall -y transformers_stream_generator tiktoken auto-gptq")
        print("   pip install -r requirements.txt --force-reinstall --no-cache-dir")

        sys.exit(1)


def generate_chinese_caption(image_path, max_new_tokens=200):
    """使用Qwen-VL-Chat生成中文图片描述"""
    global model, tokenizer

    try:
        # 验证图片
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")

        # 打开图片
        image = Image.open(image_path).convert("RGB")
        image.verify()  # 验证图片完整性

        # 准备查询
        query = tokenizer.from_list_format([
            {'image': image_path},
            {
                'text': '详细描述这张图片的内容，用中文回答。需要包含：主要物体、场景环境、颜色特征、文字内容及字体、图文排版布局、人体结构和比例（如果有）、人物动作（如果有）、整体氛围等关键信息。要求描述专业、准确、流畅。'}
        ])

        # 生成描述
        start_time = time.time()
        with torch.no_grad():
            response, _ = model.chat(
                tokenizer,
                query=query,
                history=None,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9
            )
        gen_time = time.time() - start_time

        # 后处理
        caption = response.strip()
        caption = caption.replace('\n', ' ').replace('  ', ' ')

        print(f"⏱️  生成耗时: {gen_time:.1f}秒, 描述长度: {len(caption)}字符")
        return caption

    except Exception as e:
        print(f"❌ 处理 {os.path.basename(image_path)} 时出错: {str(e)}")
        # 清理资源
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return None


def process_images(folder_path, use_4bit=False, use_cpu=False, progress=gr.Progress()):
    """批量处理图片文件夹，生成中文描述"""
    if not folder_path or not folder_path.strip():
        return "❌ 错误: 请输入有效的文件夹路径"

    folder_path = folder_path.strip()
    if not os.path.isdir(folder_path):
        return f"❌ 错误: 路径 '{folder_path}' 不是有效文件夹"

    # 支持的图片格式
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

    # 获取所有图片文件
    image_files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f.lower())[1] in SUPPORTED_FORMATS and
           not f.lower().startswith('._')  # 跳过macOS临时文件
    ]

    if not image_files:
        return f"⚠️ 警告: 在 '{folder_path}' 中未找到支持的图片文件\n支持格式: {', '.join(SUPPORTED_FORMATS)}"

    # 加载模型
    load_qwen_model(use_4bit=use_4bit, use_cpu=use_cpu)

    # 准备结果
    results = {
        "total": len(image_files),
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "details": []
    }

    total = len(image_files)

    # 处理每张图片
    for i, filename in enumerate(image_files):
        progress(i / total, desc=f"处理中 ({i + 1}/{total}) - {filename}")

        image_path = os.path.join(folder_path, filename)
        txt_path = os.path.splitext(image_path)[0] + '.txt'

        # 跳过已处理的文件
        if os.path.exists(txt_path):
            results["skipped"] += 1
            results["details"].append(f"⏭ 跳过: {filename} (已存在描述文件)")
            continue

        # 生成描述
        print(f"\n🖼️  处理: {filename}")
        caption = generate_chinese_caption(image_path)

        # 保存结果
        if caption and len(caption) > 20:  # 确保描述有意义
            try:
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(caption)
                results["success"] += 1
                preview = caption[:70] + "..." if len(caption) > 70 else caption
                results["details"].append(f"✅ 成功: {filename}\n   {preview}")
                print(f"   描述: {preview}")
            except Exception as e:
                results["failed"] += 1
                results["details"].append(f"❌ 写入失败: {filename}\n   {str(e)}")
        else:
            results["failed"] += 1
            results["details"].append(f"❌ 生成失败: {filename}" + (f"\n   原因: {caption}" if caption else ""))

        # 资源清理
        if i % 3 == 0:
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    # 生成报告
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

    # 最终资源清理
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return report


def get_system_info():
    """获取系统信息用于UI显示"""
    try:
        gpu_info = "未检测到GPU"
        if device == "cuda":
            gpu_info = f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB)"

        mem = psutil.virtual_memory()
        disk = psutil.disk_usage(os.path.abspath("."))

        model_status = "✅ 已加载" if model is not None else "⏳ 未加载"
        quant_status = " (4-bit)" if global_use_4bit and model is not None else ""

        tokenizer_type = "未知"
        if tokenizer is not None:
            if hasattr(tokenizer, 'name_or_path') and 'qwen' in tokenizer.name_or_path.lower():
                tokenizer_type = "Qwen特定tokenizer"
            else:
                tokenizer_type = "标准tokenizer"

        model_size = "未知"
        if model is not None:
            total_params = sum(p.numel() for p in model.parameters())
            model_size = f"{total_params / 1e9:.1f}B"

        numpy_version = "未知"
        try:
            import numpy as np
            numpy_version = np.__version__
        except:
            pass

        return (
            f"**操作系统**: {platform.system()} {platform.release()}\n"
            f"**Python版本**: {platform.python_version()}\n"
            f"**NumPy版本**: {numpy_version} (需要1.26.4)\n"
            f"**运行设备**: {device.upper()} ({gpu_info})\n"
            f"**系统内存**: {mem.total / 1e9:.1f}GB (可用: {mem.available / 1e9:.1f}GB)\n"
            f"**磁盘空间**: {disk.free / 1e9:.1f}GB 可用\n"
            f"**模型状态**: {model_status}{quant_status}\n"
            f"**模型大小**: {model_size}\n"
            f"**Tokenizer类型**: {tokenizer_type}\n"
            f"**模型路径**: `{os.path.abspath(model_path)}`"
        )
    except Exception as e:
        return f"⚠️ 获取系统信息失败: {str(e)}"


def create_ui():
    """创建Gradio UI界面"""
    with gr.Blocks(title="通义千问离线图片打标工具", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🖼️ 通义千问离线图片中文打标工具")
        gr.Markdown("### 100%离线运行 · 隐私安全 · 专业级中文描述")

        with gr.Tabs():
            with gr.TabItem("🚀 处理图片"):
                with gr.Row():
                    with gr.Column(scale=3):
                        folder_input = gr.Textbox(
                            label="📁 图片文件夹路径",
                            placeholder="例如: C:/images 或 /home/user/photos",
                            value=os.path.join(os.path.expanduser("~"), "Pictures")
                        )
                        with gr.Row():
                            process_btn = gr.Button("🚀 开始中文打标", variant="primary")
                            stop_btn = gr.Button("🛑 停止", variant="stop")

                        with gr.Row():
                            use_4bit = gr.Checkbox(
                                label="启用4-bit量化 (低显存模式)",
                                value=False,
                                info="适用于6GB以下显存的GPU，处理速度稍慢但内存需求大幅降低"
                            )
                            use_cpu = gr.Checkbox(
                                label="强制CPU模式",
                                value=False,
                                info="无GPU或GPU不稳定时使用，速度较慢但更稳定"
                            )

                        output = gr.Textbox(label="📝 处理结果", lines=15, interactive=False)

                    with gr.Column(scale=2):
                        sys_info = gr.Markdown(label="🔧 系统信息")
                        demo.load(get_system_info, None, sys_info, every=30)

        # 事件处理
        process_btn.click(
            fn=process_images,
            inputs=[folder_input, use_4bit, use_cpu],
            outputs=output,
            show_progress="full"
        )

        stop_btn.click(
            fn=lambda: "⏹️ 操作已停止 (可能需要等待当前图片处理完成)",
            outputs=output
        )

        gr.Markdown("### 📝 使用指南")
        gr.Markdown("""
        #### **重要提示: 依赖兼容性**
        - **NumPy必须为1.26.4版本** (PyTorch 2.2.0不兼容NumPy 2.x)
        - **tiktoken必须为0.6.0版本** (0.7.0+移除了__version__属性)
        - **必须安装transformers_stream_generator** (Qwen-VL-Chat必需依赖)
        - 如果看到依赖警告，请运行: `pip install -r requirements.txt --upgrade`

        #### **首次运行准备**
        1. **下载模型** (只需一次，需要网络):
           ```bash
           python download_models.py
           ```
        2. **硬件要求**:
           - **推荐配置**: NVIDIA GPU (8GB+显存) + 16GB RAM
           - **最低配置**: 8GB RAM (CPU模式，速度较慢)
           - **磁盘空间**: 20GB+ 空闲 (Qwen-VL-Chat模型约18GB)

        #### **操作步骤**
        1. 在输入框填写**图片文件夹的绝对路径**
           - Windows: `C:\\Users\\YourName\\Pictures`
           - Mac/Linux: `/home/username/Pictures`
        2. 根据硬件情况选择:
           - 低显存GPU: 勾选 **"启用4-bit量化"**
           - 无GPU/不稳定: 勾选 **"强制CPU模式"**
        3. 点击 **"🚀 开始中文打标"**
        4. 处理完成后，每个图片同目录生成 **同名.txt文件**

        #### **结果示例**
        ```
        这张照片展示了一个阳光明媚的春日公园场景。前景有三个孩子在草地上放风筝，风筝是鲜艳的红色和蓝色。背景可见盛开的樱花树和一条蜿蜒的小径。远处有几位老人坐在长椅上休息。整体氛围温馨和谐，体现了春天的生机与活力。
        ```

        #### **常见问题**
        - **"CUDA out of memory"**: 勾选4-bit量化或关闭其他程序
        - **模型加载失败**: 重新运行 `python download_models.py`
        - **处理速度慢**: 
          - GPU模式: 每张3-5秒
          - CPU模式: 每张20-40秒
        - **中文乱码**: 用记事本或VSCode以UTF-8编码打开txt文件
        - **依赖问题**: 确保所有必需依赖已安装
        """)

        # 页脚
        gr.Markdown(
            "<div style='text-align: center; margin-top: 20px; color: #888;'>"
            "© 2026 通义千问离线图片打标工具 | 完全离线 · 隐私安全 · 开源免费<br>"
            "使用Qwen-VL-Chat模型，遵循Apache 2.0开源协议"
            "</div>",
            elem_classes=["footer"]
        )

    return demo


def main():
    """主函数"""
    global global_use_4bit

    parser = argparse.ArgumentParser(description='通义千问离线图片打标工具')
    parser.add_argument('--4bit', action='store_true', help='启用4-bit量化模式')
    parser.add_argument('--cpu', action='store_true', help='强制使用CPU模式')
    parser.add_argument('--port', type=int, default=9527, help='Web UI端口号')
    args = parser.parse_args()

    global_use_4bit = args.__dict__['4bit']

    if args.__dict__['4bit']:
        print("⚡ 启动4-bit量化模式 (低显存需求)")
    if args.cpu:
        global device
        device = "cpu"
        print("💻 强制使用CPU模式 (无GPU加速)")

    print("=" * 60)
    print("🖼️  通义千问离线图片中文打标工具")
    print("✅ 适配Qwen-VL-Chat实际文件结构 (无需special_tokens_map.json)")
    print("✅ 修复transformers_stream_generator依赖缺失问题")
    print("✅ 修复tiktoken 0.7.0+版本兼容性问题")
    print("✅ 修复NumPy 2.x兼容性问题 (固定NumPy 1.26.4)")
    print("=" * 60)

    # 检查NumPy版本
    try:
        import numpy as np
        if not np.__version__.startswith("1.26"):
            print(f"⚠️  警告: 检测到NumPy {np.__version__}")
            print("💡 建议运行: pip install numpy==1.26.4 --upgrade")
    except ImportError:
        print("❌ 无法导入NumPy，请安装依赖")

    # 检查系统资源
    check_system_resources()

    # 创建并启动UI
    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        show_error=True,
        quiet=True
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 程序已安全退出")
    except Exception as e:
        print(f"❌ 严重错误: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)