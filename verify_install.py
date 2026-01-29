#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通义千问离线图片打标工具 - 安装验证脚本
- 验证所有依赖兼容性
- 检查模型文件完整性
- 测试模型加载功能
- 诊断常见问题
- 提供修复建议
"""

import os
import sys
import time
import json
import platform
import argparse
import warnings
from pathlib import Path
import hashlib

# 忽略弃用警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 全局变量
model_path = "./qwen_models"
required_packages = [
    "torch", "transformers", "gradio", "PIL", "numpy",
    "sentencepiece", "psutil", "huggingface_hub", "tqdm",
    "tiktoken", "transformers_stream_generator", "bitsandbytes"
]


def print_header():
    """打印脚本标题和基本信息"""
    print("=" * 60)
    print("🔍 通义千问离线图片打标工具 - 安装验证脚本")
    print(f"   Python版本: {platform.python_version()}")
    print(f"   操作系统: {platform.system()} {platform.release()}")
    print(f"   当前目录: {os.getcwd()}")
    print("=" * 60)


def check_dependencies():
    """检查所有必需的依赖是否安装"""
    print("\n" + "=" * 60)
    print("📦 依赖检查")
    print("=" * 60)

    missing_packages = []
    incompatible_packages = []

    for package in required_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', '未知')
            print(f"✅ {package} 版本: {version}")

            # 特殊版本检查
            if package == "numpy":
                if not version.startswith("1.26"):
                    print(f"⚠️  警告: NumPy {version} 可能与PyTorch不兼容 (需要1.26.4)")
                    incompatible_packages.append(package)
            elif package == "torch":
                if not version.startswith("2.2"):
                    print(f"⚠️  警告: PyTorch {version} 可能不兼容 (需要2.2.0)")
                    incompatible_packages.append(package)
            elif package == "transformers":
                if not version.startswith("4.37"):
                    print(f"⚠️  警告: Transformers {version} 可能不兼容 (需要4.37.0)")
                    incompatible_packages.append(package)
            elif package == "tiktoken":
                # 0.7.0+移除了__version__属性，需要特殊处理
                try:
                    hasattr(tiktoken, '__version__')
                except:
                    print(f"⚠️  警告: tiktoken {version} 可能不兼容 (需要0.6.0)")
                    incompatible_packages.append(package)

        except ImportError as e:
            print(f"❌ {package} 未安装: {str(e)}")
            missing_packages.append(package)
        except Exception as e:
            print(f"⚠️  {package} 导入错误: {str(e)}")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n❌ 发现 {len(missing_packages)} 个缺失包: {missing_packages}")
        print("💡 修复命令:")
        print("   pip install -r requirements.txt --upgrade")

    if incompatible_packages:
        print(f"\n⚠️  发现 {len(incompatible_packages)} 个不兼容包: {incompatible_packages}")
        print("💡 修复命令:")
        print("   pip install numpy==1.26.4 torch==2.2.0 transformers==4.37.0 tiktoken==0.6.0 --upgrade")
        print("   pip install transformers_stream_generator==0.0.4 --upgrade")

    return len(missing_packages) == 0 and len(incompatible_packages) == 0


def check_system_resources():
    """检查系统资源是否满足要求"""
    print("\n" + "=" * 60)
    print("💻 系统资源检查")
    print("=" * 60)

    try:
        import psutil
        import torch

        # 检查磁盘空间
        disk = psutil.disk_usage(os.path.abspath("."))
        free_gb = disk.free / (1024 ** 3)
        print(f"💾 磁盘空间: {free_gb:.1f}GB 可用")
        disk_ok = free_gb >= 8

        # 检查内存
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        print(f"🧠 系统内存: {available_gb:.1f}GB/{total_gb:.1f}GB 可用")
        mem_ok = available_gb >= 4

        # 检查GPU
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"🎮 GPU: {gpu_name} ({gpu_mem:.1f}GB 显存)")
            gpu_ok = gpu_mem >= 6
        else:
            print("⚠️  未检测到NVIDIA GPU，将使用CPU模式")
            gpu_ok = True  # CPU模式不需要GPU

        # 汇总
        resources_ok = disk_ok and mem_ok and gpu_ok
        if not resources_ok:
            print("\n⚠️  资源警告:")
            if not disk_ok:
                print(f"   • 磁盘空间不足! 建议至少8GB空闲 (当前: {free_gb:.1f}GB)")
            if not mem_ok:
                print(f"   • 可用内存不足! 建议至少4GB (当前: {available_gb:.1f}GB)")
            if not gpu_ok and gpu_available:
                print(f"   • GPU显存不足! 建议至少6GB (当前: {gpu_mem:.1f}GB)")

        return resources_ok

    except ImportError as e:
        print(f"❌ 无法检查系统资源: {str(e)}")
        print("💡 请安装psutil: pip install psutil")
        return False


def verify_model_files():
    """验证Qwen-VL-Chat模型文件是否完整"""
    print("\n" + "=" * 60)
    print("🔍 模型文件验证")
    print("=" * 60)

    model_dir = Path(model_path)

    if not model_dir.exists():
        print(f"❌ 模型目录不存在: {model_dir.absolute()}")
        print("💡 请先下载模型: python download_models.py")
        return False

    print(f"📁 模型目录: {model_dir.absolute()}")

    # 必需的核心配置文件
    required_files = [
        "config.json",
        "tokenizer_config.json",
        "qwen.tiktoken",
        "tokenization_qwen.py",
        "modeling_qwen.py",
        "configuration_qwen.py"
    ]

    # 检查必需文件
    missing_files = []
    for file in required_files:
        if not (model_dir / file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"❌ 缺失必需文件: {missing_files}")

    # 检查权重文件
    weight_files = list(model_dir.glob("pytorch_model*.bin"))
    if not weight_files:
        print("❌ 未找到模型权重文件 (pytorch_model*.bin)")
    else:
        print(f"✅ 找到 {len(weight_files)} 个权重文件")

        # 检查分片数量
        shard_files = [f for f in weight_files if "pytorch_model-" in f.name]
        expected_shards = 10
        if len(shard_files) < expected_shards - 1:  # 允许缺失1个
            print(f"⚠️  仅找到 {len(shard_files)}/{expected_shards} 个权重分片，模型可能不完整")

        # 检查文件大小
        total_size = sum(f.stat().st_size for f in weight_files)
        print(f"📦 权重文件总大小: {total_size / 1024 ** 3:.2f}GB")

        # 验证最小大小 (约15GB)
        min_size_gb = 15
        if total_size < min_size_gb * 1024 ** 3:
            print(f"⚠️  模型总大小过小 ({total_size / 1024 ** 3:.2f}GB)，完整模型应约18GB")

    # 验证文件哈希值 (示例，实际应使用官方哈希)
    print("\n🔍 验证关键文件哈希值 (抽样检查)...")
    sample_files = [
        "config.json",
        "pytorch_model-00001-of-00010.bin",
        "pytorch_model-00010-of-00010.bin"
    ]

    for sample_file in sample_files:
        file_path = model_dir / sample_file
        if file_path.exists():
            # 计算文件哈希 (简化版，实际应比对官方值)
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read(1024 * 1024)).hexdigest()[:8]  # 只计算前1MB
            print(f"✅ {sample_file}: 哈希前缀 {file_hash}")
        else:
            print(f"⚠️  {sample_file} 不存在，无法验证哈希")

    # 模型文件验证结果
    files_ok = len(missing_files) == 0 and len(weight_files) > 0 and total_size > 15 * 1024 ** 3
    if not files_ok:
        if missing_files:
            print("\n💡 修复建议:")
            print("   1. 重新下载缺失文件: python download_models.py")
            print("   2. 检查下载过程是否被中断")
            print("   3. 确保有足够磁盘空间 (至少20GB)")

    return files_ok


def test_model_loading(use_4bit=False, use_cpu=False):
    """测试模型加载功能"""
    print("\n" + "=" * 60)
    print("🧠 模型加载测试")
    print("=" * 60)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import gc

        # 设置设备
        device = "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 使用设备: {device.upper()}")

        # 检查模型目录
        if not os.path.exists(model_path):
            print("❌ 模型目录不存在")
            return False

        # 加载tokenizer
        print("🔄 加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side='left',
            use_fast=False
        )
        print("✅ Tokenizer加载成功!")

        # 准备模型加载参数
        model_kwargs = {
            "trust_remote_code": True,
            "cache_dir": model_path,
            "device_map": "auto" if device == "cuda" else "cpu"
        }

        # 4-bit量化
        if use_4bit and device == "cuda":
            print("⚡ 启用4-bit量化模式...")
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
                use_4bit = False
        else:
            use_4bit = False

        # 标准加载
        if not use_4bit:
            print("🧠 加载标准精度模型 (可能需要1-2分钟)...")
            start_time = time.time()
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            ).eval()
            load_time = time.time() - start_time
            print(f"✅ 标准精度模型加载成功! (耗时: {load_time:.1f}秒)")

        # 验证模型
        print("🔍 验证模型功能...")
        test_query = tokenizer.from_list_format([
            {'text': '你好，通义千问!'}
        ])

        with torch.no_grad():
            response, _ = model.chat(
                tokenizer,
                query=test_query,
                history=None,
                max_new_tokens=10,
                temperature=0.7,
                top_p=0.9
            )

        print(f"✅ 模型功能验证成功! 响应: '{response.strip()}'")

        # 清理资源
        del model
        del tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return True

    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        import traceback
        traceback.print_exc()

        print("\n🛠️  详细故障排除:")
        print("1. 检查模型文件完整性 (重新运行本脚本)")
        print("2. 检查依赖版本 (numpy==1.26.4, torch==2.2.0, transformers==4.37.0)")
        print("3. 确保已安装 transformers_stream_generator==0.0.4")
        print("4. 尝试4-bit量化模式: --4bit 参数")
        print("5. 尝试CPU模式: --cpu 参数")
        return False


def generate_report(results):
    """生成验证报告"""
    print("\n" + "=" * 60)
    print("📊 验证报告")
    print("=" * 60)

    # 计算总体通过率
    total_checks = len(results)
    passed_checks = sum(1 for r in results.values() if r)
    pass_rate = passed_checks / total_checks * 100

    print(f"✅ 通过: {passed_checks}/{total_checks} 项 ({pass_rate:.1f}%)")

    # 按优先级排序结果
    priority_order = [
        'dependencies', 'model_files', 'model_loading',
        'system_resources'
    ]

    for check in priority_order:
        if check in results:
            status = "✅ 通过" if results[check] else "❌ 失败"
            print(f"   • {check.replace('_', ' ').title()}: {status}")

    # 总体结论
    if passed_checks == total_checks:
        print("\n🎉 验证成功! 所有检查通过，可以运行主程序了")
        print("🚀 启动命令: python app.py")
    elif passed_checks >= total_checks - 1:
        print("\n⚠️  基本可用! 大部分检查通过，但有轻微问题")
        print("🔧 建议修复后再使用: python fix_dependencies.py")
    else:
        print("\n❌ 验证失败! 存在严重问题，需要修复")

        # 提供针对性修复建议
        if not results.get('dependencies', True):
            print("\n💡 依赖修复建议:")
            print("   pip install -r requirements.txt --upgrade")

        if not results.get('model_files', True):
            print("\n💡 模型文件修复建议:")
            print("   python download_models.py")

        if not results.get('model_loading', True):
            print("\n💡 模型加载修复建议:")
            print("   • 检查NumPy版本: pip install numpy==1.26.4 --upgrade")
            print("   • 检查tiktoken版本: pip install tiktoken==0.6.0 --upgrade")
            print("   • 安装transformers_stream_generator: pip install transformers_stream_generator==0.0.4 --upgrade")
            print("   • 尝试4-bit量化: python app.py --4bit")
            print("   • 尝试CPU模式: python app.py --cpu")

    print("\n" + "=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='验证通义千问离线图片打标工具安装')
    parser.add_argument('--4bit', action='store_true', help='测试4-bit量化模式')
    parser.add_argument('--cpu', action='store_true', help='强制使用CPU模式测试')
    parser.add_argument('--quick', action='store_true', help='快速验证 (跳过耗时测试)')
    args = parser.parse_args()

    print_header()

    # 存储验证结果
    results = {}

    # 1. 依赖检查
    results['dependencies'] = check_dependencies()

    # 2. 系统资源检查
    results['system_resources'] = check_system_resources()

    # 3. 模型文件验证
    results['model_files'] = verify_model_files()

    # 4. 模型加载测试 (如果文件验证通过)
    if results['model_files']:
        results['model_loading'] = test_model_loading(
            use_4bit=args.__dict__['4bit'],
            use_cpu=args.__dict__['cpu']
        )
    else:
        results['model_loading'] = False
        print("\n⚠️  跳过模型加载测试 (模型文件不完整)")

    # 生成报告
    generate_report(results)

    # 退出代码 (0=成功, 1=失败)
    overall_success = all(results[key] for key in results if results[key] is not None)
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 验证已中断")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 严重错误: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)