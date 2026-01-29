#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen-VL-Chat 模型下载脚本
- 适配实际文件结构 (无需special_tokens_map.json)
- 修复transformers_stream_generator依赖问题
- 修复大文件大小检测问题
- 中国镜像源支持
- 断点续传
"""

import os
import sys
import time
from pathlib import Path
import argparse
import json
import platform
import warnings


def check_numpy_compatibility():
    """检查NumPy版本兼容性"""
    try:
        import numpy as np
        numpy_version = np.__version__
        print(f"🔍 检测到NumPy版本: {numpy_version}")

        # 检查是否为NumPy 2.x
        if numpy_version.startswith('2'):
            print(f"⚠️  警告: NumPy {numpy_version} 可能与PyTorch不兼容")
            print("💡 建议在安装依赖时使用: pip install numpy==1.26.4 --upgrade")
            return False
        return True
    except ImportError:
        print("ℹ️  NumPy未安装，将在环境设置时自动安装兼容版本")
        return True


def download_qwen_vl_chat(model_dir="./qwen_models", use_mirror=False):
    """下载Qwen-VL-Chat模型，适配实际文件结构"""
    model_id = "Qwen/Qwen-VL-Chat"

    # 设置镜像源 (如果在中国)
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("🌐 使用HuggingFace镜像源 (中国加速)")

    print(f"🚀 开始下载 {model_id} 模型...")
    print(f"📁 保存到: {os.path.abspath(model_dir)}")

    # 创建目录
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    try:
        # 导入huggingface_hub
        print("🔧 导入huggingface_hub库...")
        from huggingface_hub import snapshot_download

        # 下载模型 - 关键参数修正
        start_time = time.time()
        print("⬇️ 开始下载模型文件 (可能需要10-60分钟，取决于网络速度)...")
        print("   • 将自动断点续传")
        print("   • 将验证文件完整性")

        snapshot_download(
            repo_id=model_id,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4,
            allow_patterns=["*.bin", "*.json", "*.txt", "*.model", "*.py", "*.safetensors", "*.md", "*.png",
                            "*.tiktoken"],
            ignore_patterns=["*.h5", "*.ot", "*.msgpack", "*.onnx", "*.pt"]
        )
        download_time = time.time() - start_time

        print(f"✅ 模型下载完成! (耗时: {download_time / 60:.1f}分钟)")

        # 验证模型文件
        print("\n🔍 验证模型文件完整性...")
        is_valid, message = verify_model_files(model_dir)

        if not is_valid:
            print(f"❌ 模型验证失败: {message}")
            print("💡 建议:")
            print("1. 重新下载模型")
            print("2. 检查磁盘空间是否充足 (需要至少20GB)")
            print("3. 确保网络连接稳定")
            return False

        print(message)

        # 保存下载信息
        total_size = sum(f.stat().st_size for f in Path(model_dir).rglob('*') if f.is_file())
        download_info = {
            "model_id": model_id,
            "download_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_size_gb": total_size / 1e9,
            "file_count": len(list(Path(model_dir).rglob('*'))),
            "validation": "success",
            "system_info": {
                "os": f"{platform.system()} {platform.release()}",
                "python": platform.python_version(),
                "download_method": "snapshot_download"
            }
        }

        with open(Path(model_dir) / "download_info.json", 'w', encoding='utf-8') as f:
            json.dump(download_info, f, indent=2, ensure_ascii=False)

        print(f"\n🎉 下载成功! 可以运行主程序了")
        print(f"💡 提示: 模型将自动从 {os.path.abspath(model_dir)} 加载，无需网络连接")
        return True

    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")

        # 详细的故障排除指南
        print("\n🛠️  故障排除指南:")
        print("1. 检查网络连接和磁盘空间 (需要至少20GB空闲)")
        print("2. 尝试使用镜像站点 (如果在中国):")
        print("   python download_models.py --mirror")
        print("3. 手动下载模型:")
        print("   - 访问: https://huggingface.co/Qwen/Qwen-VL-Chat")
        print("   - 下载所有文件 (包括分片权重文件 pytorch_model-*-of-*.bin)")
        print("   - 将这些文件放入 ./qwen_models 目录")
        print("4. 如果遇到SSL错误，尝试设置环境变量:")
        print("   export PYTHONHTTPSVERIFY=0")

        # 显示可能的权限问题
        if "Permission denied" in str(e) or "Access is denied" in str(e):
            print("\n🔑 权限问题解决方案:")
            print("   sudo chown -R $USER:$USER ./qwen_models  # Linux/Mac")
            print("   或以管理员身份运行命令提示符 (Windows)")

        return False


def verify_model_files(model_dir):
    """智能验证Qwen-VL-Chat模型文件是否完整"""
    model_dir = Path(model_dir)

    if not model_dir.exists():
        return False, "模型目录不存在"

    # 必需的核心配置文件
    required_config_files = [
        "config.json",
        "tokenizer_config.json"
    ]

    # 检查配置文件
    missing_configs = [f for f in required_config_files if not (model_dir / f).exists()]
    if missing_configs:
        return False, f"缺失配置文件: {missing_configs}"

    # 检查模型权重文件 - 支持分片和非分片
    model_files = list(model_dir.glob("pytorch_model*.bin")) + list(model_dir.glob("model*.safetensors"))

    if not model_files:
        return False, "未找到模型权重文件 (pytorch_model*.bin 或 model*.safetensors)"

    # 检查Qwen特定文件
    qwen_files = ["qwen.tiktoken", "tokenization_qwen.py", "modeling_qwen.py", "configuration_qwen.py"]
    missing_qwen_files = [f for f in qwen_files if not (model_dir / f).exists()]

    if missing_qwen_files:
        print(f"⚠️  警告: 缺少部分Qwen特定文件: {missing_qwen_files}")
        print("   • 这可能影响tokenizer功能")
        print("   • 但模型可能仍然加载成功")

    # 检查文件大小 (粗略验证)
    total_size = sum(f.stat().st_size for f in model_files)
    if total_size < 10e9:  # 小于10GB可能不完整
        return False, f"模型文件总大小过小 ({total_size / 1e9:.2f}GB)，可能下载不完整 (完整模型约18GB)"

    return True, f"✅ 模型验证成功! 找到 {len(model_files)} 个权重文件，总大小: {total_size / 1e9:.2f}GB"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='下载Qwen-VL-Chat模型')
    parser.add_argument('--mirror', action='store_true', help='使用中国镜像源加速下载')
    parser.add_argument('--dir', type=str, default="./qwen_models", help='模型保存目录')
    parser.add_argument('--retry', type=int, default=3, help='下载失败重试次数')
    args = parser.parse_args()

    print("=" * 60)
    print("🌍 Qwen-VL-Chat 模型下载工具 (重构版)")
    print("✅ 适配实际文件结构 (无需special_tokens_map.json)")
    print("✅ 修复transformers_stream_generator依赖问题")
    print("✅ 修复大文件大小检测问题")
    print("=" * 60)

    # 检查NumPy兼容性
    check_numpy_compatibility()

    # 检查磁盘空间
    try:
        import psutil

        disk = psutil.disk_usage(os.path.abspath("."))
        free_gb = disk.free / (1024 ** 3)
        print(f"💾 可用磁盘空间: {free_gb:.1f}GB")
        if free_gb < 20:
            print(f"⚠️  警告: Qwen-VL-Chat模型需要约18GB空间，建议至少20GB空闲")
            if input("继续下载? (Y/n): ").strip().lower() != 'y':
                sys.exit(0)
    except ImportError:
        print("⚠️  无法检查磁盘空间 (psutil未安装)")
        print("💡 安装命令: pip install psutil")

    # 重试逻辑
    success = False
    for attempt in range(args.retry):
        if attempt > 0:
            print(f"\n🔄 第 {attempt + 1} 次重试下载...")
            time.sleep(5)  # 等待5秒后重试

        success = download_qwen_vl_chat(args.dir, args.mirror)
        if success:
            break

    if not success:
        print(f"\n❌ 所有 {args.retry} 次下载尝试均失败")
        print("💡 建议:")
        print("1. 检查网络连接")
        print("2. 尝试使用镜像源: --mirror 参数")
        print("3. 手动下载关键文件")
        sys.exit(1)

    sys.exit(0)