#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen-VL-Chat 依赖修复工具
- 一键修复所有依赖问题
- 清理冲突包
- 重建环境
- 验证安装
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
import argparse
import platform


def clean_environment():
    """清理环境中的冲突包"""
    print("🧹 清理冲突依赖...")

    # 要清理的冲突包列表
    conflicting_packages = [
        "transformers_stream_generator", "tiktoken", "auto-gptq",
        "accelerate", "sentencepiece", "einops", "bitsandbytes",
        "transformers", "torch", "numpy"
    ]

    for package in conflicting_packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", package],
                           check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ 已清理: {package}")
        except:
            pass

    # 清理pip缓存
    try:
        subprocess.run([sys.executable, "-m", "pip", "cache", "purge"],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ pip缓存已清理")
    except:
        pass


def install_requirements():
    """安装依赖"""
    print("📦 安装Qwen-VL-Chat完整依赖...")

    try:
        # 安装核心依赖
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt",
            "--upgrade", "--no-cache-dir", "--force-reinstall"
        ], check=True)
        print("✅ 核心依赖安装成功")

        # 额外安装Qwen-VL特定依赖
        print("🔧 安装Qwen-VL特定依赖...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "qwen_vl_utils==0.0.1", "jsonlines==4.0.0",
            "packaging==23.2", "pydantic==1.10.14"
        ], check=True)
        print("✅ Qwen-VL特定依赖安装成功")

        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败: {e}")
        print("💡 尝试手动安装命令:")
        print("   pip install -r requirements.txt --upgrade --no-cache-dir --force-reinstall")
        return False


def verify_installation():
    """验证安装"""
    print("\n🔍 验证Qwen-VL-Chat依赖安装...")

    verification_results = {}

    # 关键依赖验证
    critical_packages = [
        "transformers", "torch", "transformers_stream_generator",
        "tiktoken", "auto_gptq", "qwen_vl_utils"
    ]

    for package in critical_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            verification_results[package] = {
                'status': 'success',
                'version': version,
                'message': f"✅ {package} {version} 已安装"
            }
            print(f"✅ {package} {version} 已安装")
        except ImportError as e:
            verification_results[package] = {
                'status': 'failed',
                'error': str(e),
                'message': f"❌ {package} 未安装: {str(e)}"
            }
            print(f"❌ {package} 未安装: {str(e)}")

    # 检查qwen.tiktoken文件
    model_dir = Path("./qwen_models")
    if model_dir.exists():
        tiktoken_file = model_dir / "qwen.tiktoken"
        if tiktoken_file.exists():
            verification_results['qwen.tiktoken'] = {
                'status': 'success',
                'message': "✅ qwen.tiktoken 文件存在"
            }
            print("✅ qwen.tiktoken 文件存在")
        else:
            verification_results['qwen.tiktoken'] = {
                'status': 'failed',
                'message': "❌ qwen.tiktoken 文件缺失，需要重新下载模型"
            }
            print("❌ qwen.tiktoken 文件缺失，需要重新下载模型")

    # 生成验证报告
    successful = sum(1 for r in verification_results.values() if r['status'] == 'success')
    total = len(verification_results)

    print(f"\n📊 验证报告: {successful}/{total} 项通过")

    if successful == total:
        print("🎉 所有依赖验证成功！")
        print("🚀 现在可以运行: python app.py")
        return True
    else:
        print("⚠️  部分依赖验证失败，需要修复")
        return False


def main():
    parser = argparse.ArgumentParser(description='Qwen-VL-Chat依赖修复工具')
    parser.add_argument('--full', action='store_true', help='完整修复（清理环境后重新安装）')
    parser.add_argument('--light', action='store_true', help='轻量修复（仅安装缺失依赖）')
    args = parser.parse_args()

    print("=" * 60)
    print("🔧 Qwen-VL-Chat 依赖修复工具")
    print(f"   Python版本: {platform.python_version()}")
    print(f"   操作系统: {platform.system()} {platform.release()}")
    print("=" * 60)

    # 全量修复
    if args.full:
        print("🔄 执行完整修复流程...")
        clean_environment()
        install_requirements()
        verify_installation()

    # 轻量修复
    elif args.light:
        print("🔄 执行轻量修复流程...")
        install_requirements()
        verify_installation()

    # 默认操作
    else:
        print("💡 使用建议:")
        print("   1. 首次安装或遇到严重问题: --full 参数")
        print("   2. 仅缺失部分依赖: --light 参数")
        print("\n🔧 推荐执行完整修复:")
        print("   python fix_dependencies.py --full")

    print("\n" + "=" * 60)
    print("✅ 修复流程完成")
    print("💡 后续步骤:")
    print("   1. 验证模型文件: python verify_install.py")
    print("   2. 下载模型: python download_models.py")
    print("   3. 启动应用: python app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()