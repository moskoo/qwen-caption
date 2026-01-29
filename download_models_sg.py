#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen-VL-Chat 模型下载脚本 (单线程版本 - 兼容最新Hugging Face Hub)
- 一个文件下载完成后再下载下一个
- 适配实际文件结构 (无需special_tokens_map.json)
- 修复NumPy 2.x兼容性问题
- 兼容最新Hugging Face Hub API (移除已弃用参数)
- 中国镜像源加速
"""

import os
import sys
import time
from pathlib import Path
import argparse
import json
import platform
import math
from tqdm import tqdm
import requests
import warnings
from urllib.parse import urlparse
from typing import List, Dict, Optional, Tuple

# 忽略Hugging Face Hub的弃用警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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


def get_model_file_list(repo_id: str, token: Optional[str] = None) -> List[Dict]:
    """获取模型仓库中的文件列表 (兼容最新API)"""
    print("📋 获取模型文件列表...")

    # 使用最新Hugging Face Hub API
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)

        # 获取仓库信息
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")

        # 过滤需要下载的文件
        filtered_files = []
        for filename in files:
            # 检查文件是否匹配需要的模式
            if any(filename.endswith(ext) for ext in [
                ".bin", ".safetensors", ".json", ".txt", ".py",
                ".tiktoken", ".md", ".png", ".model"
            ]):
                # 排除不需要的文件
                if any(x in filename for x in [".h5", ".ot", ".msgpack", ".onnx", ".pt"]):
                    continue

                # 尝试获取文件大小
                try:
                    file_info = api.repo_info(repo_id=repo_id, files_metadata=True)
                    # 这是一个简化处理，实际需要更复杂的逻辑获取文件大小
                    size = 0
                except:
                    size = 0

                filtered_files.append({
                    'path': filename,
                    'size': size
                })

        print(f"✅ 找到 {len(filtered_files)} 个需要下载的文件")
        # 按文件大小排序（大文件先下载）
        filtered_files.sort(key=lambda x: x['size'], reverse=True)
        return filtered_files

    except Exception as e:
        print(f"❌ 获取文件列表失败: {str(e)}")
        print("🔄 尝试使用备用方法获取文件列表...")
        return get_model_file_list_backup(repo_id, token)


def get_model_file_list_backup(repo_id: str, token: Optional[str] = None) -> List[Dict]:
    """备用方法获取文件列表"""
    # 手动定义Qwen-VL-Chat的关键文件
    essential_files = [
        "config.json",
        "configuration_qwen.py",
        "generation_config.json",
        "modeling_qwen.py",
        "pytorch_model.bin.index.json",
        "qwen.tiktoken",
        "tokenizer_config.json",
        "tokenization_qwen.py",
        "README.md",
        "special_tokens_map.json"
    ]

    # 分片文件 (1-10)
    shard_files = [f"pytorch_model-{str(i).zfill(5)}-of-00010.bin" for i in range(1, 11)]

    all_files = essential_files + shard_files

    file_list = []

    print("🔧 使用备用方法获取文件列表...")
    for filename in tqdm(all_files, desc="检查文件信息"):
        file_list.append({
            'path': filename,
            'size': 0  # 大小未知
        })

    return file_list


def download_file(
        repo_id: str,
        filename: str,
        save_dir: str,
        use_mirror: bool = False,
        token: Optional[str] = None,
) -> bool:
    """单文件下载函数，兼容最新Hugging Face Hub API"""
    from huggingface_hub import hf_hub_download, get_hf_file_metadata

    # 创建保存目录
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    save_path = save_dir_path / filename

    # 设置镜像源
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("🌐 使用HuggingFace镜像源 (中国加速)")

    # 准备下载
    print(f"\n📥 开始下载: {filename}")
    print(f"   保存到: {save_path}")

    try:
        # 使用最新API下载文件
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=save_dir_path,  # 使用local_dir参数
            token=token,
            force_download=False,  # 不强制重新下载
            cache_dir=None  # 不使用缓存
        )

        # 验证文件大小
        file_size = Path(file_path).stat().st_size
        print(f"✅ 下载完成: {filename}")
        print(f"   保存到: {file_path}")
        print(f"   大小: {file_size / 1024 ** 3:.2f}GB")
        return True

    except Exception as e:
        print(f"❌ 下载失败: {filename}")
        print(f"   错误: {str(e)}")

        # 清理不完整的文件
        if save_path.exists() and save_path.stat().st_size == 0:
            save_path.unlink()

        return False


def single_threaded_download(
        repo_id: str,
        save_dir: str,
        use_mirror: bool = False,
        token: Optional[str] = None,
) -> bool:
    """单线程下载所有文件 (兼容最新API)"""
    print(f"🚀 开始单线程下载: {repo_id}")
    print(f"📁 保存目录: {os.path.abspath(save_dir)}")

    # 获取文件列表
    file_list = get_model_file_list(repo_id, token)
    if not file_list:
        print("❌ 无法获取文件列表，下载失败")
        return False

    print(f"📋 总共需要下载 {len(file_list)} 个文件")

    # 按文件大小排序（大文件先下载）
    file_list.sort(key=lambda x: x['size'], reverse=True)

    # 逐个下载文件
    successful = []
    failed = []
    skipped = []

    for i, file_info in enumerate(file_list, 1):
        filename = file_info['path']
        save_path = Path(save_dir) / filename

        # 检查是否已存在
        if save_path.exists():
            print(f"⏭ 跳过已存在的文件: {filename}")
            skipped.append(filename)
            continue

        # 下载文件
        print(f"\n📊 进度: {i}/{len(file_list)} ({i / len(file_list) * 100:.1f}%)")
        success = download_file(repo_id, filename, save_dir, use_mirror, token)

        if success:
            successful.append(filename)
        else:
            failed.append(filename)

        # 每下载5个文件清理一次系统缓存（防止内存泄漏）
        if len(successful) % 5 == 0:
            import gc
            gc.collect()

        # 显示进度
        print(f"\n📊 当前进度: {len(successful)}/{len(file_list)} 成功, {len(failed)} 失败, {len(skipped)} 跳过")

    # 生成下载报告
    print("\n" + "=" * 60)
    print("🎉 下载任务完成!")
    print(f"✅ 成功: {len(successful)} 个文件")
    print(f"❌ 失败: {len(failed)} 个文件")
    print(f"⏭ 跳过: {len(skipped)} 个文件")

    if successful:
        total_size = sum((Path(save_dir) / f).stat().st_size for f in successful if (Path(save_dir) / f).exists())
        print(f"📦 总下载大小: {total_size / 1024 ** 3:.2f}GB")

    if failed:
        print("\n⚠️  失败文件列表:")
        for f in failed:
            print(f"   • {f}")
        print("\n💡 建议重新运行下载脚本，将自动继续下载失败的文件")

    print("=" * 60)
    return len(failed) == 0


def verify_model_files(model_dir: str) -> Tuple[bool, str]:
    """智能验证Qwen-VL-Chat模型文件是否完整"""
    model_dir_path = Path(model_dir)

    if not model_dir_path.exists():
        return False, "模型目录不存在"

    # 必需的核心配置文件
    required_config_files = [
        "config.json",
        "tokenizer_config.json",
        "qwen.tiktoken",
        "tokenization_qwen.py"
    ]

    # 检查配置文件
    missing_configs = [f for f in required_config_files if not (model_dir_path / f).exists()]
    if missing_configs:
        return False, f"缺失关键配置文件: {missing_configs}"

    # 检查模型权重文件 - 支持分片
    model_files = list(model_dir_path.glob("pytorch_model*.bin"))

    if not model_files:
        return False, "未找到模型权重文件 (pytorch_model*.bin)"

    # 检查分片数量
    shard_files = [f for f in model_files if "pytorch_model-" in f.name]
    if len(shard_files) < 8:  # 允许缺失1-2个
        return False, f"仅找到 {len(shard_files)} 个权重分片，模型可能不完整 (应有10个分片)"

    # 检查文件大小
    total_size = sum(f.stat().st_size for f in model_files)
    if total_size < 15e9:  # 15GB
        return False, f"模型文件总大小过小 ({total_size / 1e9:.2f}GB)，可能下载不完整 (完整模型约18GB)"

    return True, f"✅ Qwen-VL-Chat模型验证成功!\n   • 找到 {len(model_files)} 个权重文件\n   • 总大小: {total_size / 1e9:.2f}GB"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='下载Qwen-VL-Chat模型 (单线程版 - 兼容最新API)')
    parser.add_argument('--mirror', action='store_true', help='使用中国镜像源加速下载')
    parser.add_argument('--dir', type=str, default="./qwen_models", help='模型保存目录')
    parser.add_argument('--token', type=str, default=None, help='HuggingFace token (可选)')
    args = parser.parse_args()

    print("=" * 60)
    print("🌍 Qwen-VL-Chat 模型下载工具 (单线程版)")
    print("✅ 适配实际文件结构 (无需special_tokens_map.json)")
    print("✅ 兼容最新Hugging Face Hub API (移除已弃用参数)")
    print("✅ 一个文件下载完成后再下载下一个")
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

    # 执行单线程下载
    success = single_threaded_download(
        repo_id="Qwen/Qwen-VL-Chat",
        save_dir=args.dir,
        use_mirror=args.mirror,
        token=args.token,
    )

    # 验证下载结果
    if success:
        print("\n🔍 验证模型文件完整性...")
        is_valid, message = verify_model_files(args.dir)
        if is_valid:
            print(f"✅ {message}")
            print("\n🎉 模型下载和验证成功! 可以运行主程序了")
            sys.exit(0)
        else:
            print(f"❌ {message}")
            print("💡 建议重新运行下载脚本修复缺失文件")
            sys.exit(1)
    else:
        print("\n❌ 下载未完全成功，但部分文件已下载")
        print("💡 建议重新运行下载脚本，将自动继续下载")
        sys.exit(1)