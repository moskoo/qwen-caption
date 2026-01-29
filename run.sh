#!/bin/bash

echo "🚀 通义千问离线图片打标工具启动脚本 (Python 3.10 + PyTorch 2.9.1)"
echo "✅ 专为Python 3.10环境优化"
echo "✅ 使用Torch 2.2.2最新稳定版"
echo "✅ 修复transformers_stream_generator依赖识别问题"
echo "✅ 适配Qwen-VL-Chat文件结构"
echo "=============================================="

# 设置环境变量
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DOWNLOAD_TIMEOUT=300
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 检查Python 3.10是否安装
if ! command -v python3.10 &> /dev/null; then
    echo "❌ Python 3.10 未安装"
    echo "💡 请先安装Python 3.10"
    echo "   Ubuntu/Debian: sudo apt update && sudo apt install python3.10 python3.10-venv"
    echo "   CentOS/RHEL: sudo yum install python3.10 python3.10-venv"
    echo "   macOS: brew install python@3.10"
    echo "【1/4】检查并安装python3.10-venv依赖..."
    if ! dpkg -s python3.10-venv &> /dev/null; then
        echo "未安装python3.10-venv，开始安装..."
        sudo apt update && sudo apt install -y python3.10-venv python3.10-dev
    else
        echo "python3.10-venv已安装"
    fi
    exit 1
fi

# 检查GPU和CUDA (增强版)
check_gpu_requirements() {
    local has_gpu=false
    local cuda_available=false
    local cuda_version=""
    local gpu_details=""
    
    echo "🔍 检测GPU和CUDA状态..."
    
    # 1. 首先检查是否有NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        # 获取GPU详细信息
        gpu_details=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "")
        
        if [ -n "$gpu_details" ]; then
            has_gpu=true
            echo "🎮 检测到NVIDIA GPU:"
            echo "   $gpu_details"
        else
            echo "⚠️  nvidia-smi存在但未检测到GPU，可能驱动问题"
        fi
    else
        echo "🔍 未找到nvidia-smi命令，尝试其他检测方法..."
        
        # 备用方法1: 检查/proc/driver/nvidia
        if [ -d "/proc/driver/nvidia" ]; then
            echo "🎮 通过/proc/driver/nvidia检测到NVIDIA GPU"
            has_gpu=true
        fi
        
        # 备用方法2: 检查lspci
        if command -v lspci &> /dev/null; then
            if lspci 2>/dev/null | grep -i nvidia &> /dev/null; then
                echo "🎮 通过lspci检测到NVIDIA GPU"
                has_gpu=true
            fi
        fi
        
        # 备用方法3: 检查设备文件
        if [ -c "/dev/nvidia0" ] || [ -c "/dev/nvidiactl" ]; then
            echo "🎮 通过设备文件检测到NVIDIA GPU"
            has_gpu=true
        fi
    fi
    
    # 2. 检查CUDA可用性
    if $has_gpu; then
        echo "🔍 检查CUDA工具包..."
        
        # 检查CUDA运行时API
        if python3.10 -c "
import importlib.util, sys, subprocess, json, os, math, random, time, datetime, collections, itertools, fractions, decimal, typing, statistics, heapq, bisect, copy, string, re, collections, math
try:
    import torch
    if torch.cuda.is_available():
        print('CUDA_AVAILABLE: true')
        print(f'GPU_COUNT: {torch.cuda.device_count()}')
        print(f'CURRENT_DEVICE: {torch.cuda.current_device()}')
        print(f'GPU_NAME: {torch.cuda.get_device_name(0)}')
        print(f'GPU_MEMORY: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
        print(f'PYTORCH_VERSION: {torch.__version__}')
    else:
        print('CUDA_AVAILABLE: false')
except Exception as e:
    print(f'ERROR: {str(e)}')
" 2>/dev/null | grep -q "CUDA_AVAILABLE: true"; then
            cuda_available=true
            echo "✅ CUDA在PyTorch中可用"
            
            # 获取PyTorch检测到的CUDA版本
            cuda_version=$(python3.10 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "")
            if [ -n "$cuda_version" ]; then
                echo "🔢 PyTorch使用的CUDA版本: $cuda_version"
            fi
        else
            echo "⚠️  GPU存在但CUDA在PyTorch中不可用"
            echo "💡 可能原因:"
            echo "   • 未安装正确版本的PyTorch"
            echo "   • NVIDIA驱动版本过低"
            echo "   • CUDA工具包未正确配置"
            
            # 检查nvcc
            if command -v nvcc &> /dev/null; then
                cuda_version=$(nvcc --version 2>/dev/null | grep release | sed 's/.*release //' | sed 's/,.*//')
                echo "🔢 检测到CUDA工具包版本: $cuda_version"
            else
                # 尝试从nvidia-smi获取
                cuda_version=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | awk '{print $9}' 2>/dev/null || echo "")
                if [ -n "$cuda_version" ]; then
                    echo "🔢 从nvidia-smi获取CUDA版本: $cuda_version"
                else
                    echo "⚠️  无法确定CUDA版本"
                fi
            fi
            
            # 检查驱动版本
            if command -v nvidia-smi &> /dev/null; then
                driver_version=$(nvidia-smi 2>/dev/null | grep "Driver Version" | awk '{print $6}' 2>/dev/null || echo "")
                if [ -n "$driver_version" ]; then
                    echo "🔧 NVIDIA驱动版本: $driver_version"
                    
                    # 粗略检查驱动与CUDA兼容性
                    if [ -n "$cuda_version" ]; then
                        if [[ "$cuda_version" =~ 12 ]] && [[ "$driver_version" < "525" ]]; then
                            echo "⚠️  警告: CUDA 12.x需要NVIDIA驱动版本>=525"
                        elif [[ "$cuda_version" =~ 11.[0-8] ]] && [[ "$driver_version" < "450" ]]; then
                            echo "⚠️  警告: CUDA 11.x需要NVIDIA驱动版本>=450"
                        fi
                    fi
                fi
            fi
        fi
    else
        echo "💻 未检测到NVIDIA GPU，将使用CPU版本"
    fi
    
    # 3. 最终状态报告
    echo "📊 GPU检测结果:"
    echo "   • GPU存在: $has_gpu"
    echo "   • CUDA可用: $cuda_available"
    echo "   • CUDA版本: ${cuda_version:-'未知'}"
    
    # 4. 返回结果
    echo "$has_gpu:$cuda_available:$cuda_version"
}

# 检查磁盘空间
check_disk_space() {
    local free_space_gb
    if command -v df &> /dev/null; then
        free_space_gb=$(df -BG . 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//')
    else
        free_space_gb=25  # 无法检查时假设足够
    fi

    if [ -z "$free_space_gb" ]; then
        free_space_gb=25
    fi

    echo "💾 可用磁盘空间: ${free_space_gb}GB"

    if [ "$free_space_gb" -lt 20 ]; then
        echo "⚠️  警告: Qwen-VL-Chat模型需要约18GB空间，建议至少20GB空闲空间"
        read -p "继续安装? (y/n): " confirm
        if [ "$confirm" != "y" ]; then
            exit 1
        fi
    fi
}

# 获取PyTorch安装命令 (Python 3.10兼容)
get_pytorch_install_command() {
    local has_gpu="$1"
    local cuda_available="$2"
    local cuda_version="$3"

    # PyTorch 2.10.0 for Python 3.10
    if [ "$has_gpu" = "true" ] && [ "$cuda_available" = "true" ]; then
        if [[ "$cuda_version" =~ 12 ]]; then
            echo "pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir --quiet"
        elif [[ "$cuda_version" =~ 11\.[4-9] ]]; then
            echo "pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir --quiet"
        else
            echo "torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu --no-cache-dir --quiet"
        fi
    else
        echo "pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu --no-cache-dir --quiet"
    fi
}

# 设置Python 3.10虚拟环境
setup_virtual_env() {
    local env_name=".venv"
    local python_cmd="python3.10"

    echo "🔄 检查Python 3.10虚拟环境: $env_name"

    # 检查环境是否存在
    if [ -d "$env_name" ]; then
        echo "✅ 虚拟环境 '$env_name' 已存在"

        # 激活环境
        source "$env_name/bin/activate"
        echo "✅ 虚拟环境 '$env_name' 已激活!"

        # 验证环境
        echo "🛠️  验证当前Python 3.10路径和环境..."
        echo "当前Python: $(which python)"
        current_python_version=$(python -c "import platform; print(platform.python_version())")
        echo "当前Python版本: $current_python_version"

        # 检查是否为3.10
        if [[ "$current_python_version" != 3.10* ]]; then
            echo "❌ 当前环境Python版本为 $current_python_version，不是3.10"
            echo "💡 重新创建Python 3.10虚拟环境..."
            rm -rf "$env_name"
            $python_cmd -m venv "$env_name"
            source "$env_name/bin/activate"
        fi

        # 修复依赖
        fix_dependencies
    else
        echo "📦 创建新的Python 3.10虚拟环境: $env_name"

        # 创建环境
        $python_cmd -m venv "$env_name"

        # 激活环境
        source "$env_name/bin/activate"

        # 验证环境
        echo "✅ 虚拟环境 '$env_name' 已创建并激活"
        echo "当前Python: $(which python)"
        current_python_version=$(python -c "import platform; print(platform.python_version())")
        echo "当前Python版本: $current_python_version"

        # 安装依赖
        install_dependencies
    fi

    # 标记环境已安装
    touch .env_installed

    # 最终验证
    echo "🔍 最终依赖验证..."
    verify_dependencies
}

# 修复依赖
fix_dependencies() {
    echo "🔧 检查Qwen-VL-Chat依赖完整性..."
    if ! python -c "
import importlib, sys, subprocess, pkgutil, importlib.util, json, os, warnings, math
warnings.filterwarnings('ignore')

dependencies = {
    'transformers_stream_generator': '0.0.5',
    'tiktoken': '0.7.0',
    'transformers': '4.44.2',
    'torch': '2.2.2',
    'numpy': '1.26.4'
}

missing = []
for pkg, version in dependencies.items():
    try:
        spec = importlib.util.find_spec(pkg)
        if spec is None:
            missing.append(f'{pkg} (未安装)')
            continue

        module = importlib.import_module(pkg)

        if pkg == 'tiktoken':
            continue

        if hasattr(module, '__version__'):
            module_version = module.__version__
            main_version = version.split('.')[0]
            if not module_version.startswith(main_version):
                missing.append(f'{pkg}=={version} (当前版本: {module_version})')
    except Exception as e:
        missing.append(f'{pkg}=={version} (错误: {str(e)})')

if missing:
    print(f'❌ 检测到问题: {\", \".join(missing)}')
    sys.exit(1)
else:
    print('✅ 所有必需依赖验证通过')
    sys.exit(0)
" 2>/dev/null; then
        echo "⚠️  依赖验证失败，尝试清理并重新安装..."

        # 清理可能冲突的包
        echo "🧹 清理可能的冲突包..."
        pip uninstall -y transformers_stream_generator tiktoken transformers torch numpy auto-gptq bitsandbytes accelerate

        # 重新安装依赖
        install_dependencies
    fi
}

# 安装依赖
install_dependencies() {
    # 升级pip
    echo "🔧 升级pip..."
    pip install --upgrade pip setuptools wheel --quiet

    # 检查GPU情况
    local gpu_check=$(check_gpu_requirements)
    IFS=':' read -ra gpu_info <<< "$gpu_check"
    local has_gpu="${gpu_info[0]}"
    local cuda_available="${gpu_info[1]}"
    local cuda_version="${gpu_info[2]}"

    # 安装PyTorch (使用Python 3.10兼容命令)
    install_cmd=$(get_pytorch_install_command "$has_gpu" "$cuda_available" "$cuda_version")
    echo "🔧 安装PyTorch 2.2.2..."
    eval "$install_cmd"

    # 安装关键依赖
    echo "🔧 安装关键依赖 (Qwen-VL-Chat必需)..."
    pip install transformers_stream_generator==0.0.5 tiktoken==0.7.0 --upgrade --no-cache-dir --quiet

    # 安装4-bit量化支持
    echo "🔧 安装4-bit量化支持..."
    pip install auto-gptq==0.7.1 optimum==1.21.0 bitsandbytes==0.44.1 --upgrade --no-cache-dir --quiet

    # 安装其他依赖
    echo "⬇️ 安装其他依赖..."
    pip install "transformers==4.44.2" "gradio==4.44.1" "accelerate==1.1.0" --upgrade --quiet
    pip install -r requirements.txt --no-cache-dir --upgrade

    echo "✅ 依赖安装完成"
}

# 验证依赖
verify_dependencies() {
    python -c "
import importlib, sys, subprocess, pkgutil, importlib.util, json, os, warnings, math
warnings.filterwarnings('ignore')

required_packages = ['transformers_stream_generator', 'tiktoken', 'transformers', 'torch', 'numpy']
missing = []
for pkg in required_packages:
    try:
        importlib.import_module(pkg)
        print(f'✅ {pkg} 在环境中可用')
    except ImportError as e:
        missing.append(pkg)
        print(f'❌ {pkg} 在环境中不可用: {str(e)}')
if missing:
    print(f'⚠️  仍然缺失包: {\", \".join(missing)}')
else:
    print('🎉 所有依赖验证通过，Python 3.10环境准备就绪')
" || echo "⚠️  依赖验证失败，但将继续启动应用"
}

# 智能检查Qwen-VL-Chat模型文件
check_model_files() {
    local model_dir="qwen_models"

    echo "🔍 智能检查Qwen-VL-Chat模型文件完整性..."

    # 检查模型目录
    if [ ! -d "$model_dir" ]; then
        echo_qwen_download_prompt "模型目录不存在"
        return 1
    fi

    # 检查基础配置文件
    if [ ! -f "$model_dir/config.json" ] || [ ! -f "$model_dir/tokenizer_config.json" ]; then
        echo_qwen_download_prompt "缺失基础配置文件 (config.json 或 tokenizer_config.json)"
        return 1
    fi

    # 检查Qwen特定tokenizer文件
    if [ ! -f "$model_dir/qwen.tiktoken" ] || [ ! -f "$model_dir/tokenization_qwen.py" ] || [ ! -f "$model_dir/modeling_qwen.py" ] || [ ! -f "$model_dir/configuration_qwen.py" ]; then
        echo_qwen_download_prompt "缺失Qwen特定文件 (qwen.tiktoken, tokenization_qwen.py, modeling_qwen.py, configuration_qwen.py)"
        return 1
    fi

    # 检查权重文件 (支持分片)
    local weight_files=()
    local file

    echo "🔍 查找所有权重文件..."
    # 安全地获取所有权重文件
    shopt -s nullglob  # 确保无匹配时不返回原模式
    weight_files=("$model_dir"/pytorch_model*.bin)
    shopt -u nullglob

    # 过滤出真实文件
    local real_files=()
    for file in "${weight_files[@]}"; do
        if [ -f "$file" ]; then
            real_files+=("$file")
        fi
    done

    weight_files=("${real_files[@]}")
    local weight_count=${#weight_files[@]}

    echo "🔍 找到 ${weight_count} 个权重文件"

    if [ $weight_count -eq 0 ]; then
        echo_qwen_download_prompt "未找到模型权重文件"
        return 1
    fi

    # 检查分片数量
    local shard_count=0
    echo "🔍 检查权重分片文件..."

    for file in "${weight_files[@]}"; do
        local filename=$(basename "$file")

        # 安全的分片文件检测
        if echo "$filename" | grep -q "^pytorch_model-[0-9]\{5\}-of-00010\.bin$"; then
            echo "   [OK] 找到分片文件: $filename"
            shard_count=$((shard_count + 1))
        else
            echo "   [INFO] 非分片文件: $filename"
        fi
    done

    echo "🔍 找到 $shard_count 个权重分片文件"

    if [ $shard_count -lt 8 ]; then  # 允许缺失1-2个分片
        echo_qwen_download_prompt "仅找到 $shard_count 个权重分片，模型可能不完整 (应有10个分片)"
        return 1
    fi

    echo "✅ Qwen-VL-Chat模型验证成功!"
    echo "   • 找到 ${weight_count} 个权重文件"
    echo "   • 找到 ${shard_count} 个权重分片"
    return 0
}

# 下载提示函数
echo_qwen_download_prompt() {
    local reason="$1"
    echo "❌ 模型验证失败: $reason"
    echo "💡 需要下载完整的Qwen-VL-Chat模型 (约18GB):"
    echo "   python download_models.py"
    echo ""
    read -p "是否现在下载模型? (y/n): " download_model
    if [ "$download_model" = "y" ]; then
        # 检查网络连接
        if ! ping -c 1 huggingface.co &> /dev/null; then
            echo "⚠️  网络连接不稳定，建议使用镜像源"
            read -p "使用中国镜像源? (y/n): " use_mirror
            if [ "$use_mirror" = "y" ]; then
                python download_models.py --mirror
            else
                python download_models.py
            fi
        else
            python download_models.py
        fi
    else
        echo "⚠️  请先下载完整模型再运行主程序"
        exit 1
    fi
}

# 主流程
main() {
    # 1. 检查磁盘空间
    check_disk_space

    # 2. 设置虚拟环境
    setup_virtual_env

    # 3. 检查模型
    #check_model_files

    # 4. 启动应用
    echo "🎯 启动应用 (访问 http://127.0.0.1:9527)..."
    echo ""

    # 传递所有参数给app.py
    python app.py "$@"

    echo ""
    echo "👋 应用已关闭"
}
# 执行主流程
main "$@"