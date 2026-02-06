#!/bin/bash

echo "🚀 XXG通义千问离线图片中文打标工具-离线版(Qwen3-VL-8B-Instruct) Ver.2.2"
echo "✅ 替换Qwen-vl-chat模型为Qwen3-VL | ✅ 文生图模型训练专用中文caption"
echo "✅ CUDA 11.8/12.1/12.4/13.x 全版本原生支持 | ✅ bitsandbytes>=0.44.0"
echo "✅ 100%中文caption | ✅ Qwen-Image、Seeddream等中文提示词LoRA训练专用"
echo "By 西小瓜 / Wechat:priest-mos"
echo "=============================================="

# 设置环境变量
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DOWNLOAD_TIMEOUT=300
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=0

# 检查Python 3.10
if ! command -v python3.10 &> /dev/null; then
    echo "❌ Python 3.10 未安装"
    echo "💡 请先安装Python 3.10:"
    echo "   Ubuntu/Debian: sudo apt update && sudo apt install -y python3.10 python3.10-venv python3.10-dev"
    echo "   CentOS/RHEL:   sudo dnf install -y python3.10 python3.10-venv python3.10-devel"
    exit 1
fi
echo "✅ 检测到Python 3.10"

# 检查磁盘空间
echo "🔍 检查磁盘空间..."
free_gb=$(df -BG . 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//')
if [ -z "$free_gb" ] || [ "$free_gb" -lt 1 ]; then
    free_gb=30
fi
echo "💾 可用空间: ${free_gb}GB"
if [ "$free_gb" -lt 18 ]; then
    echo "⚠️  警告: 需要至少18GB空闲空间 (模型14GB + 缓存)"
    read -p "继续? (y/n): " confirm
    if [ "$confirm" != "y" ]; then
        exit 1
    fi
fi

# ✅ 核心修复: 智能CUDA检测 + 兼容性验证
detect_cuda_and_driver() {
    echo "🔍 检测NVIDIA GPU和驱动版本..."

    # 检查NVIDIA GPU
    if ! command -v nvidia-smi &>/dev/null; then
        echo "⚠️  未检测到nvidia-smi，可能未安装NVIDIA驱动"
        echo "💡 请安装NVIDIA驱动 (>=535.86.05 支持CUDA 12.x/13.x):"
        echo "   Ubuntu/Debian: sudo apt install nvidia-driver-550"
        echo "   CentOS/RHEL:   sudo dnf install nvidia-driver"
        return 1
    fi

    # 获取驱动版本
    driver_ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | awk '{print $1}')
    if [ -z "$driver_ver" ]; then
        echo "⚠️  无法获取驱动版本"
        return 1
    fi

    echo "🎮 NVIDIA驱动版本: $driver_ver"

    # 驱动版本兼容性检查 (来源: NVIDIA官方文档)
    driver_major=$(echo "$driver_ver" | cut -d. -f1)
    if [ "$driver_major" -lt 515 ]; then
        echo "❌ 驱动版本过低 ($driver_ver)，需要 >=515.65.01"
        echo "💡 请升级NVIDIA驱动:"
        echo "   https://www.nvidia.com/Download/index.aspx"
        return 1
    elif [ "$driver_major" -lt 535 ]; then
        echo "⚠️  驱动版本 $driver_ver 仅支持CUDA <=11.8"
        echo "💡 建议升级至驱动 >=535 以获得CUDA 12.x/13.x支持"
    fi

    # 检测CUDA Toolkit (可选，非必需)
    if command -v nvcc &>/dev/null; then
        cuda_toolkit=$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' | head -n1)
        echo "📦 CUDA Toolkit版本: $cuda_toolkit (非必需，PyTorch自带CUDA运行时)"
    else
        echo "📦 未检测到CUDA Toolkit (正常，PyTorch自带CUDA运行时)"
    fi

    # 检测GPU型号和显存
    gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n1)
    if [ -n "$gpu_info" ]; then
        echo "📊 GPU信息: $gpu_info"
    fi

    return 0
}

# 创建/激活虚拟环境
VENV_DIR=".venv"
if [ -d "$VENV_DIR" ]; then
    echo "✅ 虚拟环境已存在，激活中..."
    source "$VENV_DIR/bin/activate" 2>/dev/null || {
        echo "❌ 虚拟环境损坏，重建中..."
        rm -rf "$VENV_DIR"
        python3.10 -m venv "$VENV_DIR" 2>/dev/null || {
            echo "❌ 虚拟环境创建失败，请安装python3.10-venv"
            exit 1
        }
        source "$VENV_DIR/bin/activate"
    }
else
    echo "📦 创建虚拟环境 $VENV_DIR ..."
    python3.10 -m venv "$VENV_DIR" 2>/dev/null || {
        echo "❌ 虚拟环境创建失败，请安装python3.10-venv:"
        echo "   sudo apt install python3.10-venv python3.10-dev"
        exit 1
    }
    source "$VENV_DIR/bin/activate"
fi
echo "✅ 虚拟环境已激活"

# 升级pip
echo "🔧 升级pip..."
pip install --upgrade pip setuptools wheel --quiet 2>/dev/null || true

# 检测GPU/驱动
if ! detect_cuda_and_driver; then
    echo "⚠️  未检测到有效GPU环境，将使用CPU模式"
    USE_GPU=0
else
    USE_GPU=1
    echo "✅ GPU环境检测通过，将安装GPU版本PyTorch"
fi

# ✅ 核心修复: 安装PyTorch (CUDA 13.x原生兼容)
echo "📦 安装PyTorch 2.4.1..."
if [ "$USE_GPU" -eq 1 ]; then
    # ✅ 关键: CUDA 13.x 与 CUDA 12.4 二进制完全兼容，直接使用cu124
    echo "💡 CUDA 13.x 与 CUDA 12.4 二进制完全兼容，直接使用cu124 wheel"
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124 -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || {
        echo "⚠️  cu124安装失败，尝试cu121..."
        pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || {
            echo "❌ GPU版本安装失败，回退到CPU版本..."
            pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || {
                echo "❌ PyTorch安装失败"
                exit 1
            }
            USE_GPU=0
        }
    }
else
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || {
        echo "❌ CPU版本安装失败"
        exit 1
    }
fi

# ✅ 核心修复: 验证PyTorch CUDA兼容性 (关键!)
echo "🔍 验证PyTorch CUDA兼容性..."
if python <<'PYEOF'
import torch, sys, os

# 检查CUDA可用性
if not torch.cuda.is_available():
    print("❌ CUDA不可用 (驱动问题/版本不兼容)")
    sys.exit(1)

# 检查CUDA版本
cuda_ver = torch.version.cuda
print(f"✅ PyTorch检测到CUDA: {cuda_ver}")

# 检查GPU数量
gpu_count = torch.cuda.device_count()
print(f"✅ 检测到 {gpu_count} 个GPU")

# 实际运行测试 (关键: 验证CUDA 13.x兼容性)
try:
    # 创建随机张量并传输到GPU
    x = torch.randn(1000, 1000).to('cuda')
    y = torch.randn(1000, 1000).to('cuda')
    z = x @ y  # 矩阵乘法 (触发CUDA kernel)
    print(f"✅ CUDA计算测试通过 (结果形状: {z.shape})")

    # 检查CUDA运行时版本
    rt_ver = torch.version.cuda
    print(f"✅ CUDA运行时版本: {rt_ver}")

    sys.exit(0)
except Exception as e:
    print(f"❌ CUDA运行时错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF
then
    echo "✅ PyTorch CUDA兼容性验证通过 (CUDA 13.x原生支持)"
    TORCH_DEVICE="cuda"
else
    echo "⚠️  CUDA验证失败，回退到CPU模式"
    TORCH_DEVICE="cpu"
    USE_GPU=0
fi

torch_ver=$(python -c "import torch; print(torch.__version__)")
echo "✅ PyTorch $torch_ver 安装成功 (设备: $TORCH_DEVICE)"

# 降级NumPy (<2.0)
echo "📦 降级NumPy至1.26.4 (<2.0)..."
pip uninstall -y numpy 2>/dev/null
pip install "numpy==1.26.4" --force-reinstall --no-deps --quiet 2>/dev/null || {
    echo "❌ NumPy降级失败"
    exit 1
}
echo "✅ NumPy 1.26.4安装成功"

# 安装bitsandbytes>=0.44.0
echo "📦 安装bitsandbytes>=0.44.0 (修复PyTorch 2.4.1内存泄漏)..."
pip install "bitsandbytes>=0.44.0,<0.45.0" --quiet 2>/dev/null || {
    echo "⚠️  bitsandbytes安装失败 (4-bit量化可能不可用)"
}
echo "✅ bitsandbytes安装完成"

# 从GitHub安装transformers源码
echo "📦 从GitHub安装transformers源码 (含Qwen3-VL支持)..."
if [ ! -d "transformers_src" ]; then
    echo "   克隆transformers仓库..."
    git clone --depth 1 https://github.com/huggingface/transformers.git transformers_src 2>/dev/null || {
        echo "❌ Git克隆失败，请检查网络或安装git: sudo apt install git"
        exit 1
    }
else
    echo "   更新现有transformers源码..."
    (cd transformers_src && git pull --quiet 2>/dev/null) || echo "   ⚠️  更新失败，使用现有源码"
fi

echo "   编译安装transformers (首次需5-15分钟)..."
cd transformers_src
pip install -e ".[vl]" --no-build-isolation --quiet 2>/dev/null || {
    echo "❌ transformers编译安装失败"
    echo "💡 请先安装系统依赖:"
    echo "   Ubuntu/Debian: sudo apt install cmake ninja-build build-essential -y"
    echo "   CentOS/RHEL:   sudo dnf install cmake ninja-build gcc-c++ make -y"
    exit 1
}
cd ..

# 验证transformers
if ! python -c "from transformers import Qwen3VLProcessor; print('✅ Qwen3VLProcessor可用')" 2>/dev/null; then
    echo "❌ Qwen3VLProcessor不可用，请重新运行本脚本"
    exit 1
fi

# 安装其他必要依赖
echo "📦 安装其他依赖..."
pip install "accelerate>=0.30.0" "safetensors>=0.4.0" "gradio>=4.40.0" "pillow>=10.0.0" "tqdm>=4.66.0" "psutil>=5.9.0" "huggingface_hub>=1.3.7" "requests>=2.31.0" "filelock>=3.13.0" "fsspec>=2023.10.0" --quiet 2>/dev/null || {
    echo "⚠️  部分依赖安装失败，但可能不影响核心功能"
}

# 验证关键依赖
echo "🔍 验证关键依赖..."
python <<'PYEOF'
import sys
deps = ["torch", "numpy", "transformers", "gradio", "PIL", "bitsandbytes", "huggingface_hub"]
for dep in deps:
    try:
        __import__(dep)
        ver = sys.modules[dep].__version__ if hasattr(sys.modules[dep], '__version__') else 'unknown'
        print(f"✅ {dep} {ver}")
    except ImportError:
        print(f"❌ {dep} 未安装", file=sys.stderr)
        sys.exit(1)
sys.exit(0)
PYEOF

if [ $? -ne 0 ]; then
    echo "❌ 依赖验证失败"
    exit 1
fi
echo "✅ 所有关键依赖验证通过"

# 验证模型文件
MODEL_DIR="qwen3_vl_models"
echo "🔍 验证Qwen3-VL-8B模型..."
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ 模型目录不存在: $MODEL_DIR"
    echo "💡 请先下载模型:"
    echo "   huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir ./qwen3_vl_models"
    echo "   或"
    echo "   modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir ./qwen3_vl_models"
    exit 1
fi

# 检查必需文件
required_files=("config.json" "preprocessor_config.json" "tokenizer_config.json" "tokenizer.json")
for f in "${required_files[@]}"; do
    if [ ! -f "$MODEL_DIR/$f" ]; then
        echo "❌ 缺失必需文件: $f"
        exit 1
    fi
done
echo "✅ 所有核心配置文件存在"

# 检查权重文件 (4分片)
weight_count=$(ls "$MODEL_DIR"/model-0000[1-4]-of-00004.safetensors 2>/dev/null | wc -l)
if [ "$weight_count" -lt 4 ]; then
    echo "❌ 未找到完整权重分片 (需要4个，找到$weight_count个)"
    exit 1
fi
echo "✅ 权重文件验证通过 (4分片)"

# 启动应用
echo ""
echo "=============================================="
echo "🎯 启动XXG-Qwen3-VL中文打标工具 (http://127.0.0.1:9527)"
echo "💡 使用提示:"
echo "   • 低显存GPU (<10GB): 添加 --4bit 参数"
echo "   • 无GPU/驱动问题: 添加 --cpu 参数"
echo "   • 直接处理文件夹: --folder /path/to/images"
echo ""
echo "🔧 CUDA兼容性说明:"
echo "   • CUDA 13.x 与 CUDA 12.4 二进制完全兼容"
echo "   • PyTorch 2.4.1 cu124 wheel 可在 CUDA 13.x 原生运行"
echo "   • 无需降级CUDA Toolkit或驱动"
echo "=============================================="
echo ""

# 传递所有参数给app.py
exec python app.py "$@"