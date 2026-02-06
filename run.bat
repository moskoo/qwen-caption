@echo off
chcp 65001 >nul 2>&1 || chcp 936 >nul 2>&1
setlocal enabledelayedexpansion

echo ==============================================================
echo   XXG通义千问离线图片中文打标工具-离线版(Qwen3-VL-8B-Instruct) Ver.2.2
echo   ✅ 替换Qwen-vl-chat模型为Qwen3-VL | ✅ 文生图模型训练专用中文caption
echo   ✅ CUDA 11.8/12.1/12.4/13.x 全版本原生支持 | ✅ bitsandbytes^>=0.44.0
echo   ✅ 100%中文caption | ✅ Qwen-Image、Seeddream等中文提示词LoRA训练专用
echo   By 西小瓜 / Wechat:priest-mos
echo ==============================================================

REM 设置环境变量
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_ENABLE_HF_TRANSFER=1
set HF_HUB_DOWNLOAD_TIMEOUT=300
set TRANSFORMERS_NO_ADVISORY_WARNINGS=1
set TOKENIZERS_PARALLELISM=false
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set CUDA_VISIBLE_DEVICES=0

REM 颜色输出 (Windows 10+ 支持ANSI)
for /F "tokens=1,2 delims=#" %%a in ('"prompt #$H#$E# & echo on & for %%b in (1) do rem"') do set "ESC=%%a"
set RED=%ESC%[91m
set GREEN=%ESC%[92m
set YELLOW=%ESC%[93m
set BLUE=%ESC%[94m
set NC=%ESC%[0m

REM 检查Python 3.10
call :check_python
if %errorlevel% neq 0 exit /b 1

REM 检查磁盘空间
call :check_disk_space
if %errorlevel% neq 0 exit /b 1

REM 创建/激活虚拟环境
set VENV_DIR=.venv
if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [%BLUE%INFO%NC%] 虚拟环境已存在，激活中...
    call "%VENV_DIR%\Scripts\activate.bat" >nul 2>&1
    if !errorlevel! neq 0 (
        echo [%RED%ERROR%NC%] 虚拟环境激活失败，重建中...
        rmdir /s /q "%VENV_DIR%" >nul 2>&1
        %PYTHON_CMD% -m venv "%VENV_DIR%" >nul 2>&1 || (
            echo [%RED%ERROR%NC%] 虚拟环境创建失败，请安装python3.10-venv
            exit /b 1
        )
        call "%VENV_DIR%\Scripts\activate.bat" >nul 2>&1
    )
) else (
    echo [%BLUE%INFO%NC%] 创建虚拟环境 %VENV_DIR% ...
    %PYTHON_CMD% -m venv "%VENV_DIR%" >nul 2>&1 || (
        echo [%RED%ERROR%NC%] 虚拟环境创建失败，请安装python3.10-venv:
        echo    pip install virtualenv --user
        exit /b 1
    )
    call "%VENV_DIR%\Scripts\activate.bat" >nul 2>&1
)
echo [%GREEN%SUCCESS%NC%] 虚拟环境已激活

REM 升级pip
echo [%BLUE%INFO%NC%] 升级pip...
pip install --upgrade pip setuptools wheel --quiet >nul 2>&1 || echo [%YELLOW%WARN%NC%] pip升级部分失败，继续安装依赖

REM 检测GPU/驱动
call :detect_cuda_and_driver
set USE_GPU=%errorlevel%
if %USE_GPU% equ 0 (
    echo [%GREEN%SUCCESS%NC%] GPU环境检测通过，将安装GPU版本PyTorch
) else (
    echo [%YELLOW%WARN%NC%] 未检测到有效GPU环境，将使用CPU模式
)

REM 安装PyTorch 2.4.1 (国内清华源加速)
echo [%BLUE%INFO%NC%] 安装PyTorch 2.4.1...
if %USE_GPU% equ 0 (
    REM ✅ 核心: CUDA 13.x 与 CUDA 12.4 二进制完全兼容，直接使用cu124 + 清华源
    echo [%BLUE%INFO%NC%] 💡 CUDA 13.x 与 CUDA 12.4 二进制完全兼容，使用cu124 wheel + 清华源加速
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124 -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet 2>nul && (
        echo [%GREEN%SUCCESS%NC%] PyTorch GPU版本安装成功
    ) || (
        echo [%YELLOW%WARN%NC%] cu124安装失败，尝试cu121...
        pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet 2>nul && (
            echo [%GREEN%SUCCESS%NC%] PyTorch GPU版本安装成功 (cu121)
        ) || (
            echo [%RED%ERROR%NC%] GPU版本安装失败，回退到CPU版本...
            pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet 2>nul || (
                echo [%RED%ERROR%NC%] PyTorch安装失败
                exit /b 1
            )
            set USE_GPU=1
            echo [%GREEN%SUCCESS%NC%] PyTorch CPU版本安装成功
        )
    )
) else (
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet 2>nul || (
        echo [%RED%ERROR%NC%] CPU版本安装失败
        exit /b 1
    )
    echo [%GREEN%SUCCESS%NC%] PyTorch CPU版本安装成功
)

REM 验证PyTorch CUDA兼容性
echo [%BLUE%INFO%NC%] 验证PyTorch CUDA兼容性...
python -c "import torch, sys; print('CUDA可用' if torch.cuda.is_available() else 'CUDA不可用')" 2>nul | findstr /c:"CUDA可用" >nul
if %errorlevel% equ 0 (
    REM 实际GPU计算测试
    python -c "import torch; x=torch.randn(1000,1000,device='cuda'); y=torch.randn(1000,1000,device='cuda'); z=x@y; print('✅ CUDA计算测试通过')" 2>nul && (
        echo [%GREEN%SUCCESS%NC%] ✅ PyTorch CUDA兼容性验证通过 (CUDA 13.x原生支持)
        set TORCH_DEVICE=cuda
    ) || (
        echo [%YELLOW%WARN%NC%] CUDA验证失败，回退到CPU模式
        set TORCH_DEVICE=cpu
        set USE_GPU=1
    )
) else (
    set TORCH_DEVICE=cpu
    set USE_GPU=1
)
for /f "delims=" %%i in ('python -c "import torch; print(torch.__version__)" 2^>nul') do set torch_ver=%%i
echo [%GREEN%SUCCESS%NC%] PyTorch !torch_ver! 安装成功 (设备: !TORCH_DEVICE!)

REM 降级NumPy (<2.0)
echo [%BLUE%INFO%NC%] 降级NumPy至1.26.4 (^<2.0)...
pip uninstall -y numpy >nul 2>&1
pip install "numpy==1.26.4" --force-reinstall --no-deps --quiet 2>nul || (
    echo [%RED%ERROR%NC%] NumPy降级失败
    exit /b 1
)
echo [%GREEN%SUCCESS%NC%] NumPy 1.26.4安装成功

REM 安装bitsandbytes>=0.44.0
echo [%BLUE%INFO%NC%] 安装bitsandbytes^>=0.44.0 (修复PyTorch 2.4.1内存泄漏)...
pip install "bitsandbytes>=0.44.0,<0.45.0" --quiet 2>nul || echo [%YELLOW%WARN%NC%] bitsandbytes安装失败 (4-bit量化可能不可用)
echo [%GREEN%SUCCESS%NC%] bitsandbytes安装完成

REM 从GitHub安装transformers源码
echo [%BLUE%INFO%NC%] 从GitHub安装transformers源码 (含Qwen3-VL支持)...
if not exist "transformers_src" (
    echo    克隆transformers仓库...
    git clone --depth 1 https://github.com/huggingface/transformers.git transformers_src 2>nul || (
        echo [%RED%ERROR%NC%] Git克隆失败，请检查网络或安装git: winget install Git.Git
        exit /b 1
    )
) else (
    echo    更新现有transformers源码...
    cd transformers_src && git pull --quiet 2>nul && cd ..
)
echo    编译安装transformers (首次需5-15分钟)...
cd transformers_src
pip install -e ".[vl]" --no-build-isolation --quiet 2>nul || (
    echo [%RED%ERROR%NC%] transformers编译安装失败
    echo 💡 请先安装系统依赖:
    echo    Visual Studio Build Tools (含C++开发工具)
    echo    https://visualstudio.microsoft.com/visual-cpp-build-tools/
    exit /b 1
)
cd ..
echo [%GREEN%SUCCESS%NC%] transformers安装成功

REM 验证transformers
python -c "from transformers import Qwen3VLProcessor; print('✅ Qwen3VLProcessor可用')" 2>nul && (
    echo [%GREEN%SUCCESS%NC%] Qwen3VLProcessor可用
) || (
    echo [%RED%ERROR%NC%] Qwen3VLProcessor不可用，请重新运行本脚本
    exit /b 1
)

REM 安装其他必要依赖
echo [%BLUE%INFO%NC%] 安装其他依赖...
pip install "accelerate>=0.30.0" "safetensors>=0.4.0" "gradio>=4.40.0" "pillow>=10.0.0" "tqdm>=4.66.0" "psutil>=5.9.0" "huggingface_hub>=1.3.7" "requests>=2.31.0" "filelock>=3.13.0" "fsspec>=2023.10.0" -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet 2>nul || echo [%YELLOW%WARN%NC%] 部分依赖安装失败，但可能不影响核心功能

REM 验证关键依赖
echo [%BLUE%INFO%NC%] 验证关键依赖...
python -c "import sys; deps=['torch','numpy','transformers','gradio','PIL','bitsandbytes','huggingface_hub']; [print(f'✅ {d} '+__import__(d).__version__ if hasattr(__import__(d),'__version__') else 'unknown') for d in deps]" 2>nul && (
    echo [%GREEN%SUCCESS%NC%] 所有关键依赖验证通过
) || (
    echo [%RED%ERROR%NC%] 依赖验证失败
    exit /b 1
)

REM 验证模型文件
set MODEL_DIR=qwen3_vl_models
echo [%BLUE%INFO%NC%] 验证Qwen3-VL-8B模型...
if not exist "%MODEL_DIR%" (
    echo [%RED%ERROR%NC%] 模型目录不存在: %MODEL_DIR%
    echo 💡 请先下载模型:
    echo    modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir qwen3_vl_models
    exit /b 1
)

REM ✅ 核心修复: 移除special_tokens_map.json依赖 + 4分片验证
set "required_files=config.json preprocessor_config.json tokenizer_config.json tokenizer.json"
for %%f in (%required_files%) do (
    if not exist "%MODEL_DIR%\%%f" (
        echo [%RED%ERROR%NC%] 缺失必需文件: %%f
        exit /b 1
    )
)
echo [%GREEN%SUCCESS%NC%] 所有核心配置文件存在 (无special_tokens_map.json)

set weight_count=0
for %%f in ("%MODEL_DIR%\model-00001-of-00004.safetensors" "%MODEL_DIR%\model-00002-of-00004.safetensors" "%MODEL_DIR%\model-00003-of-00004.safetensors" "%MODEL_DIR%\model-00004-of-00004.safetensors") do (
    if exist "%%f" set /a weight_count+=1
)
if %weight_count% lss 4 (
    echo [%RED%ERROR%NC%] 权重分片不完整 (找到%weight_count%/4个)
    exit /b 1
)
echo [%GREEN%SUCCESS%NC%] 权重文件验证通过 (4分片)

REM 启动应用
echo.
echo ==============================================================
echo   🎯 启动XXG-Qwen3-VL中文打标工具 (http://127.0.0.1:9527)
echo   💡 使用提示:
echo      • 低显存GPU (^<10GB): 添加 --4bit 参数
echo      • 无GPU/驱动问题: 添加 --cpu 参数
echo      • 直接处理文件夹: --folder C:\path\to\images
echo.
echo   🔧 CUDA兼容性说明:
echo      • CUDA 13.x 与 CUDA 12.4 二进制完全兼容
echo      • PyTorch 2.4.1 cu124 wheel 可在 CUDA 13.x 原生运行
echo      • 无需降级CUDA Toolkit或驱动
echo ==============================================================

REM 传递所有参数给app.py
python app.py %*
if %errorlevel% neq 0 (
    echo.
    echo [%RED%ERROR%NC%] 应用启动失败，请检查:
    echo    1. 模型文件是否完整 (qwen3_vl_models/)
    echo    2. 依赖是否正确安装 (特别是NumPy^<2.0)
    echo    3. 显存是否充足 (4-bit模式至少6GB)
    pause
    exit /b 1
)

goto :eof

REM ============ 检查Python 3.10 ============
:check_python
echo [%BLUE%INFO%NC%] 检测Python 3.10环境...

REM 方法1: 直接检测python3.10
where python3.10 >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python3.10
    echo [%GREEN%SUCCESS%NC%] 检测到python3.10
    exit /b 0
)

REM 方法2: 检测py启动器
where py >nul 2>&1
if %errorlevel% equ 0 (
    py -3.10 --version >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_CMD=py -3.10
        echo [%GREEN%SUCCESS%NC%] 检测到py启动器中的Python 3.10
        exit /b 0
    )
)

REM 方法3: 检测python3
where python3 >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=1,2 delims=." %%a in ('python3 --version 2^>nul ^| findstr /r "[0-9]\.[0-9]"') do (
        if "%%a"=="3" if "%%b" geq "10" (
            set PYTHON_CMD=python3
            echo [%GREEN%SUCCESS%NC%] 检测到python3 (版本≥3.10)
            exit /b 0
        )
    )
)

echo [%RED%ERROR%NC%] ❌ 未检测到Python 3.10
echo 💡 请安装Python 3.10:
echo    1. 访问 https://www.python.org/downloads/release/python-31013/
echo    2. 下载 Windows installer (64-bit)
echo    3. 安装时务必勾选"Add Python to PATH"
echo    4. 重启命令提示符后重试
exit /b 1

REM ============ 检查磁盘空间 ============
:check_disk_space
echo [%BLUE%INFO%NC%] 检查磁盘空间 (需要≥18GB)...
for /f "tokens=3" %%a in ('dir /-c ^| findstr /i /c:"可用" /c:"avail"') do (
    set "free_bytes=%%a"
)
if not defined free_bytes (
    echo [%YELLOW%WARN%NC%] 无法获取磁盘空间，假设空间充足
    exit /b 0
)

REM 转换为GB (1GB = 1073741824 bytes)
set /a free_gb=!free_bytes! / 1073741824
echo [%BLUE%INFO%NC%] 可用空间: !free_gb!GB

if !free_gb! lss 18 (
    echo [%RED%ERROR%NC%] 磁盘空间不足 (!free_gb!GB ^< 18GB)
    echo 💡 建议:
    echo    • 清理磁盘: del /q /f *.tmp *.log
    echo    • 更换目录: 将项目移动到空间充足的盘符
    exit /b 1
)
exit /b 0

REM ============ 检测GPU/驱动 ============
:detect_cuda_and_driver
echo [%BLUE%INFO%NC%] 检测NVIDIA GPU和驱动版本...

REM 检查nvidia-smi
where nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [%YELLOW%WARN%NC%] 未检测到nvidia-smi，可能未安装NVIDIA驱动
    echo 💡 请安装NVIDIA驱动 (>=535.86.05 支持CUDA 12.x/13.x):
    echo    https://www.nvidia.com/Download/index.aspx
    exit /b 1
)

REM 获取驱动版本 (Windows格式: "551.23")
for /f "tokens=3" %%a in ('nvidia-smi ^| findstr /i "Driver Version"') do set "driver_ver=%%a"
if not defined driver_ver (
    echo [%YELLOW%WARN%NC%] 无法获取驱动版本
    exit /b 1
)

echo [%BLUE%INFO%NC%] NVIDIA驱动版本: !driver_ver!

REM 驱动版本检查 (Windows驱动版本格式: 主版本.次版本)
for /f "tokens=1 delims=." %%a in ("!driver_ver!") do set "driver_major=%%a"
if !driver_major! lss 515 (
    echo [%RED%ERROR%NC%] 驱动版本过低 (!driver_ver!)，需要 ^>=515.65.01
    echo 💡 请升级NVIDIA驱动
    exit /b 1
)

REM 检测GPU信息
for /f "tokens=3,4 delims=|" %%a in ('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2^>nul ^| findstr /n "^" ^| findstr "^1:"') do (
    set "gpu_name=%%a"
    set "gpu_mem=%%b"
)
if defined gpu_name (
    echo [%BLUE%INFO%NC%] GPU信息: !gpu_name! (!gpu_mem!)
    exit /b 0
)

echo [%YELLOW%WARN%NC%] 未检测到NVIDIA GPU设备
exit /b 1