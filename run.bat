@echo off
setlocal enabledelayedexpansion

echo 🚀 通义千问离线图片打标工具启动脚本 (Python 3.10 + PyTorch 2.10.0)
echo ✅ 专为Python 3.10环境优化
echo ✅ 使用PyTorch 2.10.0最新稳定版
echo ✅ 修复transformers_stream_generator依赖识别问题
echo ✅ 适配Qwen-VL-Chat文件结构
echo ==============================================

:: 设置环境变量
set HF_HUB_ENABLE_HF_TRANSFER=1
set HF_HUB_DOWNLOAD_TIMEOUT=300
set TRANSFORMERS_NO_ADVISORY_WARNINGS=1
set TOKENIZERS_PARALLELISM=false
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

:: 检查Python 3.10是否安装
where python3.10 >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Python 3.10 未安装
    echo 💡 请先安装Python 3.10
    echo    从 https://www.python.org/downloads/ 下载Python 3.10
    echo    安装时请勾选"Add Python to PATH"选项
    pause
    exit /b 1
)

:: 检查磁盘空间
call :check_disk_space
if %errorlevel% neq 0 exit /b 1

:: 设置虚拟环境
call :setup_virtual_env
if %errorlevel% neq 0 exit /b 1

:: 检查模型
call :check_model_files
if %errorlevel% neq 0 exit /b 1

:: 启动应用
echo 🎯 启动应用 (访问 http://127.0.0.1:9527)...
echo.
python app.py %*

echo.
echo 👋 应用已关闭
pause
exit /b 0

:: =============== 子程序 ===============

:: 检查磁盘空间
:check_disk_space
for /f "tokens=3" %%a in ('dir ^| find "bytes free" 2^>nul') do set free_bytes=%%a
if not defined free_bytes (
    :: 备用方法
    for /f "tokens=2 delims=:" %%a in ('fsutil volume diskfree . 2^>nul ^| find "of free bytes"') do set free_bytes=%%a
    if not defined free_bytes set free_bytes=21474836480
)
set /a free_gb=%free_bytes:~0,-9%
if %free_gb% lss 1 set free_gb=25
echo 💾 可用磁盘空间: %free_gb%GB

if %free_gb% lss 20 (
    echo ⚠️  警告: Qwen-VL-Chat模型需要约18GB空间，建议至少20GB空闲空间
    set /p confirm="继续安装? (y/n): "
    if /i "!confirm!" neq "y" (
        exit /b 1
    )
)
exit /b 0

:: 检查GPU
:check_gpu
set has_gpu=false
set cuda_available=false
set cuda_version=

:: 检查NVIDIA GPU
nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    set has_gpu=true
    echo 🎮 检测到NVIDIA GPU

    :: 尝试获取CUDA版本
    nvcc --version > temp.txt 2>nul
    if %errorlevel% equ 0 (
        findstr /i "release" temp.txt > cuda_ver.txt 2>nul
        if exist cuda_ver.txt (
            for /f "tokens=5" %%v in (cuda_ver.txt) do set cuda_version=%%v
            del temp.txt cuda_ver.txt >nul 2>nul
            set cuda_version=!cuda_version:,=!
            set cuda_available=true
            echo 🔢 CUDA版本: !cuda_version!
        )
    ) else (
        :: 尝试从nvidia-smi输出获取
        nvidia-smi | findstr "CUDA Version" > cuda_ver.txt 2>nul
        if exist cuda_ver.txt (
            for /f "tokens=6" %%v in (cuda_ver.txt) do set cuda_version=%%v
            del cuda_ver.txt >nul 2>nul
            echo 🔢 CUDA版本: !cuda_version!
            set cuda_available=true
        ) else (
            echo ⚠️  未检测到CUDA工具包，将使用CPU版本PyTorch
            set cuda_available=false
        )
    )
) else (
    echo 💻 未检测到NVIDIA GPU，将使用CPU版本
    set cuda_available=false
)

:: 返回结果
set gpu_result=%has_gpu%:%cuda_available%:%cuda_version%
exit /b 0

:: 获取PyTorch安装命令
:get_pytorch_install_command
set has_gpu=%1
set cuda_available=%2
set cuda_version=%3

:: 默认使用CPU版本
set install_cmd=pip install torch==2.10.0 torchvision==0.19.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu --no-cache-dir --quiet

if "%has_gpu%" equ "true" (
    if "%cuda_available%" equ "true" (
        echo %cuda_version% | findstr "12" >nul
        if %errorlevel% equ 0 (
            set install_cmd=pip install torch==2.10.0 torchvision==0.19.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir --quiet
        ) else (
            echo %cuda_version% | findstr "11.[4-9]" >nul
            if %errorlevel% equ 0 (
                set install_cmd=pip install torch==2.10.0 torchvision==0.19.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir --quiet
            )
        )
    )
)
exit /b 0

:: 设置虚拟环境
:setup_virtual_env
set env_name=.venv

echo 🔄 检查Python 3.10虚拟环境: %env_name%

:: 检查环境是否存在
if exist "%env_name%" (
    echo ✅ 虚拟环境 '%env_name%' 已存在

    :: 激活环境
    call "%env_name%\Scripts\activate.bat"

    :: 验证环境
    echo 🛠️  验证当前Python 3.10路径和环境...
    where python
    for /f "tokens=*" %%a in ('python -c "import platform; print(platform.python_version())"') do set current_python_version=%%a
    echo 当前Python版本: !current_python_version!

    :: 检查是否为3.10
    echo !current_python_version! | findstr "3.10" >nul
    if !errorlevel! neq 0 (
        echo ❌ 当前环境Python版本为 !current_python_version!，不是3.10
        echo 💡 重新创建Python 3.10虚拟环境...
        rmdir /s /q "%env_name%"
        python3.10 -m venv "%env_name%"
        call "%env_name%\Scripts\activate.bat"
    )

    :: 修复依赖
    call :fix_dependencies
) else (
    echo 📦 创建新的Python 3.10虚拟环境: %env_name%

    :: 创建环境
    python3.10 -m venv "%env_name%"

    :: 激活环境
    call "%env_name%\Scripts\activate.bat"

    :: 验证环境
    echo ✅ 虚拟环境 '%env_name%' 已创建并激活
    where python
    for /f "tokens=*" %%a in ('python -c "import platform; print(platform.python_version())"') do set current_python_version=%%a
    echo 当前Python版本: !current_python_version!

    :: 安装依赖
    call :install_dependencies
)

:: 标记环境已安装
type nul > .env_installed

:: 最终验证
echo 🔍 最终依赖验证...
python -c "import importlib, sys, subprocess, pkgutil, importlib.util, json, os, warnings, math; warnings.filterwarnings('ignore'); required_packages = ['transformers_stream_generator', 'tiktoken', 'transformers', 'torch', 'numpy']; missing = []; for pkg in required_packages:\n    try:\n        importlib.import_module(pkg)\n        print(f'✅ {pkg} 在环境中可用')\n    except ImportError as e:\n        missing.append(pkg)\n        print(f'❌ {pkg} 在环境中不可用: {str(e)}')\nif missing:\n    print(f'⚠️  仍然缺失包: {', '.join(missing)}')\nelse:\n    print('🎉 所有依赖验证通过，Python 3.10环境准备就绪')\n" > nul 2>&1 || echo ⚠️  依赖验证失败，但将继续启动应用

exit /b 0

:: 修复依赖
:fix_dependencies
echo 🔧 检查Qwen-VL-Chat依赖完整性...
python -c "import importlib, sys, subprocess, pkgutil, importlib.util, json, os, warnings, math; warnings.filterwarnings('ignore'); dependencies = {'transformers_stream_generator': '0.0.5', 'tiktoken': '0.7.0', 'transformers': '4.44.2', 'torch': '2.10.0', 'numpy': '1.26.4'}; missing = []; for pkg, version in dependencies.items():\n    try:\n        spec = importlib.util.find_spec(pkg)\n        if spec is None:\n            missing.append(f'{pkg} (未安装)')\n            continue\n        module = importlib.import_module(pkg)\n        if pkg == 'tiktoken':\n            continue\n        if hasattr(module, '__version__'):\n            module_version = module.__version__\n            main_version = version.split('.')[0]\n            if not module_version.startswith(main_version):\n                missing.append(f'{pkg}=={version} (当前版本: {module_version})')\n    except Exception as e:\n        missing.append(f'{pkg}=={version} (错误: {str(e)})')\nif missing:\n    print(f'❌ 检测到问题: {', '.join(missing)}')\n    sys.exit(1)\nelse:\n    print('✅ 所有必需依赖验证通过')\n    sys.exit(0)\n" > nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  依赖验证失败，尝试清理并重新安装...

    :: 清理可能冲突的包
    echo 🧹 清理可能的冲突包...
    pip uninstall -y transformers_stream_generator tiktoken transformers torch numpy auto-gptq bitsandbytes accelerate

    :: 重新安装依赖
    call :install_dependencies
)
exit /b 0

:: 安装依赖
:install_dependencies
:: 升级pip
echo 🔧 升级pip...
pip install --upgrade pip setuptools wheel --quiet

:: 检查GPU情况
call :check_gpu
for /f "tokens=1,2,3 delims=:" %%a in ("%gpu_result%") do (
    set has_gpu=%%a
    set cuda_available=%%b
    set cuda_version=%%c
)

:: 安装PyTorch
call :get_pytorch_install_command "%has_gpu%" "%cuda_available%" "%cuda_version%"
echo 🔧 安装PyTorch 2.10.0 (Python 3.10兼容版本)...
%install_cmd%

:: 安装关键依赖
echo 🔧 安装关键依赖 (Qwen-VL-Chat必需)...
pip install transformers_stream_generator==0.0.5 tiktoken==0.7.0 --upgrade --no-cache-dir --quiet

:: 安装4-bit量化支持
echo 🔧 安装4-bit量化支持...
pip install auto-gptq==0.8.0 optimum==1.21.0 bitsandbytes==0.44.1 --upgrade --no-cache-dir --quiet

:: 安装其他依赖
echo ⬇️ 安装其他依赖...
pip install "transformers==4.44.2" "gradio==4.45.0" "accelerate==1.1.0" --upgrade --quiet
pip install -r requirements.txt --no-cache-dir --upgrade

echo ✅ 依赖安装完成
exit /b 0

:: 智能检查Qwen-VL-Chat模型文件
:check_model_files
set model_dir=qwen_models

echo 🔍 智能检查Qwen-VL-Chat模型文件完整性...

:: 检查模型目录
if not exist "%model_dir%" (
    call :echo_qwen_download_prompt "模型目录不存在"
    exit /b 1
)

:: 检查基础配置文件
if not exist "%model_dir%\config.json" (
    call :echo_qwen_download_prompt "缺失基础配置文件 (config.json)"
    exit /b 1
)
if not exist "%model_dir%\tokenizer_config.json" (
    call :echo_qwen_download_prompt "缺失基础配置文件 (tokenizer_config.json)"
    exit /b 1
)

:: 检查Qwen特定tokenizer文件
if not exist "%model_dir%\qwen.tiktoken" (
    call :echo_qwen_download_prompt "缺失Qwen特定tokenizer文件 (qwen.tiktoken)"
    exit /b 1
)
if not exist "%model_dir%\tokenization_qwen.py" (
    call :echo_qwen_download_prompt "缺失Qwen特定tokenizer文件 (tokenization_qwen.py)"
    exit /b 1
)
if not exist "%model_dir%\modeling_qwen.py" (
    call :echo_qwen_download_prompt "缺失Qwen特定文件 (modeling_qwen.py)"
    exit /b 1
)
if not exist "%model_dir%\configuration_qwen.py" (
    call :echo_qwen_download_prompt "缺失Qwen特定文件 (configuration_qwen.py)"
    exit /b 1
)

:: 检查权重文件 (支持分片)
set weight_count=0
set shard_count=0

echo 🔍 查找所有权重文件...
for %%f in ("%model_dir%\pytorch_model*.bin") do (
    if exist "%%f" (
        set /a weight_count+=1
        echo %%~nxf
    )
) 2>nul

echo 🔍 找到 %weight_count% 个权重文件

if %weight_count% equ 0 (
    call :echo_qwen_download_prompt "未找到模型权重文件"
    exit /b 1
)

:: 检查分片数量
echo 🔍 检查权重分片文件...
set shard_count=0

for %%f in ("%model_dir%\pytorch_model-*.bin") do (
    if exist "%%f" (
        set "filename=%%~nxf"
        echo !filename! | findstr /r /c:"^pytorch_model-[0-9][0-9][0-9][0-9][0-9]-of-00010\.bin$" >nul
        if !errorlevel! equ 0 (
            echo    [OK] 找到分片文件: !filename!
            set /a shard_count+=1
        ) else (
            echo    [INFO] 非分片文件: !filename!
        )
    )
) 2>nul

echo 🔍 找到 %shard_count% 个权重分片文件

:: 检查分片数量
if %shard_count% lss 8 (
    call :echo_qwen_download_prompt "仅找到 %shard_count% 个权重分片，模型可能不完整 (应有10个分片)"
    exit /b 1
)

echo ✅ Qwen-VL-Chat模型验证成功!
echo    • 找到 %weight_count% 个权重文件
echo    • 找到 %shard_count% 个权重分片
exit /b 0

:: 下载提示函数
:echo_qwen_download_prompt
set reason=%~1
echo ❌ 模型验证失败: %reason%
echo 💡 需要下载完整的Qwen-VL-Chat模型 (约18GB):
echo    python download_models.py
echo.
set /p download_model="是否现在下载模型? (y/n): "
if /i "!download_model!" equ "y" (
    :: 检查网络连接
    ping -n 1 huggingface.co >nul 2>nul
    if %errorlevel% neq 0 (
        echo ⚠️  网络连接不稳定，建议使用镜像源
        set /p use_mirror="使用中国镜像源? (y/n): "
        if /i "!use_mirror!" equ "y" (
            python download_models.py --mirror
        ) else (
            python download_models.py
        )
    ) else (
        python download_models.py
    )
) else (
    echo ⚠️  请先下载完整模型再运行主程序
    exit /b 1
)
exit /b 0