@echo off
chcp 65001 >nul 2>&1 || chcp 936 >nul 2>&1
setlocal enabledelayedexpansion

echo ==============================================================
echo   Qwen3-VL-8B-Instruct 模型下载工具 (Windows版)
echo   ✅ 国内ModelScope优先 (50-100MB/s) | ✅ 4分片权重验证
echo   ✅ 无special_tokens_map.json依赖 | ✅ 智能磁盘空间检查
echo   By 西小瓜 / Wechat:priest-mos
echo ==============================================================

REM 设置环境变量
set MODEL_DIR=qwen3_vl_models
set MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct
set REQUIRED_SPACE_GB=18

REM 颜色输出 (Windows 10+ 支持)
if not defined ESC (
    for /F "tokens=1,2 delims=#" %%a in ('"prompt #$H#$E# & echo on & for %%b in (1) do rem"') do set "ESC=%%a"
)
set RED=%ESC%[91m
set GREEN=%ESC%[92m
set YELLOW=%ESC%[93m
set BLUE=%ESC%[94m
set NC=%ESC%[0m

REM 日志函数
:log
echo [%BLUE%INFO%NC%] %~1
goto :eof

:warn
echo [%YELLOW%WARN%NC%] %~1
goto :eof

:error
echo [%RED%ERROR%NC%] %~1
goto :eof

:success
echo [%GREEN%SUCCESS%NC%] %~1
goto :eof

REM 检查磁盘空间 (智能适配Windows)
call :check_disk_space
if %errorlevel% neq 0 exit /b 1

REM 检查Python 3.10
call :check_python
if %errorlevel% neq 0 exit /b 1

REM 创建模型目录
if not exist "%MODEL_DIR%" (
    mkdir "%MODEL_DIR%" 2>nul
    if !errorlevel! neq 0 (
        call :error "无法创建模型目录: %MODEL_DIR%"
        exit /b 1
    )
)
call :log "模型目录: %cd%\%MODEL_DIR%"

REM 检查是否已存在完整模型
call :verify_model "%MODEL_DIR%"
if !errorlevel! equ 0 (
    call :success "✅ 模型已存在且完整，跳过下载"
    goto :launch_info
)

REM 选择下载源
echo.
call :log "选择下载源:"
echo   1. ModelScope (国内镜像，推荐 ⭐ 50-100MB/s)
echo   2. Hugging Face (国际源，需代理/良好网络)
set /p choice="输入选项 (1/2) [默认:1]: "
if "!choice!"=="" set choice=1

echo.
if "!choice!"=="1" (
    call :download_modelscope
) else if "!choice!"=="2" (
    call :download_hf
) else (
    call :error "无效选项，仅支持 1 或 2"
    exit /b 1
)

REM 验证下载完整性
echo.
call :log "验证模型完整性..."
call :verify_model "%MODEL_DIR%"
if !errorlevel! neq 0 (
    call :error "模型验证失败，请重新下载"
    exit /b 1
)

:launch_info
echo.
call :success "✅ 模型下载/验证完成!"
echo.
echo  📁 模型路径: %cd%\%MODEL_DIR%
echo  💾 总大小: 约14GB (4分片结构)
echo  🚀 下一步: 双击 run.bat 启动打标工具
echo.
pause
exit /b 0

REM ============ 智能磁盘空间检查 ============
:check_disk_space
call :log "检查磁盘空间 (需要≥%REQUIRED_SPACE_GB%GB)..."
for /f "tokens=3" %%a in ('dir /-c "%cd%" ^| findstr /i /c:"可用"') do (
    set "free_bytes=%%a"
)
if not defined free_bytes (
    call :warn "无法获取磁盘空间，假设空间充足"
    exit /b 0
)

REM 转换为GB (1GB = 1073741824 bytes)
set /a free_gb=!free_bytes! / 1073741824
call :log "可用空间: !free_gb!GB"

if !free_gb! lss %REQUIRED_SPACE_GB% (
    call :error "磁盘空间不足 (!free_gb!GB ^< %REQUIRED_SPACE_GB%GB)"
    call :log "💡 建议:"
    call :log "   • 清理磁盘: del /q /f *.tmp *.log"
    call :log "   • 更换目录: 将项目移动到空间充足的盘符"
    exit /b 1
)
exit /b 0

REM ============ 检查Python 3.10 ============
:check_python
call :log "检测Python 3.10环境..."

REM 方法1: 直接检测python3.10
where python3.10 >nul 2>&1
if !errorlevel! equ 0 (
    set PYTHON_CMD=python3.10
    call :log "✅ 检测到python3.10"
    exit /b 0
)

REM 方法2: 检测py启动器
where py >nul 2>&1
if !errorlevel! equ 0 (
    py -3.10 --version >nul 2>&1
    if !errorlevel! equ 0 (
        set PYTHON_CMD=py -3.10
        call :log "✅ 检测到py启动器中的Python 3.10"
        exit /b 0
    )
)

REM 方法3: 检测python3
where python3 >nul 2>&1
if !errorlevel! equ 0 (
    for /f "tokens=1,2 delims=." %%a in ('python3 --version 2^>nul ^| findstr /r "[0-9]\.[0-9]"') do (
        if "%%a"=="3" if "%%b" geq "10" (
            set PYTHON_CMD=python3
            call :log "✅ 检测到python3 (版本≥3.10)"
            exit /b 0
        )
    )
)

call :error "❌ 未检测到Python 3.10"
call :log "💡 请安装Python 3.10:"
call :log "   1. 访问 https://www.python.org/downloads/release/python-31013/"
call :log "   2. 下载 Windows installer (64-bit)"
call :log "   3. 安装时务必勾选 ^"Add Python to PATH^""
call :log "   4. 重启命令提示符后重试"
exit /b 1

REM ============ ModelScope下载 (国内加速) ============
:download_modelscope
call :log "使用ModelScope下载 (国内镜像加速)..."
call :log "💡 优势: 无需代理 | 50-100MB/s | 自动断点续传"

REM 安装modelscope (用户目录)
call :log "   安装ModelScope (用户目录)..."
%PYTHON_CMD% -m pip install modelscope -q --user 2>nul
if !errorlevel! neq 0 (
    call :error "ModelScope安装失败"
    exit /b 1
)

REM 修复PATH (用户目录Scripts)
for /f "delims=" %%i in ('echo %LOCALAPPDATA%\Programs\Python\Python310\Scripts') do set "USER_SCRIPTS=%%i"
if not exist "!USER_SCRIPTS!" (
    for /f "delims=" %%i in ('echo %APPDATA%\Python\Python310\Scripts') do set "USER_SCRIPTS=%%i"
)
if exist "!USER_SCRIPTS!" (
    set "PATH=!USER_SCRIPTS!;!PATH!"
)

REM 验证modelscope命令
where modelscope >nul 2>&1
if !errorlevel! neq 0 (
    call :error "modelscope命令不可用，请检查PATH"
    exit /b 1
)

REM 下载模型 (清理进度条混杂)
call :log "   开始下载Qwen3-VL-8B-Instruct (约14GB)..."
powershell -Command "$progress = ''; modelscope download --model '%MODEL_NAME%' --local_dir '%MODEL_DIR%' 2>&1 | ForEach-Object { if ($_ -match '100%%') { Write-Host $_ } elseif ($_ -match 'Downloading') { $progress = $_.Trim(); Write-Host -NoNewline \"`r⏳ 正在下载... \" } }; Write-Host \"`r                                                                                        `r✅ ModelScope下载完成!\""

if not exist "%MODEL_DIR%\config.json" (
    call :error "下载失败或中断，请重新运行本脚本"
    exit /b 1
)
call :success "✅ ModelScope下载完成!"
exit /b 0

REM ============ Hugging Face下载 (国际源) ============
:download_hf
call :log "使用Hugging Face下载 (国际源)..."
call :log "💡 提示: 需要良好网络或代理，国内用户建议选择ModelScope"

REM 安装huggingface_hub (用户目录)
call :log "   安装huggingface_hub..."
%PYTHON_CMD% -m pip install huggingface_hub hf_transfer -q --user 2>nul
if !errorlevel! neq 0 (
    call :error "huggingface_hub安装失败"
    exit /b 1
)

REM 设置环境变量 (加速下载)
set HF_HUB_ENABLE_HF_TRANSFER=1
set HF_HUB_DOWNLOAD_TIMEOUT=600

REM 下载模型
call :log "   开始下载Qwen3-VL-8B-Instruct (约14GB)..."
powershell -Command "$env:HF_HUB_ENABLE_HF_TRANSFER=1; huggingface-cli download '%MODEL_NAME%' --local-dir '%MODEL_DIR%' --local-dir-use-symlinks false 2>&1 | ForEach-Object { if ($_ -match '100%%') { Write-Host $_ } elseif ($_ -match 'eta') { Write-Host -NoNewline \"`r⏳ $_\" } }; Write-Host \"`r                                                                                        `r✅ Hugging Face下载完成!\""

if not exist "%MODEL_DIR%\config.json" (
    call :error "下载失败或中断，请检查网络或代理设置"
    exit /b 1
)
call :success "✅ Hugging Face下载完成!"
exit /b 0

REM ============ 智能模型验证 (4分片 + 无special_tokens_map.json) ============
:verify_model
set "model_dir=%~1"

REM 检查目录存在
if not exist "%model_dir%" (
    call :error "模型目录不存在: %model_dir%"
    exit /b 1
)

REM ✅ 核心修复: 移除special_tokens_map.json依赖
set "required_files=config.json preprocessor_config.json tokenizer_config.json tokenizer.json"
for %%f in (%required_files%) do (
    if not exist "%model_dir%\%%f" (
        call :error "缺失必需文件: %%f"
        exit /b 1
    )
)
call :log "✅ 所有核心配置文件存在 (无special_tokens_map.json)"

REM 检查4分片权重 (Qwen3-VL-8B官方结构)
set weight_count=0
for %%f in ("%model_dir%\model-00001-of-00004.safetensors" "%model_dir%\model-00002-of-00004.safetensors" "%model_dir%\model-00003-of-00004.safetensors" "%model_dir%\model-00004-of-00004.safetensors") do (
    if exist "%%f" set /a weight_count+=1
)

if !weight_count! lss 4 (
    call :error "权重分片不完整 (找到!weight_count!/4个)"
    call :log "💡 请重新下载完整模型"
    exit /b 1
)
call :log "✅ 权重文件验证通过 (4分片)"

REM 检查总大小 (约14GB)
set total_size=0
for %%f in ("%model_dir%\model-0000*-of-00004.safetensors") do (
    for /f "usebackq tokens=3*" %%a in (`dir "%%f" ^| findstr "File(s)"`) do set "file_size=%%a"
    set /a total_size+=file_size
)
set /a total_gb=!total_size! / 1073741824
if !total_gb! lss 12 (
    call :warn "模型总大小较小 (!total_gb!GB)，可能下载不完整 (Qwen3-VL-8B约14GB)"
) else (
    call :log "✅ 模型总大小: !total_gb!GB"
)

call :success "✅ Qwen3-VL-8B模型验证成功!"
exit /b 0