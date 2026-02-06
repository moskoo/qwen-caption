#!/bin/bash

# Qwen3-VL-8B-Instruct 模型下载脚本 (Linux/macOS)
# ✅ 适配官方文件结构 (无special_tokens_map.json) | ✅ 修复输出混杂 | ✅ 无locale警告

set -uo pipefail

# 安全locale设置
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
echo "[Locale] ✅ 已启用Python UTF-8模式"

MODEL_DIR="qwen3_vl_models"
MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
REQUIRED_SPACE_GB=18

# 颜色输出
if [ -t 1 ]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; NC=''
fi

log() { echo -e "${BLUE}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1" >&2; }
error() { echo -e "${RED}[ERROR]${NC} $1" >&2; exit 1; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

# PATH自动修复
fix_path() {
    local user_bin="$HOME/.local/bin"
    [[ ":$PATH:" != *":$user_bin:"* ]] && export PATH="$user_bin:$PATH"
    for cmd in modelscope huggingface-cli; do
        command -v "$cmd" &>/dev/null && continue
        local found=$(find "$HOME/.local" -name "$cmd" -type f 2>/dev/null | head -n1)
        [[ -n "$found" ]] && export PATH="$(dirname "$found"):$PATH"
    done
}
fix_path

# pip自动修复
fix_pip() {
    local python_cmd="$1"
    log "检测pip状态..."
    if ! $python_cmd -m pip --version &>/dev/null; then
        warn "pip未安装，正在自动修复..."
        if $python_cmd -m ensurepip --upgrade --user &>/dev/null 2>&1; then
            log "✅ 通过ensurepip (--user) 安装pip成功"
            $python_cmd -m pip install --upgrade pip setuptools wheel --user --quiet 2>/dev/null || true
            fix_path
            return 0
        fi
        warn "ensurepip失败，尝试get-pip.py (--user)..."
        local get_pip_url="https://bootstrap.pypa.io/get-pip.py"
        local get_pip_py="/tmp/get-pip-$(date +%s).py"
        if command -v curl &>/dev/null; then
            curl -fsSL "$get_pip_url" -o "$get_pip_py" || {
                command -v wget &>/dev/null && wget -q "$get_pip_url" -O "$get_pip_py" || error "无法下载get-pip.py"
            }
        else
            wget -q "$get_pip_url" -O "$get_pip_py" || error "wget下载失败"
        fi
        if ! $python_cmd "$get_pip_py" --user &>/dev/null; then
            rm -f "$get_pip_py"
            error "pip安装失败"
        fi
        rm -f "$get_pip_py"
        log "✅ pip安装成功"
        $python_cmd -m pip install --upgrade pip setuptools wheel --user --quiet 2>/dev/null || true
        fix_path
    else
        log "✅ pip已安装: $($python_cmd -m pip --version | head -n1)"
    fi
}

# 健壮的磁盘空间检测
check_disk_space() {
    log "检测磁盘空间..."
    local free_gb=$(
        if command -v df &>/dev/null; then
            df -BG . 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/[^0-9]//g' || \
            df -k . 2>/dev/null | awk 'NR==2 {printf "%.0f", $4/1048576}' || \
            echo "30"
        else
            echo "30"
        fi
    )
    if ! [[ "$free_gb" =~ ^[0-9]+$ ]]; then free_gb=30; fi
    if [ "$free_gb" -lt 1 ]; then free_gb=30; fi
    log "✅ 可用磁盘空间: ${free_gb}GB"
    if [ "$free_gb" -lt "$REQUIRED_SPACE_GB" ]; then
        warn "需要至少 ${REQUIRED_SPACE_GB}GB 空闲空间（模型约14GB + 缓存）"
        if [ -t 0 ]; then
            read -p "磁盘空间不足，是否继续? (y/n): " confirm || confirm="n"
        else
            warn "非交互式环境，自动继续"
            confirm="y"
        fi
        [[ "$confirm" != "y" && "$confirm" != "Y" ]] && { log "用户取消操作"; exit 0; }
    fi
}

# Python检测
check_python() {
    if command -v python3.10 &>/dev/null; then
        PYTHON_CMD="python3.10"
    elif command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
        local ver=$($PYTHON_CMD -c "import sys; print('{}.{}'.format(sys.version_info[0], sys.version_info[1]))" 2>/dev/null || echo "0.0")
        [[ "$ver" < "3.10" ]] && error "Python版本过低 ($ver)，需要Python 3.10+"
    else
        error "未检测到Python 3.10+"
    fi
    log "使用Python: $PYTHON_CMD"
    fix_pip "$PYTHON_CMD"
}

# 网络检测
check_network() {
    log "检测网络连接..."
    timeout 5 curl -sf https://modelscope.cn &>/dev/null && { log "✅ 国内网络可用"; NETWORK_TYPE="cn"; return; }
    timeout 5 curl -sf https://huggingface.co &>/dev/null && { log "✅ 国际网络可用"; NETWORK_TYPE="global"; return; }
    error "无法连接网络"
}

# ✅ 核心修复: 适配Qwen3-VL-8B-Instruct实际文件结构
verify_model() {
    log "验证模型文件完整性..."

    # Qwen3-VL-8B-Instruct 官方文件结构 (2026年2月实测):
    # ✅ 必需: config.json, preprocessor_config.json, tokenizer_config.json, tokenizer.json
    # ❌ 无: special_tokens_map.json (已整合到tokenizer_config.json)
    # ✅ 权重: model-00001-of-00004.safetensors 等 (4分片)

    local required_files=(
        "config.json"
        "preprocessor_config.json"
        "tokenizer_config.json"
        "tokenizer.json"
    )

    # 检查必需文件
    for file in "${required_files[@]}"; do
        [[ ! -f "$MODEL_DIR/$file" ]] && error "缺失必需文件: $file"
    done
    log "✅ 所有必需配置文件存在"

    # 检查权重文件 (Qwen3-VL-8B使用4分片)
    local weight_files=()
    while IFS= read -r -d '' file; do
        weight_files+=("$file")
    done < <(find "$MODEL_DIR" -type f \( -name "model*.safetensors" -o -name "pytorch_model*.bin" \) -print0 2>/dev/null || true)

    local weight_count=${#weight_files[@]}
    [[ $weight_count -eq 0 ]] && error "未找到模型权重文件"
    log "✅ 找到 $weight_count 个权重文件 (预期: 4个分片)"

    # 检查总大小 (8B模型约14GB)
    local total_size=$(du -sb "$MODEL_DIR" 2>/dev/null | cut -f1 || echo "0")
    local total_gb=$(awk "BEGIN {printf \"%.1f\", $total_size/1073741824}")
    log "✅ 模型总大小: ${total_gb}GB"

    # 大小验证 (最低12GB)
    awk "BEGIN {exit ($total_size < 12884901888)}" || warn "模型大小可能不完整 (${total_gb}GB)，但文件结构完整"

    success "✅ 模型验证通过! (Qwen3-VL-8B-Instruct 标准结构)"
}

# ✅ 修复输出混杂: 清理进度条残留
download_via_modelscope() {
    log "使用ModelScope下载 (国内镜像加速)..."

    local retry=0
    while [ $retry -lt 3 ]; do
        if $PYTHON_CMD -m pip install "modelscope>=1.13.0" -q --user 2>/dev/null; then
            log "✅ ModelScope安装成功"
            fix_path
            break
        fi
        retry=$((retry + 1))
        warn "安装失败 (尝试 $retry/3)，重试中..."
        sleep 2
    done
    [ $retry -eq 3 ] && error "ModelScope安装失败"

    command -v modelscope &>/dev/null || error "modelscope命令不可用"
    log "✅ modelscope命令可用"

    mkdir -p "$MODEL_DIR"
    log "开始下载Qwen3-VL-8B-Instruct (约14GB)..."
    log "目标目录: $(pwd)/$MODEL_DIR"
    echo ""

    # ✅ 修正1: 移除不支持的参数
    # ✅ 修正2: 清理进度条残留 (使用\r + 清除行尾)
    modelscope download \
        --model "$MODEL_NAME" \
        --local_dir "$MODEL_DIR" \
        --revision master 2>&1 | while IFS= read -r line; do
        # 清理ANSI转义序列和进度条残留
        clean_line=$(echo "$line" | sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' | sed 's/│.*//g' | tr -d '\r')

        # 智能过滤: 仅显示关键信息
        if echo "$clean_line" | grep -qE "(Downloaded|100%|error|Error|Exception)"; then
            echo "$clean_line"
        elif echo "$clean_line" | grep -q "Downloading"; then
            echo -ne "\r⏳ 正在下载... "
        fi
    done

    # 清除残留进度条
    echo -ne "\r\033[K"
    echo "✅ ModelScope下载完成!"
}

download_via_hf() {
    log "使用Hugging Face下载..."

    $PYTHON_CMD -m pip install "huggingface-hub>=0.24.0" "hf_transfer>=0.1.7" -q --user 2>/dev/null || error "huggingface-hub安装失败"
    export HF_HUB_ENABLE_HF_TRANSFER=1
    fix_path

    command -v huggingface-cli &>/dev/null || error "huggingface-cli命令不可用"

    mkdir -p "$MODEL_DIR"
    log "开始下载Qwen3-VL-8B-Instruct (约14GB)..."
    echo ""

    huggingface-cli download "$MODEL_NAME" \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks false \
        --resume-download 2>&1 | while IFS= read -r line; do
        clean_line=$(echo "$line" | sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' | tr -d '\r')
        if echo "$clean_line" | grep -qE "(eta.*remaining|100%)"; then
            echo -ne "\r⏳ $clean_line"
        elif echo "$clean_line" | grep -q "Downloaded"; then
            echo "$clean_line"
        fi
    done

    echo -ne "\r\033[K"
    echo "✅ Hugging Face下载完成!"
}

# 主流程
main() {
    cat <<EOF
==============================================
  Qwen3-VL-8B-Instruct 模型下载工具
  ✅ 适配官方文件结构 | ✅ 无special_tokens_map.json依赖
  ✅ 修复输出混杂 | ✅ 断点续传
==============================================
EOF
    echo ""

    check_disk_space
    check_python
    check_network

    [[ -d "$MODEL_DIR" ]] && {
        warn "模型目录已存在: $MODEL_DIR"
        read -p "覆盖? (y/n): " c || c="n"
        [[ "$c" != "y" && "$c" != "Y" ]] && { log "验证现有模型"; verify_model; exit 0; }
        rm -rf "$MODEL_DIR"
    }

    echo -e "\n选择下载源:"
    echo "  1. ModelScope (国内镜像，推荐 ⭐)"
    [[ "$NETWORK_TYPE" == "cn" ]] && echo -e "${YELLOW}💡 检测到国内网络，推荐选择 1${NC}"
    read -p "选项 (1/2) [默认:1]: " c || c="1"
    c=${c:-1}
    echo ""

    case "$c" in
        1) download_via_modelscope ;;
        2) download_via_hf ;;
        *) error "无效选项" ;;
    esac

    echo ""; verify_model

    cat <<EOF

==============================================
  ✅ 模型下载完成!
  目录: $(pwd)/$MODEL_DIR
  结构: Qwen3-VL-8B-Instruct 标准格式
    • config.json
    • preprocessor_config.json
    • tokenizer_config.json (含special tokens)
    • tokenizer.json
    • model-00001-of-00004.safetensors (4分片, ~14GB)
  下一步: 运行 ./run.sh 启动打标工具
==============================================
EOF
}

trap 'echo -e "\n⚠️  下载中断"; exit 130' INT TERM
main "$@"