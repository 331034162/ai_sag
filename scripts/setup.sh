#!/usr/bin/env bash
# ============================================================
# ai_sag 一键环境搭建 & 启动脚本 (Linux / macOS)
#
# 用法：
#   ./setup.sh                   默认 CPU 模式，全部安装 + 启动
#   ./setup.sh gpu               GPU 模式
#   ./setup.sh check             仅检查环境
#   ./setup.sh install           仅安装依赖（不启动）
#   ./setup.sh start             仅启动服务（不安装）
#   ./setup.sh help              显示帮助
#
# 首次使用会引导你配置 .env 文件。
# ============================================================

set -euo pipefail

# ---- 路径 ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$ROOT_DIR/ai_sag"
ENV_DIR="$ROOT_DIR/.venv"
API_PORT="${API_PORT:-8777}"
WEB_PORT="${WEB_PORT:-8080}"
MODE="cpu"

# ---- 颜色 ----
if [ -t 1 ]; then
    C_RESET='\033[0m'
    C_RED='\033[91m'
    C_GREEN='\033[92m'
    C_YELLOW='\033[93m'
    C_BLUE='\033[94m'
    C_CYAN='\033[96m'
    C_BOLD='\033[1m'
else
    C_RESET='' C_RED='' C_GREEN='' C_YELLOW='' C_BLUE='' C_CYAN='' C_BOLD=''
fi

# ---- 解析参数 ----
ACTION="all"
case "${1:-}" in
    gpu)     MODE="gpu"  ;;
    check)   ACTION="check"   ;;
    install) ACTION="install" ;;
    start)   ACTION="start"   ;;
    help|-h|--help) ACTION="help" ;;
    "")      ;;
    *)       echo -e "${C_RED}[ERROR]${C_RESET} 未知参数: $1（可用: gpu / check / install / start / help）"
             exit 1 ;;
esac

title() { echo -e "\n  ${C_BOLD}${C_BLUE}══════════════════════════════════════════════════${C_RESET}"
          echo -e "  ${C_BOLD}${C_BLUE}  $*${C_RESET}"
          echo -e "  ${C_BOLD}${C_BLUE}══════════════════════════════════════════════════${C_RESET}" ; }
step()  { echo -e "  ${C_CYAN}[$*]${C_RESET}" ; }
ok()    { echo -e "  ${C_GREEN}✓${C_RESET} $*" ; }
warn()  { echo -e "  ${C_YELLOW}⚠${C_RESET} $*" ; }

title "ai_sag 环境搭建脚本"
echo -e "  仓库根目录: $ROOT_DIR"
echo -e "  Python 包:  $SRC_DIR"
echo -e "  虚拟环境:  $ENV_DIR"
if [ "$MODE" = "gpu" ]; then
    echo -e "  运行模式:  ${C_CYAN}GPU (CUDA 12.4)${C_RESET}"
else
    echo -e "  运行模式:  CPU"
fi

if [ "$ACTION" = "help" ]; then
    echo ""
    echo "用法: ./setup.sh [选项]"
    echo ""
    echo "选项："
    echo "  （无参数）   CPU 模式，安装依赖 + 启动服务"
    echo "  gpu         GPU 模式，安装 CUDA 依赖 + 启动服务"
    echo "  check       仅检查 Python / MySQL / .env 环境"
    echo "  install     仅安装依赖，不启动服务"
    echo "  start       仅启动 API + Web UI"
    echo "  help        显示此帮助"
    echo ""
    echo "首次使用："
    echo "  1. 准备 .env 文件（会从 .env.example 复制模板）"
    echo "  2. 编辑 .env 填写 MySQL / LLM API Key / Embedding 路径"
    echo "  3. 运行 ./setup.sh"
    echo ""
    echo "更多帮助：docs/STARTUP.md"
    exit 0
fi

# ---- 检查 Python ----
step "检查 Python 环境..."
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo -e "${C_RED}[ERROR]${C_RESET} 未找到 Python，请先安装 Python 3.10+"
    exit 1
fi

PYTHON=$(command -v python3 || command -v python)
PY_VER=$("$PYTHON" --version 2>&1 | awk '{print $2}')
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo -e "${C_RED}[ERROR]${C_RESET} Python 版本过低: $PY_VER，需要 3.10+"
    exit 1
fi
ok "Python $PY_VER ($PYTHON)"

# ---- 仅检查模式 ----
check_env() {
    step "检查环境变量..."
    local env_path
    env_path=$("$PYTHON" -c "from dotenv import find_dotenv; print(find_dotenv() or 'NOT_FOUND')" 2>/dev/null || echo "NOT_FOUND")
    if [ "$env_path" = "NOT_FOUND" ]; then
        warn "未找到 .env，首次启动将引导配置"
    else
        ok ".env 已找到"
    fi

    # 检查 .env 中的必填项（MySQL / 各场景 _LLM_NAME / Embedding 路径）
    "$PYTHON" -c "
import os
vars = {
    'SAG_MYSQL_HOST': 'MySQL地址', 'SAG_MYSQL_USER': 'MySQL用户',
    'SAG_MYSQL_PASSWORD': 'MySQL密码',
    'SAG_LLM_PROFILE_ANSWER_LLM_NAME': '答案生成场景 profile 名',
    'SAG_LLM_PROFILE_GENRE_CLASSIFY_LLM_NAME': '体裁分类场景 profile 名',
    'SAG_LLM_PROFILE_EVENT_EXTRACT_LLM_NAME': '事件抽取场景 profile 名',
    'SAG_LLM_PROFILE_QUERY_REWRITE_LLM_NAME': '查询重写场景 profile 名',
    'SAG_LLM_PROFILE_ENTITY_EXTRACT_LLM_NAME': '实体抽取场景 profile 名',
    'SAG_LLM_PROFILE_RERANK_LLM_NAME': '重排场景 profile 名',
    'SAG_EMBEDDING_MODEL_PATH': 'Embedding模型路径'
}
missing = [(k, d) for k, d in vars.items() if not os.environ.get(k)]
if missing:
    print()
    for k, d in missing:
        print(f'     - {d}: 未设置')
    print()
" 2>/dev/null || true

    # 检查 llm_profiles.yaml
    if [ ! -f "$SRC_DIR/llm_profiles.yaml" ]; then
        warn "未找到 llm_profiles.yaml，请从 llm_profiles.yaml.example 复制并填写 api_key"
    fi
}

if [ "$ACTION" = "check" ]; then
    check_env
    exit 0
fi

# ---- 创建虚拟环境 ----
if [ ! -f "$ENV_DIR/bin/python" ]; then
    step "创建虚拟环境..."
    "$PYTHON" -m venv "$ENV_DIR"
    ok "虚拟环境已创建: $ENV_DIR"
else
    step "虚拟环境已存在，跳过创建..."
    echo -e "  ${C_YELLOW}→${C_RESET} $ENV_DIR"
fi

# 激活
. "$ENV_DIR/bin/activate"

# 升级 pip
step "升级 pip..."
pip install --upgrade pip -q
ok "pip 已升级"

# ---- 安装依赖 ----
if [ "$ACTION" != "start" ]; then
    step "安装 Python 依赖（$MODE 模式）..."
    if [ "$MODE" = "gpu" ]; then
        # GPU 模式：requirements-gpu.txt 已自包含全部依赖，直接安装即可，不要叠加 CPU 版
        pip install -r "$SRC_DIR/requirements-gpu.txt" -q || { echo -e "${C_RED}[ERROR]${C_RESET} GPU 依赖安装失败，请手动检查"; exit 1; }
        ok "GPU 依赖安装完成"
    else
        pip install -r "$SRC_DIR/requirements.txt" -q
        ok "基础依赖安装完成"
    fi
fi

# ---- .env 检查 ----
if [ ! -f "$SRC_DIR/.env" ]; then
    step ".env 文件不存在，正在从模板创建..."
    cp "$SRC_DIR/.env.example" "$SRC_DIR/.env"
    echo -e "  ${C_YELLOW}→${C_RESET} 已创建 $SRC_DIR/.env"

    echo ""
    echo -e "  ${C_BOLD}============================================${C_RESET}"
    echo -e "  ${C_YELLOW}  请编辑 .env 文件，填写以下必填项：${C_RESET}"
    echo ""
    echo "     1. MySQL 连接信息（SAG_MYSQL_*）"
    echo "     2. LLM API Key 及地址（SAG_LLM_*）"
    echo "     3. Embedding 模型路径（SAG_EMBEDDING_MODEL_PATH）"
    echo ""
    echo "   文件位置: $SRC_DIR/.env"
    echo -e "  ${C_BOLD}============================================${C_RESET}"
    echo ""

    if [ -n "${EDITOR:-}" ]; then
        read -rp "  是否现在编辑 .env 文件？[Y/n] " yn
        if [ "$yn" != "n" ] && [ "$yn" != "N" ]; then
            $EDITOR "$SRC_DIR/.env"
        fi
    fi
    echo ""
    echo -e "  ${C_YELLOW}  .env 编辑完成后，请重新运行 ./setup.sh 启动服务。${C_RESET}"
    exit 0
fi

# ---- 仅安装模式 ----
if [ "$ACTION" = "install" ]; then
    step "依赖安装完成，可以编辑 .env 后运行 ./setup.sh start 启动服务"
    exit 0
fi

# ---- 预检 ----
check_env

# ---- 必要目录 ----
mkdir -p "$SRC_DIR/logs"

# ---- 清理旧进程的函数 ----
cleanup() {
    echo ""
    echo -e "  ${C_YELLOW}正在停止服务...${C_RESET}"
    [ -n "${API_PID:-}" ] && kill "$API_PID" 2>/dev/null || true
    [ -n "${WEB_PID:-}" ] && kill "$WEB_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    echo -e "  ${C_GREEN}✓${C_RESET} 服务已停止"
}
trap cleanup EXIT INT TERM

# ---- 启动服务 ----
echo ""
title "启动 ai_sag 服务"

echo -e "  API 服务:   http://localhost:$API_PORT"
echo -e "  API 文档:   http://localhost:$API_PORT/docs"
echo -e "  Web UI:     http://localhost:$WEB_PORT"
echo ""
echo -e "  ${C_YELLOW}按 Ctrl+C 停止所有服务${C_RESET}"
echo ""

# 后台启动 API
cd "$ROOT_DIR"
python -m ai_sag.api --host 0.0.0.0 --port "$API_PORT" &
API_PID=$!
echo -e "  ${C_GREEN}✓${C_RESET} API 服务已启动 (PID: $API_PID)"

# 后台启动 Web UI
python -m ai_sag.web --host 0.0.0.0 --port "$WEB_PORT" --api "http://localhost:$API_PORT" &
WEB_PID=$!
echo -e "  ${C_GREEN}✓${C_RESET} Web UI 已启动 (PID: $WEB_PID)"

echo ""
echo -e "  ${C_GREEN}浏览器打开 http://localhost:$WEB_PORT 即可使用${C_RESET}"

# 等待任意子进程退出
wait