@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

:: ============================================================
:: ai_sag 一键环境搭建 & 启动脚本 (Windows)
::
:: 用法：
::   setup.bat                   默认 CPU 模式，全部安装 + 启动
::   setup.bat gpu               GPU 模式
::   setup.bat check             仅检查环境
::   setup.bat install           仅安装依赖（不启动）
::   setup.bat start             仅启动服务（不安装）
::   setup.bat help              显示帮助
::
:: 首次使用会引导你配置 .env 文件。
:: ============================================================

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
for %%i in ("%SCRIPT_DIR%") do set "SCRIPT_DIR=%%~fi"

:: 仓库根目录 = scripts/ 的上级目录
for %%i in ("%SCRIPT_DIR%\..") do set "ROOT_DIR=%%~fi"

set "ENV_DIR=%ROOT_DIR%\.venv"
set "SRC_DIR=%ROOT_DIR%\ai_sag"
set "API_PORT=8777"
set "WEB_PORT=8080"
set "MODE=cpu"

:: ============================================================
:: 颜色定义
:: ============================================================
for /f %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"
set "C_RESET=%ESC%[0m"
set "C_RED=%ESC%[91m"
set "C_GREEN=%ESC%[92m"
set "C_YELLOW=%ESC%[93m"
set "C_BLUE=%ESC%[94m"
set "C_CYAN=%ESC%[96m"
set "C_BOLD=%ESC%[1m"

:: 检查是否支持 ANSI（Windows 10 1703+）
reg query "HKEY_CURRENT_USER\Console" /v VirtualTerminalLevel >nul 2>&1
if %errorlevel% neq 0 (
    set "C_RESET=" & set "C_RED=" & set "C_GREEN=" & set "C_YELLOW="
    set "C_BLUE=" & set "C_CYAN=" & set "C_BOLD="
)

:: ============================================================
:: 解析参数
:: ============================================================
if /i "%~1"==""    set "ACTION=all"  & goto :parse_done
if /i "%~1"=="gpu" set "MODE=gpu"    & set "ACTION=all"  & goto :parse_done
if /i "%~1"=="check"  set "ACTION=check"  & goto :parse_done
if /i "%~1"=="install" set "ACTION=install" & goto :parse_done
if /i "%~1"=="start"  set "ACTION=start"  & goto :parse_done
if /i "%~1"=="help"   set "ACTION=help"   & goto :parse_done
if /i "%~1"=="-h"     set "ACTION=help"   & goto :parse_done
if /i "%~1"=="--help" set "ACTION=help"   & goto :parse_done
echo %C_RED%[ERROR]%C_RESET% 未知参数: %~1（可用: gpu / check / install / start / help）
exit /b 1

:parse_done
call :title "ai_sag 环境搭建脚本"
echo   仓库根目录: %ROOT_DIR%
echo   Python 包:  %SRC_DIR%
echo   虚拟环境:  %ENV_DIR%
if /i "%MODE%"=="gpu" (
    echo   运行模式:  %C_CYAN%GPU (CUDA 12.4)%C_RESET%
) else (
    echo   运行模式:  CPU
)
echo.

:: ============================================================
:: HELP
:: ============================================================
if /i "%ACTION%"=="help" (
    echo 用法: setup.bat [选项]
    echo.
    echo 选项：
    echo   （无参数）   CPU 模式，安装依赖 + 启动服务
    echo   gpu         GPU 模式，安装 CUDA 依赖 + 启动服务
    echo   check       仅检查 Python / MySQL / .env 环境
    echo   install     仅安装依赖，不启动服务
    echo   start       仅启动 API + Web UI
    echo   help        显示此帮助
    echo.
    echo 首次使用：
    echo   1. 准备 .env 文件（会从 .env.example 复制模板）
    echo   2. 编辑 .env 填写 MySQL / LLM API Key / Embedding 路径
    echo   3. 运行 setup.bat
    echo.
    echo 更多帮助：docs\STARTUP.md
    exit /b 0
)

:: ============================================================
:: 检查 Python
:: ============================================================
call :step "检查 Python 环境..."
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo %C_RED%[ERROR]%C_RESET% 未找到 python，请先安装 Python 3.10+
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
for /f "tokens=1-2 delims=." %%a in ("%PY_VER%") do (
    set "PY_MAJOR=%%a" & set "PY_MINOR=%%b"
)
if %PY_MAJOR% lss 3 (
    echo %C_RED%[ERROR]%C_RESET% Python 版本过低: %PY_VER%，需要 3.10+
    exit /b 1
)
if %PY_MAJOR% equ 3 if %PY_MINOR% lss 10 (
    echo %C_RED%[ERROR]%C_RESET% Python 版本过低: %PY_VER%，需要 3.10+
    exit /b 1
)
echo   %C_GREEN%✓%C_RESET% Python %PY_VER%

:: ============================================================
:: 仅检查环境
:: ============================================================
if /i "%ACTION%"=="check" (
    call :check_env
    exit /b 0
)

:: ============================================================
:: 创建虚拟环境
:: ============================================================
if not exist "%ENV_DIR%\Scripts\python.exe" (
    call :step "创建虚拟环境..."
    python -m venv "%ENV_DIR%"
    if %errorlevel% neq 0 (
        echo %C_RED%[ERROR]%C_RESET% 创建虚拟环境失败
        exit /b 1
    )
    echo   %C_GREEN%✓%C_RESET% 虚拟环境已创建: %ENV_DIR%
) else (
    call :step "虚拟环境已存在，跳过创建..."
    echo   %C_YELLOW%→%C_RESET% %ENV_DIR%
)

:: 激活虚拟环境
call "%ENV_DIR%\Scripts\activate.bat"

:: 升级 pip
call :step "升级 pip..."
python -m pip install --upgrade pip -q
echo   %C_GREEN%✓%C_RESET% pip 已升级

:: ============================================================
:: 安装依赖
:: ============================================================
if /i "%ACTION%"=="start" goto :skip_install

call :step "安装 Python 依赖（%MODE% 模式）..."
if /i "%MODE%"=="gpu" (
    :: GPU 模式：requirements-gpu.txt 已自包含全部依赖，直接安装即可，不要叠加 CPU 版
    pip install -r "%SRC_DIR%\requirements-gpu.txt" -q
    if %errorlevel% neq 0 (
        echo %C_RED%[ERROR]%C_RESET% GPU 依赖安装失败，请手动检查
        exit /b 1
    )
    echo   %C_GREEN%✓%C_RESET% GPU 依赖安装完成
) else (
    pip install -r "%SRC_DIR%\requirements.txt" -q
    if %errorlevel% neq 0 (
        echo %C_RED%[ERROR]%C_RESET% 基础依赖安装失败
        exit /b 1
    )
    echo   %C_GREEN%✓%C_RESET% 基础依赖安装完成
)

:skip_install

:: ============================================================
:: .env 检查 & 引导
:: ============================================================
if not exist "%SRC_DIR%\.env" (
    call :step ".env 文件不存在，正在从模板创建..."
    copy "%SRC_DIR%\.env.example" "%SRC_DIR%\.env" >nul
    echo   %C_YELLOW%→%C_RESET% 已创建 %SRC_DIR%\.env

    echo.
    echo   %C_BOLD%============================================%C_RESET%
    echo   %C_YELLOW%  请编辑 .env 文件，填写以下必填项：%C_RESET%
    echo.
    echo     1. MySQL 连接信息（SAG_MYSQL_*）
    echo     2. LLM API Key 及地址（SAG_LLM_*）
    echo     3. Embedding 模型路径（SAG_BGE_MODEL_PATH）
    echo.
    echo   文件位置: %SRC_DIR%\.env
    echo   %C_BOLD%============================================%C_RESET%

    echo.
    choice /c YN /m "是否现在打开 .env 文件进行编辑"
    if !errorlevel! equ 1 (
        start notepad "%SRC_DIR%\.env"
    )
    echo.
    echo %C_YELLOW%   .env 编辑完成后，请重新运行 setup.bat 启动服务。%C_RESET%
    exit /b 0
)

:: ============================================================
:: 仅安装模式
:: ============================================================
if /i "%ACTION%"=="install" (
    call :step "依赖安装完成，可以编辑 .env 后运行 setup.bat start 启动服务"
    exit /b 0
)

:: ============================================================
:: 预检环境变量
:: ============================================================
call :check_env

:: ============================================================
:: 检查并创建必要目录
:: ============================================================
if not exist "%SRC_DIR%\logs" mkdir "%SRC_DIR%\logs" >nul

:: ============================================================
:: 启动服务
:: ============================================================
echo.
call :title "启动 ai_sag 服务"

echo   API 服务:   http://localhost:%API_PORT%
echo   API 文档:   http://localhost:%API_PORT%/docs
echo   Web UI:     http://localhost:%WEB_PORT%
echo.
echo   %C_YELLOW%按 Ctrl+C 可分别停止每个窗口%RESET%
echo.

:: 方式：开两个新的 cmd 窗口，各跑一个服务
:: API 窗口
start "ai_sag API (port %API_PORT%)" cmd /c ^
    "cd /d "%ROOT_DIR%" && ^
     call "%ENV_DIR%\Scripts\activate.bat" && ^
     echo [%date% %time%] ai_sag API 启动中... && ^
     python -m ai_sag.api --host 0.0.0.0 --port %API_PORT% && ^
     pause"

:: Web UI 窗口
start "ai_sag Web UI (port %WEB_PORT%)" cmd /c ^
    "cd /d "%ROOT_DIR%" && ^
     call "%ENV_DIR%\Scripts\activate.bat" && ^
     echo [%date% %time%] ai_sag Web UI 启动中... && ^
     python -m ai_sag.web --host 0.0.0.0 --port %WEB_PORT% --api http://localhost:%API_PORT% && ^
     pause"

echo   %C_GREEN%✓%C_RESET% 两个服务已在独立窗口中启动
echo   %C_GREEN%✓%C_RESET% 浏览器打开 http://localhost:%WEB_PORT% 即可使用
echo.
exit /b 0

:: ============================================================
:: 子程序：环境检查
:: ============================================================
:check_env
call :step "检查环境变量..."
python -c "from dotenv import load_dotenv, find_dotenv; import os; p=find_dotenv(); print(p or 'NOT_FOUND')" >"%TEMP%\ai_sag_env.txt" 2>&1
set /p ENV_PATH=<"%TEMP%\ai_sag_env.txt"
del "%TEMP%\ai_sag_env.txt" >nul 2>&1

if "%ENV_PATH%"=="NOT_FOUND" (
    echo   %C_YELLOW%⚠%C_RESET% 未找到 .env，首次启动将引导配置
) else (
    echo   %C_GREEN%✓%C_RESET% .env 已找到
)

:: 检查 MySQL 连接
python -c "import os; v=['SAG_MYSQL_HOST','SAG_MYSQL_USER','SAG_MYSQL_PASSWORD','SAG_LLM_API_KEY','SAG_BGE_MODEL_PATH']; m=[k for k in v if not os.environ.get(k)]; exit(1 if m else 0)" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo   %C_YELLOW%⚠%C_RESET% 部分配置可能未填写，请检查 .env：
    python -c "import os; v={'SAG_MYSQL_HOST':'MySQL地址','SAG_MYSQL_USER':'MySQL用户','SAG_MYSQL_PASSWORD':'MySQL密码','SAG_LLM_API_KEY':'LLM API Key','SAG_BGE_MODEL_PATH':'Embedding模型路径'}; [print(f'     - {d}: 未设置') for k,d in v.items() if not os.environ.get(k)]"
    echo.
)
exit /b 0

:: ============================================================
:: 子程序：显示标题
:: ============================================================
:title
echo.
echo   %C_BOLD%%C_BLUE%══════════════════════════════════════════════════%C_RESET%
echo   %C_BOLD%%C_BLUE%  %~1%C_RESET%
echo   %C_BOLD%%C_BLUE%══════════════════════════════════════════════════%C_RESET%
exit /b 0

:: ============================================================
:: 子程序：显示步骤
:: ============================================================
:step
echo   %C_CYAN%[%~1]%C_RESET%
exit /b 0