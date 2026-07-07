"""Web 模块：提供单页 Web UI，通过 fetch 调用 ai_sag API 进行交互式测试。

启动：
    python -m ai_sag.web                         # 默认 0.0.0.0:8080
    python -m ai_sag.web --port 8080 --api http://localhost:8777

页面能力：
    - 文档上传（文件/纯文本）
    - 文档列表、详情、下载、删除、更新元信息
    - 文档全文查询（关键词命中+上下文）
    - SAG 检索（返回切片+trace）
    - 问答（答案+切片+trace）
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

_WEB_DIR = Path(__file__).resolve().parent

# 默认 API 地址：与 api.py 默认端口 8777 对齐
_DEFAULT_API = os.environ.get("AISAG_WEB_API_BASE", "http://localhost:8777")


def create_web_app(api_base: str = _DEFAULT_API) -> FastAPI:
    """创建 Web UI 应用实例。

    该工厂函数负责把静态前端页面挂载到 FastAPI，并将后端 API 地址
    注入到前端，使前端通过 fetch 直接调用 ai_sag 的 REST 接口，
    实现「Web 页面 ↔ API ↔ 业务内核」三层解耦。

    Args:
        api_base: 后端 API 基址（如 http://localhost:8777），
                  默认取环境变量 AISAG_WEB_API_BASE，便于部署时灵活切换。

    Returns:
        FastAPI 应用实例，仅暴露两个路由：
          - GET /        返回单页 HTML（API 基址已注入）
          - GET /health  健康检查，回显当前 api_base
    """
    # 关闭 docs/redoc，避免暴露接口文档（Web 模块只托管前端）
    app = FastAPI(title="ai_sag Web UI", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        # 读取打包在 static/ 下的单页 HTML（原生 JS，无构建依赖）
        html_path = _WEB_DIR / "static" / "index.html"
        html = html_path.read_text(encoding="utf-8")
        # 把 API 基址注入前端，避免硬编码：前端 JS 用 __API_BASE__ 占位
        return html.replace("__API_BASE__", api_base)

    @app.get("/health")
    async def health():
        # 健康检查同时回显 api_base，便于排查「前端指向错后端」的问题
        return {"status": "ok", "api_base": api_base}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="ai_sag Web UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--api", default=_DEFAULT_API, help="后端 API 地址")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    import uvicorn

    # 用环境变量把 api_base 传给 create_web_app（reload 模式下需通过 env）
    os.environ["AISAG_WEB_API_BASE"] = args.api
    uvicorn.run(
        "ai_sag.web:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=False,
    )


# 模块级 app（uvicorn 直接引用 ai_sag.web:app 时使用）
app = create_web_app(os.environ.get("AISAG_WEB_API_BASE", _DEFAULT_API))


if __name__ == "__main__":
    main()