"""
主应用入口 — 医疗咨询 AI 系统

运行模式:
  python main.py --api   → FastAPI 服务（默认，自动打开浏览器）
  python main.py         → CLI 交互界面

启动流程:
  1. 设置 HuggingFace 国内镜像 + 强制离线（环境变量优先于模型下载）
  2. 确保 data/disease/chroma_db 目录存在
  3. API 模式: FastAPI + uvicorn → startup 事件初始化系统 → 浏览器自动打开
  4. CLI 模式: run_cli() 直接触发初始化
"""
import sys
import os

# 强制 HuggingFace 从镜像加载并使用本地缓存（不联网）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
import asyncio
import webbrowser
import threading
import uvicorn
import argparse
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.logger_config import logger
from config.app_config import APP_CONFIG
from src.api.routers.cli_router import run_cli
from src.api.routers.chat_router import router as chat_router
from src.service.system_initializer import system_initializer

APP_CONFIG.ensure_data_dirs()

app = FastAPI(
    title="医疗咨询AI API",
    description="基于RAG的医疗问答接口"
)

app.include_router(chat_router)

static_dir = Path(__file__).parent / "static"
if static_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


@app.on_event("startup")
def startup_event():
    """
    FastAPI 启动事件 — 初始化系统全部组件。

    容错设计:
      - system_initializer.initialize_system() 内部每个组件独立 try/except
      - 单组件失败不影响其他组件，API 以降级模式运行
      - 不影响 FastAPI 服务进程本身
    """
    logger.info("🚀 正在初始化医疗AI系统...")
    try:
        system_initializer.initialize_system()
    except Exception as e:
        logger.error(f"❌ 系统初始化异常: {e}")
        logger.warning("⚠️ 部分组件可能未就绪，API 将以降级模式运行")

    if system_initializer.init_errors:
        logger.warning("⚠️ 以下组件初始化失败，功能可能受限:")
        for err in system_initializer.init_errors:
            logger.warning(f"  - {err}")
    else:
        logger.info("🎉 系统初始化完成，API 就绪！")

    logger.info(f"📁 ChromaDB 存储位置: {APP_CONFIG.chroma_persist_dir}")
    logger.info(f"📁 文档数据位置: {APP_CONFIG.disease_dir}")


def open_browser(host: str, port: int):
    """延迟 1.5s 后自动打开浏览器访问 API 地址"""
    import time
    time.sleep(1.5)
    url = f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}"
    webbrowser.open(url)
    logger.info(f"🌐 浏览器已打开: {url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="医疗咨询AI启动器")
    parser.add_argument("--api", action="store_true", help="以 API 模式启动 (FastAPI)")
    parser.add_argument("--host", default=APP_CONFIG.api_host, help="API 监听地址")
    parser.add_argument("--port", type=int, default=APP_CONFIG.api_port, help="API 监听端口")

    args = parser.parse_args(['--api'])

    if args.api:
        threading.Thread(target=open_browser, args=(args.host, args.port), daemon=True).start()
        config = uvicorn.Config(app, host=args.host, port=args.port, lifespan="on")
        server = uvicorn.Server(config)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(server.serve())
        except KeyboardInterrupt:
            pass
        finally:
            loop.close()
    else:
        run_cli()
