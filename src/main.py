"""
主应用入口

两种运行模式:
  python main.py --api   → FastAPI 服务 (默认，自动打开浏览器)
  python main.py         → CLI 交互界面

启动流程:
  1. 确保数据目录存在 (ensure_data_dirs)
  2. 创建 FastAPI 实例并注册路由、挂载静态文件
  3. API 模式: uvicorn 启动 → startup 事件初始化系统组件 → 自动打开页面
  4. CLI 模式: run_cli() 直接触发初始化
"""
import webbrowser
import threading
import uvicorn
import argparse
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
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
    """FastAPI 启动时初始化系统组件（容错：组件失败不影响服务启动）"""
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
    """延迟打开浏览器，等待服务器就绪"""
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
        uvicorn.run(app, host=args.host, port=args.port, reload=False)
    else:
        run_cli()
