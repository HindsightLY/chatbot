"""
主应用模块
提供API服务和命令行界面
"""
import uvicorn
import argparse
from fastapi import FastAPI
from src.logger_config import logger
from config.app_config import APP_CONFIG
from src.api.routers.cli_router import run_cli
from src.api.routers.chat_router import router as chat_router
from src.service.system_initializer import system_initializer

# 确保数据目录存在
APP_CONFIG.ensure_data_dirs()

# 创建FastAPI应用实例
app = FastAPI(
    title="医疗咨询AI API",
    description="基于RAG的医疗问答接口"
)

# 注册API路由
app.include_router(chat_router)


@app.on_event("startup")
def startup_event():
    """
    FastAPI启动时自动运行
    """
    logger.info("🚀 正在初始化医疗AI系统...")
    system_initializer.initialize_system()
    logger.info("🎉 系统初始化完成，API 就绪！")
    logger.info(f"📁 向量存储位置: {APP_CONFIG.vector_persist_dir}")
    logger.info(f"📁 文档数据位置: {APP_CONFIG.disease_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="医疗咨询AI启动器")
    parser.add_argument("--api", action="store_true", help="以 API 模式启动 (FastAPI)")
    parser.add_argument("--host", default=APP_CONFIG.api_host, help="API 监听地址")
    parser.add_argument("--port", type=int, default=APP_CONFIG.api_port, help="API 监听端口")

    args = parser.parse_args(['--api'])

    if args.api:
        # 启动 API 模式
        uvicorn.run(app, host=args.host, port=args.port, reload=False)
    else:
        # 启动 CLI 模式 (默认)
        run_cli()
