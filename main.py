"""
main.py — Application Entry Point

Run modes:
  python main.py          → Gradio UI (default, for demo)
  python main.py --api    → FastAPI server (for production)
  python main.py --both   → Both simultaneously
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from erp42_analytics.core.config import config

logging.basicConfig(
    level=logging.DEBUG if config.app.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            str(Path(__file__).parent / "logs" / "app.log"),
            encoding="utf-8",
        ),
    ]
)

Path("logs").mkdir(exist_ok=True)
logger = logging.getLogger(__name__)


def run_gradio():
    """Run Gradio UI — for demo and single-user mode."""
    from erp42_analytics.ui.gradio_app import create_app
    logger.info("Starting Gradio UI...")
    demo = create_app()
    demo.launch(
        server_port=config.app.gradio_port,
        share=config.app.gradio_share,
    )


def run_api():
    """Run FastAPI server — for production multi-user mode."""
    import uvicorn
    logger.info(
        f"Starting FastAPI server on "
        f"{config.app.api_host}:{config.app.api_port}"
    )
    uvicorn.run(
        "erp42_analytics.api.main:app",
        host=config.app.api_host,
        port=config.app.api_port,
        reload=config.app.debug,
        log_level="debug" if config.app.debug else "info",
    )


def run_both():
    """Run both Gradio + FastAPI simultaneously."""
    import threading
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    run_gradio()  # Gradio blocks main thread


def main():
    parser = argparse.ArgumentParser(
        description=f"{config.app.app_name} v{config.app.app_version}"
    )
    parser.add_argument(
        "--mode",
        choices=["gradio", "api", "both"],
        default="gradio",
        help="Run mode: gradio (demo), api (production), both",
    )
    args = parser.parse_args()

    logger.info(
        f"Starting {config.app.app_name} "
        f"v{config.app.app_version} in {args.mode} mode"
    )

    if args.mode == "api":
        run_api()
    elif args.mode == "both":
        run_both()
    else:
        run_gradio()


if __name__ == "__main__":
    main()