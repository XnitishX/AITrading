"""
AITrading Web App Launcher
──────────────────────────
Start the FastAPI web application with:
    python run_web.py
Then open http://localhost:8000 in your browser.
"""

import sys
import logging
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import LOG_FORMAT, LOG_LEVEL

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)


def main():
    import uvicorn
    print("=" * 60)
    print("  AITrading – Web Application")
    print("  Open http://localhost:8000 in your browser")
    print("=" * 60)
    uvicorn.run(
        "src.web.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
