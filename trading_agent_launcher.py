
"""
PyInstaller launcher for the local Streamlit trading-agent demo.

Do not run Streamlit through a normal command after packaging.
Run TradingAgentDemo.exe and it will start Streamlit locally.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


def find_free_port(start: int = 8501, end: int = 8599) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start


def get_bundle_app_path() -> Path:
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    else:
        base = Path(__file__).resolve().parent

    app_path = base / "app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"Could not find bundled Streamlit app: {app_path}")

    return app_path


def open_browser_later(port: int) -> None:
    time.sleep(2.0)
    webbrowser.open(f"http://localhost:{port}")


def main() -> None:
    app_path = get_bundle_app_path()
    port = find_free_port()

    # Avoid Streamlit asking to collect usage stats.
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    # Force Streamlit production mode in PyInstaller.
    # Without this, bundled Streamlit can think global.developmentMode=true,
    # then it rejects server.port with:
    # "server.port does not work when global.developmentMode is true."
    os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENTMODE", "false")

    # Make PyTorch/Streamlit slightly quieter.
    os.environ.setdefault("PYTHONWARNINGS", "ignore")

    print("Starting Trading Agent Demo...")
    print(f"App file : {app_path}")
    print(f"Local URL: http://localhost:{port}")
    print("")
    print("Keep this window open while using the app.")
    print("Close this window to stop the local server.")

    threading.Thread(target=open_browser_later, args=(port,), daemon=True).start()

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--global.developmentMode=false",
        f"--server.port={port}",
        "--server.address=127.0.0.1",
        "--browser.gatherUsageStats=false",
        "--server.headless=true",
    ]

    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()
