from __future__ import annotations

import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path

from streamlit.web import cli as stcli


def find_free_port(start: int = 8501, end: int = 8600) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("Uygun boş port bulunamadı.")


def app_path() -> str:
    if getattr(sys, "frozen", False):
        base_dir = Path(sys._MEIPASS)
        return str(base_dir / "app.py")
    return str(Path(__file__).with_name("app.py"))


def main() -> None:
    port = find_free_port()
    url = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ.update(env)

    threading.Timer(2.0, lambda: webbrowser.open(url)).start()

    sys.argv = [
        "streamlit",
        "run",
        app_path(),
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--browser.gatherUsageStats=false",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
