from __future__ import annotations

import uvicorn
from api import create_app

app = create_app()

if __name__ == "__main__":
    import socket
    import contextlib
    # Pre-bind and release port to clear any TIME_WAIT state
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", 8000))
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, access_log=False, log_level="info")
    server = uvicorn.Server(config)
    server.run()
