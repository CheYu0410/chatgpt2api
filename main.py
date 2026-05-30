from __future__ import annotations

import uvicorn
from api import create_app

app = create_app()

if __name__ == "__main__":
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        access_log=False,
        log_level="info",
    )
    server = uvicorn.Server(config)
    server.run()
