#!/bin/bash
# Start chatgpt2api - fully daemonized
PORT=8000

# Kill any existing process on the port
fuser -k ${PORT}/tcp 2>/dev/null
sleep 2

# Double-fork to fully detach from parent
(
    # First fork
    (
        # Second fork - completely detached
        cd /opt/data/chatgpt2api
        source .venv/bin/activate
        exec python main.py
    ) &
) &
