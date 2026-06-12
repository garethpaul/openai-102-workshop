#!/usr/bin/env python3
"""Launch the workshop and require a healthy local Streamlit endpoint."""

import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_SECONDS = 20


def available_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def main():
    port = available_port()
    environment = os.environ.copy()
    environment.pop("OPENAI_API_KEY", None)
    environment["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "👋_Hello.py",
            "--server.headless=true",
            "--server.address=127.0.0.1",
            f"--server.port={port}",
        ],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    deadline = time.monotonic() + TIMEOUT_SECONDS
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = process.stdout.read()
                raise RuntimeError(f"Streamlit exited before becoming healthy:\n{output}")
            try:
                with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
                    if response.status == 200 and response.read().strip() == b"ok":
                        print("Streamlit health smoke passed.")
                        return
            except OSError:
                time.sleep(0.25)
        raise RuntimeError(f"Streamlit did not become healthy within {TIMEOUT_SECONDS} seconds")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


if __name__ == "__main__":
    main()
