#!/usr/bin/env python3
"""Serve the precomputed static demo locally; no model or dataset is needed."""

from __future__ import annotations

import argparse
import errno
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    required = (WEB_DIR / "index.html", WEB_DIR / "demo_bundle.json")
    missing = [path for path in required if not path.exists()]
    if missing:
        parser.error("Missing static demo files: " + ", ".join(str(path) for path in missing))

    handler = partial(SimpleHTTPRequestHandler, directory=str(WEB_DIR))
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            parser.error(
                f"Port {args.port} is already in use. If FlowWatch is already running, open "
                f"http://127.0.0.1:{args.port}; otherwise try --port {args.port + 1}."
            )
        raise

    print(f"FlowWatch demo is ready at http://127.0.0.1:{args.port}")
    print("The demo uses precomputed examples and does not capture live traffic.")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping FlowWatch demo.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
