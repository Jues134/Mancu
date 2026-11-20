"""Launcher: set up logging, *then* start uvicorn.

Why this exists instead of `python -m uvicorn app.main:app` directly:

`main.py` installs the logging handlers on import, which is fine until the import
itself is what fails — a missing dependency, a syntax error in a file being edited,
a broken config at module scope. At that moment `main.py` has not executed a
single line, so nothing has been written to `error.log`, and the only copy of the
reason is a line in a console window that a launcher may close on failure.

Starting through this module means the handlers exist before the application is
imported, so uvicorn's own "Error loading ASGI app" line — which carries the
traceback — lands in `error.log` too.

It is also the only place that knows the port before the app starts, so the
browser opened by `main.py` and the port actually being served cannot disagree.
"""

from __future__ import annotations

import argparse
import os
import sys

from .logs import get_logger, setup


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.serve", description="启动漫簇后端")
    parser.add_argument("--host", default=os.environ.get("STUDIO_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("STUDIO_PORT") or 8100)
    )
    parser.add_argument("--log-level", default=None, help="DEBUG / INFO / WARNING")
    args = parser.parse_args(argv)

    # Export before the app is imported: `main.py` reads this for the browser URL,
    # and a mismatch would open a tab on the wrong port.
    os.environ["STUDIO_PORT"] = str(args.port)

    setup(args.log_level)
    logger = get_logger("mancu.serve")
    logger.info("启动参数 host=%s port=%s pid=%s", args.host, args.port, os.getpid())

    import uvicorn

    try:
        # `log_config=None` keeps uvicorn from installing its own handlers: with
        # them, its output goes to stderr only and never reaches the log file, and
        # it sets `propagate = False` on its loggers so ours never see it either.
        uvicorn.run("app.main:app", host=args.host, port=args.port, log_config=None)
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在退出")
        return 0
    except Exception:
        # The most useful line in this file: whatever stopped the server from
        # starting, with a stack, written where the user can find it.
        logger.exception("服务启动失败")
        return 1
    finally:
        logger.info("服务已退出")
    return 0


if __name__ == "__main__":
    sys.exit(main())
