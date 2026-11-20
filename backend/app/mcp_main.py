"""MCP entry point for the document pipeline's read-only tools.

Usage from an MCP client (Claude Desktop / Cursor / any MCP host):

    {
      "mcpServers": {
        "studio-tools": {
          "command": "python",
          "args": ["-m", "app.mcp_main"],
          "cwd": "<project>/backend"
        }
      }
    }

The server exposes the same knowledge and connected-document tools used by the
active pipeline.
"""

from __future__ import annotations

import asyncio
import sys

from .platform.mcp_tools import build_outbound_server


def main() -> int:
    server = build_outbound_server()
    try:
        asyncio.run(server.serve_stdio())
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
