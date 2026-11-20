"""Turn a locally saved render into a URL the page can load.

Generated images and clips are written under `<DATA_DIR>/assets/<run_id>/<kind>/`,
and `/api/doc/asset/{path}` serves anything under `<DATA_DIR>` without
authentication, keyed by the path's position under it.

This exists because both renderers originally recorded the *provider's* link. For a
hosted model that link is a public CDN URL and works; for the local gateway it needs
the API key in a request header, and a browser `<img src>` cannot send one — the
render came back 401 and the page showed an empty frame for a file that was sitting
on disk. Preferring the local copy fixes that and keeps history working after the
gateway restarts or the provider's link expires.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from ..config import DATA_DIR


def served_url(local: str | Path) -> str:
    """`/api/doc/asset/...` for a file under the data directory, else ""."""
    if not local:
        return ""
    path = Path(local).resolve()
    try:
        rel = path.relative_to(DATA_DIR.resolve())
    except ValueError:
        # Outside the data directory (a test's temp dir, an operator's own path):
        # the caller keeps the provider's link rather than inventing a route.
        return ""
    return "/api/doc/asset/" + "/".join(quote(part) for part in rel.parts)
