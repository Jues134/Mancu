"""Test-wide setup.

One job: keep the test suite out of the user's real log directory.

`app.logs` resolves its directory once, at import, so anything imported afterwards
writes there — and a test that drives `TestClient` through a route logs real
requests. Without this, running `pytest` mixes `http://testserver/...` lines into
the log a user opens to diagnose their own run, and the file they send for help
starts with someone else's unit tests.
"""

import os
import tempfile

os.environ.setdefault(
    "STUDIO_LOG_DIR", os.path.join(tempfile.gettempdir(), "mancu-test-logs")
)
