import sys
from pathlib import Path
from uuid import uuid4

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def _default_trace_log_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "langbot.tracing.default_log_path",
        lambda: str(tmp_path / f"default-trace-{uuid4().hex}.log"),
    )
