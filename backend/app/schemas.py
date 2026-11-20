"""Pydantic data contracts shared across the active document pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    NEEDS_INPUT = "needs_input"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class ArtifactKind(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    JSON = "json"


class Verdict(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


# ============================================================================
# Run / Step / Artifact records
# ============================================================================


class AgentResult(BaseModel):
    """What every agent returns to the orchestrator."""

    agent: str
    ok: bool = True
    summary: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    cost_usd: float = 0.0
    duration_ms: int = 0
    error: str = ""


class StepRecord(BaseModel):
    id: str
    run_id: str
    seq: int
    agent: str
    status: StepStatus = StepStatus.PENDING
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    cost_usd: float = 0.0
    started_at: datetime | None = None
    finished_at: datetime | None = None


class ArtifactRecord(BaseModel):
    id: str
    run_id: str
    step_id: str = ""
    kind: ArtifactKind = ArtifactKind.JSON
    label: str = ""
    path: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)
