"""Agent registry — the platform's "Agent 定义与配置" module.

Competition requirement this satisfies: agents must be definable and configurable
(role, skills, tools, memory), with dynamic registration and version management.

The design decision that matters: an agent is a *declared capability*, not a
hard-coded class. A declaration carries what the agent can do (skills), what it
may call (tools), what it remembers (memory scope), and which workflow stage it
belongs to. The executor then dispatches purely from declarations, which is what
makes the platform general instead of a pipeline with names.

Versioning is deliberate: an agent definition is immutable once registered, and
a new revision gets a new version id. A run records the exact version ids it
used, so a rerun reproduces the same behaviour even after agents are edited
(the alternative — mutating definitions in place — makes results unreproducible,
which is fatal for a competition that compares accuracy across configurations).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable


# ============================================================================
# Declaration model
# ============================================================================


@dataclass(frozen=True)
class AgentSkill:
    """A named capability with an explicit contract.

    `input_schema` / `output_schema` are documentation-plus-validation strings,
    not JSON Schema: they state which keys the agent reads and writes. Declaring
    them turns "agent A hands off to agent B" from a convention into a checkable
    contract, and lets the workflow validator reject an incompatible edge before
    anything runs.
    """

    name: str
    description: str
    input_keys: tuple[str, ...] = ()
    output_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentToolGrant:
    """A tool an agent is allowed to call, and the quota it gets.

    Tools are granted, never global: the verifier can read search tools but must
    not be able to call a "write script" tool, because an actor that both
    produces and edits its own output cannot independently verify it.
    """

    tool: str
    max_calls_per_run: int = 8


@dataclass(frozen=True)
class MemoryScope:
    """What an agent carries between steps.

    `none`    — stateless; every call gets only its inputs.
    `run`     — accumulates facts across the run (used by the narrative agent).
    `global`  — reads project-level memory (not used yet; declared for extension).
    """

    kind: str = "none"
    max_items: int = 20


@dataclass(frozen=True)
class AgentDefinition:
    """One immutable version of one agent."""

    id: str
    version: str
    label: str
    role: str
    stage: str
    skills: tuple[AgentSkill, ...] = ()
    tools: tuple[AgentToolGrant, ...] = ()
    memory: MemoryScope = field(default_factory=MemoryScope)
    #: Independent reviewers must not share a model with producers. The scheduler
    #: enforces this when it resolves model bindings.
    independence_group: str = ""
    #: Human-facing note shown in the UI trace.
    notes: str = ""


# ============================================================================
# Registry
# ============================================================================


class AgentRegistry:
    """Holds the live catalog of agent definitions.

    Registration is append-only per agent id: registering a new version keeps the
    previous one addressable by version, so a run started before an edit still
    resolves to the definition it began with.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, list[AgentDefinition]] = {}
        self._handlers: dict[tuple[str, str], Callable[..., Any]] = {}

    # ---------------------------------------------------------------- register

    def register(
        self,
        definition: AgentDefinition,
        handler: Callable[..., Any],
        *,
        replace: bool = False,
    ) -> None:
        """Attach an implementation to a definition.

        `replace=False` refuses to overwrite an existing (id, version): silent
        replacement would make two runs claim the same version while behaving
        differently.
        """
        key = (definition.id, definition.version)
        if key in self._handlers and not replace:
            raise ValueError(
                f"Agent {definition.id}@{definition.version} 已注册。"
                "如需改动请使用新版本号（同一版本号对应不同实现会让结果无法复现）。"
            )
        versions = self._by_id.setdefault(definition.id, [])
        if not any(v.version == definition.version for v in versions):
            versions.append(definition)
            # Newest first: `resolve` without a version picks the latest.
            versions.sort(key=lambda v: _version_key(v.version), reverse=True)
        self._handlers[key] = handler

    def resolve(self, agent_id: str, version: str | None = None) -> tuple[AgentDefinition, Callable[..., Any]]:
        versions = self._by_id.get(agent_id)
        if not versions:
            raise KeyError(f"未注册的 Agent：{agent_id}")
        chosen = versions[0] if version is None else next(
            (v for v in versions if v.version == version), None
        )
        if chosen is None:
            raise KeyError(f"Agent {agent_id} 没有版本 {version}")
        return chosen, self._handlers[(chosen.id, chosen.version)]

    def catalog(self) -> list[dict[str, Any]]:
        """Serialisable catalog for the UI and for workflow validation."""
        out: list[dict[str, Any]] = []
        for agent_id, versions in self._by_id.items():
            current = versions[0]
            out.append(
                {
                    "id": agent_id,
                    "current_version": current.version,
                    "versions": [v.version for v in versions],
                    "label": current.label,
                    "role": current.role,
                    "stage": current.stage,
                    "skills": [
                        {
                            "name": s.name,
                            "description": s.description,
                            "input_keys": list(s.input_keys),
                            "output_keys": list(s.output_keys),
                        }
                        for s in current.skills
                    ],
                    "tools": [
                        {"tool": t.tool, "max_calls_per_run": t.max_calls_per_run}
                        for t in current.tools
                    ],
                    "memory": current.memory.kind,
                    # "Independent" must mean "belongs to the checker side", not
                    # "declares any group". Every agent declares a producer or
                    # checker group, so the old test returned True for all of them
                    # and the flag carried no information.
                    "independent": current.independence_group == "checker",
                    "independence_group": current.independence_group,
                    "notes": current.notes,
                }
            )
        return out

    def check_contract(self, producer_id: str, consumer_id: str) -> list[str]:
        """Validate that one agent's output can feed another's input.

        Returned strings are human-readable violations; empty means compatible.
        The workflow validator calls this for every edge, so a mis-wired graph
        fails fast at build time instead of producing nonsense at run time.
        """
        prod, _ = self.resolve(producer_id)
        cons, _ = self.resolve(consumer_id)
        produced: set[str] = set()
        for skill in prod.skills:
            produced.update(skill.output_keys)
        needed: set[str] = set()
        for skill in cons.skills:
            needed.update(skill.input_keys)
        if not needed or not produced:
            return []
        missing = needed - produced
        return [f"{consumer_id} 需要 {k}，但 {producer_id} 不产出" for k in sorted(missing)]


def _version_key(version: str) -> tuple[int, ...]:
    """Sort versions numerically so 1.10 beats 1.9."""
    parts = []
    for chunk in version.replace("-", ".").split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            parts.append(0)
    return tuple(parts)


#: Process-wide registry. The API layer and the workflow executor share it.
registry = AgentRegistry()
