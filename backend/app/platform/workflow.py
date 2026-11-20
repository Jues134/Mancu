"""Workflow DSL — the platform's "工作流编排" module.

Competition requirement this satisfies: workflows must be expressible as
code/config and must support **parallel, branch, loop and nested sub-flows**.

Design decision that matters: the graph is data (a dict), not code. A pipeline
built from `if node_id == "x": ...` cannot be inspected, validated, drawn in a
UI, or diffed — and "visual orchestration" is an explicit requirement. Keeping
the graph as data lets the same definition drive execution, validation, the UI
and the documentation.

Four node kinds cover the required control flow:

  step      — run one agent
  parallel  — run several branches concurrently, then join
  branch    — choose one branch by a predicate over the run context
  loop      — repeat until a predicate says stop (bounded by max_rounds)
  subflow   — run another workflow as a nested graph (composition)
  human     — pause for a user decision (the "agent negotiates, user chooses" gate)

Every loop and every branch carries an explicit bound. An unbounded loop in an
LLM pipeline is a cost incident waiting to happen, so `max_rounds` is mandatory
rather than optional.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


# ============================================================================
# Node model
# ============================================================================


class WorkflowError(ValueError):
    """Raised when a graph is malformed; messages name the offending node."""


@dataclass
class Node:
    id: str
    kind: str
    #: step: agent id. subflow: workflow id.
    ref: str = ""
    #: Explicit successor(s). A list means "fan out then join" without needing a
    #: parallel node; a single id is the ordinary sequence case.
    next: list[str] = field(default_factory=list)
    #: parallel: branch entry ids to run concurrently (joins at `next`).
    branches: list[str] = field(default_factory=list)
    #: branch: predicate over the context, evaluated as a small expression.
    when: str = ""
    then: str = ""
    otherwise: str = ""
    #: loop: body node id, stop predicate, and the mandatory bound.
    body: str = ""
    until: str = ""
    max_rounds: int = 3
    #: human: prompt shown to the user.
    prompt: str = ""
    #: Optional per-node overrides (model, timeout, retries).
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class Workflow:
    id: str
    label: str
    nodes: dict[str, Node]
    entry: str
    #: Version so a run can cite the graph it executed, like agent versions.
    version: str = "1.0.0"
    description: str = ""


# ============================================================================
# Predicate language
# ============================================================================

#: Deliberately tiny: `path op value`. No eval(), no arbitrary expressions.
#:
#: Why not eval(): a workflow file is user-editable configuration. `eval` there
#: would let a config edit execute code, and the predicates needed here are
#: comparisons on run context — a few operators cover all of them.
_PREDICATE = re.compile(
    r"^([a-zA-Z_][a-zA-Z0-9_.]*)\s*(==|!=|>=|<=|>|<)?\s*('.*?'|\".*?\"|true|false|null|-?\d+(?:\.\d+)?)$"
)


def evaluate(context: dict[str, Any], expression: str) -> bool:
    """Evaluate a `path op value` predicate against the run context.

    Paths use dots: `claims.pending_count`, `review.verdict`, `facts.confidence`.
    Missing paths resolve to None so a predicate never throws mid-run — a branch
    that cannot decide must fail loudly at validation time, not silently at
    execution time, and validation uses the same evaluator.
    """
    text = (expression or "").strip()
    if not text:
        raise WorkflowError("分支缺少判断条件")
    m = _PREDICATE.match(text)
    if not m:
        raise WorkflowError(
            f"无法解析的条件：{expression!r}（支持 `path op value`，例如 review.verdict == 'fail'）"
        )
    path, op, raw = m.groups()
    left = _resolve_path(context, path)
    right = _literal(raw)

    if op is None:
        return bool(left)
    try:
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op == ">":
            return float(left) > float(right)  # type: ignore[arg-type]
        if op == "<":
            return float(left) < float(right)  # type: ignore[arg-type]
        if op == ">=":
            return float(left) >= float(right)  # type: ignore[arg-type]
        if op == "<=":
            return float(left) <= float(right)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return False


def _resolve_path(context: dict[str, Any], path: str) -> Any:
    cur: Any = context
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def _literal(raw: str) -> Any:
    raw = raw.strip()
    if raw.startswith(("'", '"')) and raw.endswith(("'", '"')):
        return raw[1:-1]
    if raw == "true":
        return True
    if raw == "false":
        return False
    if raw == "null":
        return None
    try:
        return float(raw) if "." in raw else int(raw)
    except ValueError:
        return raw


# ============================================================================
# Builder / validator
# ============================================================================


def parse_workflow(data: dict[str, Any]) -> Workflow:
    """Build a Workflow from its JSON form, validating every reference."""
    wf_id = str(data.get("id") or "").strip()
    if not wf_id:
        raise WorkflowError("工作流缺少 id")

    raw_nodes = data.get("nodes") or []
    if not raw_nodes:
        raise WorkflowError(f"工作流 {wf_id} 没有节点")

    nodes: dict[str, Node] = {}
    for item in raw_nodes:
        raw_next = item.get("next")
        if isinstance(raw_next, list):
            next_ids = [str(x) for x in raw_next]
        elif raw_next:
            next_ids = [str(raw_next)]
        else:
            next_ids = []
        # `or 3` would silently turn an explicit 0 into 3, which would let a
        # genuinely unbounded loop through validation. Only a *missing* value
        # gets the default.
        raw_rounds = item.get("max_rounds")
        node = Node(
            id=str(item.get("id") or "").strip(),
            kind=str(item.get("kind") or "step"),
            ref=str(item.get("ref") or ""),
            next=next_ids,
            branches=[str(b) for b in (item.get("branches") or [])],
            when=str(item.get("when") or ""),
            then=str(item.get("then") or ""),
            otherwise=str(item.get("otherwise") or ""),
            body=str(item.get("body") or ""),
            until=str(item.get("until") or ""),
            max_rounds=3 if raw_rounds is None else int(raw_rounds),
            prompt=str(item.get("prompt") or ""),
            options=dict(item.get("options") or {}),
        )
        if not node.id:
            raise WorkflowError("有节点缺少 id")
        if node.id in nodes:
            raise WorkflowError(f"节点 id 重复：{node.id}")
        if node.kind not in {"step", "parallel", "branch", "loop", "subflow", "human"}:
            raise WorkflowError(f"节点 {node.id} 的类型未知：{node.kind}")
        nodes[node.id] = node

    entry = str(data.get("entry") or raw_nodes[0].get("id"))
    if entry not in nodes:
        raise WorkflowError(f"入口节点不存在：{entry}")

    wf = Workflow(
        id=wf_id,
        label=str(data.get("label") or wf_id),
        nodes=nodes,
        entry=entry,
        version=str(data.get("version") or "1.0.0"),
        description=str(data.get("description") or ""),
    )
    validate_workflow(wf)
    return wf


def validate_workflow(wf: Workflow, known_agents: set[str] | None = None) -> None:
    """Structural checks that must pass before a run is allowed to start.

    Catches what would otherwise surface as a confusing mid-run failure:
    dangling node references, a loop without a bound, a branch whose predicate
    cannot parse, and steps pointing at agents that do not exist.
    """
    errors: list[str] = []

    for node in wf.nodes.values():
        if node.kind == "step":
            if not node.ref:
                errors.append(f"节点 {node.id}（step）缺少 ref")
            elif known_agents is not None and node.ref not in known_agents:
                errors.append(f"节点 {node.id} 引用了未注册的 Agent：{node.ref}")
        elif node.kind == "subflow":
            if not node.ref:
                errors.append(f"节点 {node.id}（subflow）缺少 ref")
        elif node.kind == "parallel":
            if len(node.branches) < 2:
                errors.append(f"节点 {node.id}（parallel）至少需要两个分支")
            for b in node.branches:
                if b not in wf.nodes:
                    errors.append(f"节点 {node.id} 的并行分支不存在：{b}")
        elif node.kind == "branch":
            if not node.then or node.then not in wf.nodes:
                errors.append(f"节点 {node.id}（branch）的 then 指向不存在节点：{node.then!r}")
            if node.otherwise and node.otherwise not in wf.nodes:
                errors.append(f"节点 {node.id}（branch）的 otherwise 指向不存在节点：{node.otherwise}")
            try:
                evaluate({}, node.when)
            except WorkflowError as exc:
                errors.append(f"节点 {node.id}（branch）条件无法解析：{exc}")
        elif node.kind == "loop":
            if not node.body or node.body not in wf.nodes:
                errors.append(f"节点 {node.id}（loop）的 body 指向不存在节点：{node.body!r}")
            if node.max_rounds <= 0:
                errors.append(f"节点 {node.id}（loop）必须有正的 max_rounds（无界循环会烧钱）")
            try:
                evaluate({}, node.until)
            except WorkflowError as exc:
                errors.append(f"节点 {node.id}（loop）终止条件无法解析：{exc}")
        elif node.kind == "human":
            if not node.prompt:
                errors.append(f"节点 {node.id}（human）缺少 prompt")

    # Reachability: an unreachable node is dead config that will confuse whoever
    # edits the graph next.
    reachable: set[str] = set()
    stack = [wf.entry]
    while stack:
        nid = stack.pop()
        if nid in reachable or nid not in wf.nodes:
            continue
        reachable.add(nid)
        node = wf.nodes[nid]
        for target in _successors(node):
            if target in wf.nodes:
                stack.append(target)

    orphans = set(wf.nodes) - reachable
    if orphans:
        errors.append(f"有节点无法从入口到达：{', '.join(sorted(orphans))}")

    if errors:
        raise WorkflowError("；".join(errors))


def _successors(node: Node) -> list[str]:
    if node.kind == "branch":
        return [node.then] + ([node.otherwise] if node.otherwise else [])
    if node.kind == "loop":
        # Both the body (executed each round) and the node's own successors (run
        # once after the loop settles) are reachable.
        return [node.body, *node.next] if node.body else list(node.next)
    if node.kind == "parallel":
        return node.branches
    # step / subflow / human follow their declared successors
    return list(node.next)


def to_dict(wf: Workflow) -> dict[str, Any]:
    """Serialise back to the JSON form (the UI draws from this)."""
    return {
        "id": wf.id,
        "label": wf.label,
        "version": wf.version,
        "description": wf.description,
        "entry": wf.entry,
        "nodes": [
            {
                "id": n.id,
                "kind": n.kind,
                "ref": n.ref,
                "next": n.next,
                "branches": n.branches,
                "when": n.when,
                "then": n.then,
                "otherwise": n.otherwise,
                "body": n.body,
                "until": n.until,
                "max_rounds": n.max_rounds,
                "prompt": n.prompt,
                "options": n.options,
            }
            for n in wf.nodes.values()
        ],
    }
