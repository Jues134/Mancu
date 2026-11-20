"""Agent prompts, kept in .txt files next to this module.

Why files rather than string constants in the source: a prompt is *content*, and
content gets edited far more often than the code around it. As Python string
literals every change meant re-reading a 1500-line module to find the right
heredoc, and a stray triple-quote inside a prompt broke the module. As .txt files
each agent's prompt can be edited, diffed and copied into a model playground
without touching code at all.

Design notes:

  · **One file per agent**, named `<agent id>.txt`. The agent ids match the
    active AgentDefinition ids, so a file's owner is obvious.

  · **Shared rules come from `_global_rules.txt`** and are prepended to every
    prompt at load time. The production spec states the same numbering, JSON and
    status rules for all six agents; copying them into six files would guarantee
    they drift apart. One file, one source of truth.

  · **Files are read on every call, not at import.** A prompt edit should take
    effect without restarting the server — that is the whole point of moving
    them out of code. Reading a small UTF-8 file per model call is negligible
    next to an LLM request, and caching would make "I changed the prompt and
    nothing happened" a recurring mystery.

  · **Encoding is forced to UTF-8.** The prompts are Chinese; a Windows default
    of cp936 would decode them into mojibake that the model is happy to act on.

  · **A missing file raises.** Silently falling back to an empty prompt would
    produce a model reply that looks plausible and is unconstrained, which is
    much harder to diagnose than an exception naming the missing path.

  · **Delivery is stripped, not normalised.** The .txt keeps its trailing
    newline so editors and git are happy, but that newline is not sent: a
    trailing blank line in a system message is noise the model has to skip.
"""

from __future__ import annotations

from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent

#: Prepended to every agent prompt. Not itself an agent.
GLOBAL_RULES_FILE = "_global_rules.txt"

#: The six agents of the current production model, in the order the director
#: drives them. Kept explicit rather than discovered by listing the directory:
#: a stray file dropped here (a backup, an editor temp file) must not become
#: loadable, and a typo in an agent id must fail loudly.
AGENTS = (
    "director",
    "script_adaptor",
    "storyboard",
    "consistency",
    "video_gen",
    "quality_review",
)

#: Supporting prompt used by the video-prompt rewriter.
AUXILIARY_PROMPTS = ("visual",)


class PromptNotFound(RuntimeError):
    """Raised when an agent's prompt file is missing or empty."""


def prompt_path(agent_id: str) -> Path:
    return PROMPT_DIR / f"{agent_id}.txt"


def load_global_rules() -> str:
    """Read the shared production rules prepended to every agent prompt."""
    path = PROMPT_DIR / GLOBAL_RULES_FILE
    if not path.is_file():
        raise PromptNotFound(f"缺少全局规则文件：{path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise PromptNotFound(f"全局规则文件是空的：{path}")
    return text


def load_prompt(agent_id: str, *, with_global_rules: bool = True) -> str:
    """Read one agent's system prompt, with the shared rules in front.

    Read fresh every time on purpose - see the module docstring.

    `with_global_rules=False` returns the role half alone. That is what a
    debugging view wants: the shared rules are identical for all six agents, so
    showing them six times hides what actually differs between them.
    """
    path = prompt_path(agent_id)
    if not path.is_file():
        raise PromptNotFound(
            f"找不到提示词文件：{path}。"
            f"已知提示词：{', '.join(AGENTS + AUXILIARY_PROMPTS)}"
        )

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise PromptNotFound(f"提示词文件是空的：{path}")

    if not with_global_rules:
        return text

    # A separator line keeps the model from reading the last rule as the first
    # sentence of the role instruction.
    return f"{load_global_rules()}\n\n---\n\n{text}"


def describe_prompts() -> dict[str, dict[str, object]]:
    """Where each prompt lives and how big it is — for a diagnostics endpoint.

    Returns the path and size rather than the text: the settings UI needs to say
    "this agent's prompt is this file", and dumping six system prompts into a
    JSON response would be a lot of bytes to read once.
    """
    out: dict[str, dict[str, object]] = {}

    shared = PROMPT_DIR / GLOBAL_RULES_FILE
    out["_global_rules"] = {
        "path": str(shared),
        "exists": shared.is_file(),
        "chars": len(shared.read_text(encoding="utf-8").strip()) if shared.is_file() else 0,
    }

    for agent_id in AGENTS:
        p = prompt_path(agent_id)
        out[agent_id] = {
            "path": str(p),
            "exists": p.is_file(),
            "chars": len(p.read_text(encoding="utf-8").strip()) if p.is_file() else 0,
        }
    return out
