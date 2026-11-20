"""The six-agent production model — declarations for the current pipeline.

This is the roster the production spec describes:

  director        导演总控      理解需求、制定计划、调度、协调返工
  script_adaptor  内容改编      编剧 + 历史考据 + 资料改编
  storyboard      剧本分镜      拆镜、画面、运镜、镜头提示词
  consistency     一致性        人物、场景、道具、视觉与声音资产
  video_gen       视频生成      视频提示词、模型调用、视频及声音生成
  quality_review  质量审查      剧本/镜头/成片审核、问题定位及返工建议

What each declaration carries, and why the fields matter to the runtime:

  · `skills` — declares what state keys the agent consumes and produces. The
    executor uses these to compute dependencies, so a wrong key here does not
    just mis-document; it makes the agent start before its input exists.
  · `stage` — drives ordering: pre / gen / post / review. The pipeline reads it.
  · `independence_group` — which agents may run concurrently.
  · `memory` — what survives this run. Only the director carries cross-run
    memory, because only it makes scheduling decisions worth remembering.

The ids are lowercase snake_case and match the prompt filenames, so
`prompts/<id>.txt` is unambiguously the prompt for that agent.
"""

from __future__ import annotations

from typing import Any, Callable

from ..agents.base import AgentContext
from ..platform.registry import AgentDefinition, AgentSkill, MemoryScope, registry


#: Version for this roster. Bumped whenever a declaration below changes meaning
#: (not on wording), because a run records the version it used so a rerun can be
#: reproduced even after the agents are edited.
VERSION = "3.0.0"


def _no_impl(agent_id: str) -> Callable[..., Any]:
    """Placeholder for an agent declared but not yet implemented.

    Declaring before implementing is deliberate: the roster, the prompt and the
    state contract can be reviewed and tested while the body is still missing,
    and a run that reaches an unimplemented stage fails with this message instead
    of a KeyError on a missing registry entry.
    """

    async def _missing(ctx: AgentContext, definition: AgentDefinition, options: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(
            f"Agent「{agent_id}」已声明但尚未实现。"
            f"它属于六 Agent 方案的后续步骤，当前不可执行。"
        )

    return _missing


def definitions() -> list[tuple[AgentDefinition, Callable[..., Any]]]:
    """The six declarations, in the order the director drives them.

    Implementations are imported lazily. Importing `doc.agents` at module load
    would create a cycle: that module imports this one to register the roster.
    """

    from . import agents_six_impl as six

    def fn(name: str) -> Callable[..., Any]:
        """Resolve an implementation by its function name, or a clear stub."""
        return getattr(six, name, _no_impl(name.replace("_agent", "")))

    defs: list[tuple[AgentDefinition, Callable[..., Any]]] = [
        # ------------------------------------------------------------ director
        (
            AgentDefinition(
                id="director", version=VERSION, label="导演",
                role="理解需求、制定制作计划、识别输入类型、选择工作流、调度子 Agent、维护项目状态、协调返工",
                stage="orchestration", independence_group="orchestrator",
                skills=(
                    AgentSkill(
                        "plan", "识别输入类型并制定制作计划",
                        input_keys=("chapter_text",),
                        output_keys=("segments", "material_check", "production_plan"),
                    ),
                ),
                memory=MemoryScope(kind="run", max_items=20),
                notes="唯一总控；不直接创作内容。输入类型与工作流选择在提示词里定义。",
            ),
            fn("director_six_agent"),
        ),

        # ------------------------------------------------------ script adaptor
        (
            AgentDefinition(
                id="script_adaptor", version=VERSION, label="内容改编",
                role="编剧 + 历史考据 + 资料改编；产出 adapted_script，不覆盖 source_content",
                stage="pre", independence_group="writer",
                skills=(
                    AgentSkill(
                        "adapt", "把原始素材改编为可拍摄的脚本",
                        input_keys=("segments",),
                        output_keys=("adapted_script", "historical_facts",
                                     "unverified_facts", "preservation_rules"),
                    ),
                ),
                notes="考核点与创作并存，因此未核实内容必须单独标记而不能混入史实。",
            ),
            fn("script_adaptor_agent"),
        ),

        # ---------------------------------------------------------- storyboard
        (
            AgentDefinition(
                id="storyboard", version=VERSION, label="剧本分镜",
                role="剧本拆镜、画面、运镜、镜头提示词；先做时长分析再拆",
                stage="pre", independence_group="writer",
                skills=(
                    AgentSkill(
                        "shotlist", "把脚本拆成带时长分析的分镜表",
                        input_keys=("adapted_script",),
                        output_keys=("storyboards", "duration_analysis"),
                    ),
                ),
                notes="必须先输出 duration_analysis 再拆镜："
                      "旁白与对白可同时发生，简单累加会算出错误的时长。",
            ),
            fn("storyboard_six_agent"),
        ),

        # --------------------------------------------------------- consistency
        (
            AgentDefinition(
                id="consistency", version=VERSION, label="一致性",
                role="人物、场景、道具与声音资产；四视图提示词与跨镜头一致性规则",
                stage="pre", independence_group="assets",
                skills=(
                    AgentSkill(
                        "assets", "建立统一的人物/场景/道具/声音资产",
                        input_keys=("storyboards",),
                        output_keys=("asset_manifest", "character_assets", "scene_assets",
                                     "prop_assets", "voice_profiles",
                                     "image_generation_tasks"),
                    ),
                ),
                notes="四视图必须同一人物；模型不可靠时拆成四张独立提示词。",
            ),
            fn("consistency_agent"),
        ),

        # ----------------------------------------------------------- video_gen
        (
            AgentDefinition(
                id="video_gen", version=VERSION, label="视频生成",
                role="视频提示词、模型调用、视频及声音生成；保留原始分镜提示词",
                stage="gen", independence_group="producer",
                skills=(
                    AgentSkill(
                        "render", "按模型能力适配提示词并生成视频",
                        input_keys=("storyboards", "asset_manifest"),
                        output_keys=("video_generation_tasks", "shot_assets"),
                    ),
                ),
                notes="双提示词：storyboard_prompt 不可覆盖，final_video_prompt 是适配结果。",
            ),
            fn("video_gen_agent"),
        ),

        # ------------------------------------------------------ quality_review
        (
            AgentDefinition(
                id="quality_review", version=VERSION, label="质量审查",
                role="剧本/分镜/资产/视频/成片五阶段审核，定位问题并指定返工 Agent",
                stage="review", independence_group="checker",
                skills=(
                    AgentSkill(
                        "review", "分阶段打分并给出可执行的返工指令",
                        input_keys=("shot_assets", "storyboards", "adapted_script"),
                        output_keys=("quality_reports", "critic_verdict", "reshoot"),
                    ),
                ),
                notes="规则先行（未生成即判失败），模型补充规则看不到的问题。"
                      "必须指定 responsible_agent，否则导演无法定向返工。",
            ),
            fn("quality_review_agent"),
        ),
    ]
    return defs


def register_six() -> None:
    """Register the six-agent roster. Idempotent: registry de-duplicates by id."""
    for definition, handler in definitions():
        registry.register(definition, handler)


def registered_ids() -> list[str]:
    """Ids this module declares, without touching the registry."""
    return [d.id for d, _ in definitions()]
