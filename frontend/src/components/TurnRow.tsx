import { useState } from "react";

import { agentInitial } from "../agentLabel";

import { AskCard, type AskQuestion } from "./AskCard";
import { DecisionCard, type DecisionCardData } from "./DecisionCard";
import { PlanStub, type PlanCardData } from "./PlanCard";
import { AssetImage } from "./Lightbox";

/**
 * Conversation turn showing an agent's reasoning AND its actual artifacts.
 *
 * The requirement is that each agent exposes what it produced, not a summary:
 *
 *   director       → how it split the text, and the material verdict
 *   script_adaptor → the adapted script and factual constraints
 *   storyboard     → each shot with its generation prompt
 *   consistency    → reference portraits and reusable assets
 *   video_gen      → generated clips and the finished film
 *   quality_review → acceptance findings and rework instructions
 *
 * Collapsed by default so the conversation stays readable, but the artifacts are
 * what a reviewer opens the panel for, so a rejected turn and a portrait-bearing
 * turn both start open.
 */

/** How a shot says it was anchored, and how it relates to its predecessor.
 *  Both come from the backend (frame_chain.py); spelling them here too keeps the
 *  chip readable while an unknown value simply shows no chip. */
const ANCHOR_LABELS: Record<string, string> = {
  prev_last_frame: "上一镜尾帧",
  scene_reference: "场景参考图",
  character_sheet: "人物设定稿",
};

const LINK_LABELS: Record<string, string> = {
  CONTINUOUS_ACTION: "连续动作",
  SAME_SCENE_CUT: "同景切镜",
  SCENE_TRANSITION: "切场景",
  TIME_JUMP: "时间跳跃",
  MONTAGE: "蒙太奇",
};

/** 台账步骤的全称标签。以前的单字缩写（语/决/警）读不懂——用户要的是
 * 「这一步在干什么」，不是一个需要口口相传的代号。 */
export const STEP_ICON: Record<string, string> = {
  reasoning: "思考",
  model: "调用模型",
  retrieve: "检索",
  tool: "工具",
  think: "规划",
  decide: "程序小结",
  reject: "驳回",
  accept: "验收",
  write: "产出",
  warn: "警告",
  error: "出错",
};

/** 每类步骤的一句定位说明，跟在标题下方的小字。这不是给开发者的日志，
 * 是给用户看的过程账本：他应该不用问任何人就能读懂每行在说什么。 */
export const STEP_EXPLAIN: Record<string, string> = {
  reasoning: "模型边想边流出的原始思考过程（有思考功能的模型才有）",
  model: "把任务说明和材料原文发给大模型，等它返回结果；显示输入输出规模和耗时",
  retrieve: "在本地知识库里按检索词查原文材料",
  tool: "调用外部工具完成一步具体工作（如联网搜索）",
  think: "模型动手之前先做的推理与规划",
  decide: "模型返回后由程序（不是模型自己）统计出的小结，数字可以直接信任",
  reject: "审核没通过的项，会退回对应岗位重做",
  accept: "审核确认通过的项",
  write: "整理好并保存进任务记录的成果",
  warn: "出了状况但程序会自己处理（通常是自动重试），暂时不需要用户做任何事",
  error: "这一步失败了，需要看具体原因",
};

/** 每个岗位「这一步在做什么」的一句定位，显示在卡片头部名字旁边。
 * 台账读起来像天书，一半原因是读者不知道这个岗位存在的目的。 */
export const AGENT_ABOUT: Record<string, string> = {
  director: "总控：读懂你的要求、查知识库、定分段和任务清单；它不亲自创作内容",
  script_adaptor: "把材料原文改编成可拍摄的脚本：判断改编方式、分节，并把史实/存疑/创作分开登记",
  storyboard: "把改编好的脚本拆成一个个镜头：画面、旁白、时长和镜头之间的衔接",
  consistency: "为人物、场景、道具建立统一的视觉与声音设定，保证各镜头里的它们长得一样",
  video_gen: "按每个镜头的提示词生成画面与声音",
  quality_review: "审核每个段落的产出：对照验收清单找问题，决定通过还是打回重做",
};

/** One distinct finding, plus the shots it applies to.
 *
 * The rule checker reports per shot, so six shots sharing one defect arrived as
 * six identical rows — and because the check read a field the storyboard does not
 * have, every row also carried the same placeholder shot id. A reader saw a wall
 * of duplicates and no way to tell whether it was one problem or six.
 */
interface FindingGroup {
  severity: string;
  kind: string;
  reason: string;
  fix: string;
  shots: string[];
  /** How many raw findings collapsed into this group. */
  n: number;
}

function groupFindings(items: Array<Record<string, any>>): FindingGroup[] {
  const groups: FindingGroup[] = [];
  const at = new Map<string, number>();

  for (const it of items) {
    // The backend names these `issue_type` / `description` / `repair_instruction`;
    // older payloads use `kind` / `detail` / `fix_hint`. Both are accepted so a
    // stored run still renders instead of showing two bare dashes.
    const severity = String(it.severity || "").toUpperCase();
    const kind = String(it.issue_type || it.kind || it.stage || it.type || "");
    const reason = String(it.description || it.detail || it.reason || "");
    const fix = String(it.repair_instruction || it.fix_hint || it.fix || "");
    const shot = String(it.shot_id || it.shot || "");
    const usableShot = shot && shot !== "?" ? shot : "";

    const key = [severity, kind, reason, fix].join("\u0000");
    const index = at.get(key);
    if (index === undefined) {
      at.set(key, groups.length);
      groups.push({ severity, kind, reason, fix, shots: usableShot ? [usableShot] : [], n: 1 });
      continue;
    }
    const group = groups[index];
    group.n += 1;
    if (usableShot && !group.shots.includes(usableShot)) group.shots.push(usableShot);
  }

  return groups;
}

interface Props {
  turn: {
    role: string;
    agent: string;
    kind: string;
    content: string;
    payload: Record<string, any>;
  };
  agentName: string;
  /** 当前这条任务。问卡要往它上面提交回答。 */
  runId?: string;
  /** 任务还停在等回答吗；不是的话问卡不给按钮。 */
  canAnswer?: boolean;
  onAnswered?: () => void;
  /** 任务已经结束（成功但有失败段，或整体失败）——结果消息上给「继续生成」。 */
  canRetry?: boolean;
  onRetry?: () => void;
}

export default function TurnRow({
  turn,
  agentName,
  runId,
  canAnswer = false,
  onAnswered,
  canRetry = false,
  onRetry,
}: Props) {
  const payload = turn.payload || {};
  const steps: Array<Record<string, any>> = payload.steps || [];
  const issues: Array<Record<string, any>> = payload.issues || [];
  /** Pre-rendered reason+fix pairs for a rework, produced by the backend. */
  const blame: Array<Record<string, any>> = payload.blame || [];
  /** Which agent the review named as responsible for the rejected work. */
  const blamedAgent: string = payload.target || "";
  const artifacts: Record<string, any> = payload.artifacts || {};
  /** 结果消息里逐段的失败详情（后端在 payload.failed_detail 里给好了）。 */
  const failedDetail: Array<Record<string, any>> = payload.failed_detail || [];
  const rejected = turn.kind === "review" && !payload.approved;
  const isRework = turn.kind === "rework";
  const isError = turn.kind === "error";

  /** Pre-rendered blame pairs win when present; otherwise the raw findings. */
  const findings: Array<Record<string, any>> = blame.length ? blame : issues;
  const grouped = groupFindings(findings);

  const hasArtifacts = Object.keys(artifacts).length > 1;
  const [open, setOpen] = useState(
    rejected || isRework || isError || artifacts.kind === "portraits"
      // `assets` is the prompt deliverable — collapsed by default it looked like
      // the stage had produced nothing but its one-line summary.
      || artifacts.kind === "assets"
      // A review that recorded findings is a report, not a status line: hiding it
      // behind 展开 is why a wall of duplicates was only discovered by clicking.
      || (turn.kind === "review" && (issues.length > 0 || blame.length > 0))
      || artifacts.kind === "shots" || artifacts.kind === "clips" || artifacts.kind === "film",
  );

  if (turn.role === "user") {
    // Show what the user actually typed. A long chapter would take over the
    // screen, so it is clamped with a "展开全文" toggle rather than truncated
    // silently — the reader must be able to check the exact input.
    const chars = (payload.chars as number) || turn.content.length;
    const isLong = turn.content.length > 160;
    return (
      <div className="turn user">
        <div className="turn-head">
          <span className="turn-avatar solid">你</span>
          <span className="turn-name">你</span>
          <span className="turn-tag">{chars} 字</span>
          {payload.target_seconds ? (
            <span className="turn-tag">目标 {payload.target_seconds} 秒</span>
          ) : null}
        </div>
        <div className="turn-body">
          <div className={`turn-text user-input ${isLong && !open ? "clamped" : ""}`}>
            {turn.content}
          </div>
          {isLong && (
            <button className="trace-toggle" onClick={() => setOpen((v) => !v)}>
              {open ? "收起" : "展开全文"}
            </button>
          )}
        </div>
      </div>
    );
  }

  if (turn.kind === "decision") {
    // 判定过程自己成卡：它是用户唯一能核对"我说的话被怎么读了"的地方，
    // 折进 steps 里就等于没显示。
    return <DecisionCard card={payload as DecisionCardData} />;
  }

  if (turn.kind === "ask") {
    // 停在等你选：任务没失败，只是这一次材料要你决定从哪儿找。
    return (
      <AskCard
        card={payload as AskQuestion}
        runId={runId}
        canAnswer={canAnswer}
        onAnswered={onAnswered}
      />
    );
  }

  if (turn.kind === "plan") {
    // 清单本体在右上角（`PlanDock`）。这里留一行，说明它发生过、现在到哪儿了——
    // 既不会在屏幕上出现两份同样的卡，也不会让往回翻的人以为这里什么都没有。
    return <PlanStub card={payload as PlanCardData} />;
  }

  if (turn.kind === "warning") {
    const toolError = payload.mcp_error as Record<string, any> | undefined;
    return (
      <div className={`turn system ${toolError ? "mcp-tool-warning" : ""}`}>
        <div className="turn-body">
          <div className="turn-text muted">{turn.content}</div>
          {toolError && <div className="mcp-error-detail">
            <div className="mcp-error-tags">
              <span className="turn-tag">{toolError.provider || "外部工具"}</span>
              <span className="turn-tag">阶段：{toolError.stage || "调用"}</span>
              <span className="turn-tag">{toolError.retryable ? "可以重试" : "处理后再试"}</span>
            </div>
            {toolError.impact && <p><strong>任务影响：</strong>{toolError.impact}</p>}
            {toolError.user_action && <p><strong>下一步：</strong>{toolError.user_action}</p>}
            {toolError.error_type && <details><summary>技术信息</summary><code>{toolError.tool || "unknown"} · {toolError.error_type}</code></details>}
          </div>}
        </div>
      </div>
    );
  }

  return (
    <div className={`turn agent ${rejected || isRework || isError ? "bad" : ""}`}>
      <div className="turn-head">
        <span className="turn-avatar">{agentInitial(agentName)}</span>
        <span className="turn-name">{agentName}</span>
        {payload.segment && <span className="turn-tag">{payload.segment}</span>}
        {payload.attempts > 1 && <span className="turn-tag">第{payload.attempts}次</span>}
        {rejected && <span className="chip fail">未通过</span>}
        {isRework && <span className="chip fail">打回重做</span>}
        {turn.kind === "review" && payload.approved && <span className="chip ok">通过</span>}
        {turn.kind === "result" && (
          <span className={`chip ${failedDetail.length ? "fail" : "ok"}`}>
            {failedDetail.length ? `未完成 ${failedDetail.length} 段` : "完成"}
          </span>
        )}
        {(steps.length > 0 || hasArtifacts) && (
          <button className="trace-toggle" onClick={() => setOpen((v) => !v)}>
            {open ? "收起" : `展开（${steps.length} 步${hasArtifacts ? " + 产出" : ""}）`}
          </button>
        )}
      </div>

      {/* 岗位定位：先告诉读者「这个岗位是干嘛的」，下面的步骤和摘要才有上下文。 */}
      {AGENT_ABOUT[turn.agent] && (
        <div className="turn-about">{AGENT_ABOUT[turn.agent]}</div>
      )}

      <div className="turn-body">
        <div className="turn-text">{turn.content || "已完成"}</div>

        {/* 没通过的段逐个交代：卡在哪一步、什么原因、能不能点继续。放在正文
            下面而不是折叠区——「哪段没过」是结果的一部分，不该藏在展开里。 */}
        {turn.kind === "result" && failedDetail.length > 0 && (
          <div className="issue-list">
            <div className="issue-head">没通过的段</div>
            {failedDetail.map((item, i) => (
              <div className={`issue-row ${item.retryable === false ? "" : "bad"}`} key={i}>
                <div className="issue-line">
                  <span className="issue-kind">{item.title || item.segment_id}</span>
                  <span>卡在「{item.stage}」</span>
                  {item.reason && <span className="issue-sev p1">{item.reason}</span>}
                </div>
                {item.message && item.message !== item.reason && (
                  <div className="issue-fix">{item.message}</div>
                )}
              </div>
            ))}
            {canRetry && onRetry && (
              <button
                className="btn ghost small"
                style={{ marginTop: 8 }}
                onClick={onRetry}
                title="只重跑没通过的段，已通过的段不重做"
              >
                继续生成没通过的段
              </button>
            )}
          </div>
        )}

        {open && (issues.length > 0 || blame.length > 0) && (
          <div className="issue-list">
            {/* Three questions a reader has when a stage is rejected: which agent
                was blamed, what was wrong, and what should change. A bare list of
                complaints answers only the middle one.
                A passing review can still carry recorded findings; calling those
                "打回理由" told the reader the run had been rejected when it had
                not, so the heading follows the verdict. */}
            <div className="issue-head">
              {rejected || isRework ? "打回理由" : "审查发现"}
              {findings.length > grouped.length && (
                <span className="issue-count">
                  {findings.length} 条 · {grouped.length} 类
                </span>
              )}
              {blamedAgent && <span className="issue-target">责任方：{blamedAgent}</span>}
            </div>
            {grouped.map((g, i) => {
              // Severity moved from CRITICAL/MAJOR/MINOR to P0–P3 when the review
              // modes landed; the rule checker still emits "fail"/"warn". Both are
              // accepted so a blocking finding never renders as a passing note.
              const sev = g.severity;
              const bad =
                sev === "P0" || sev === "P1" || sev === "CRITICAL" || sev === "MAJOR" || sev === "FAIL";

              return (
                <div className={`issue-row ${bad ? "bad" : ""}`} key={i}>
                  <div className="issue-line">
                    {sev && <span className={`issue-sev ${sev.toLowerCase()}`}>{sev}</span>}
                    {g.kind && <span className="issue-kind">{g.kind}</span>}
                    {/* The count is the point: one rule defect on six shots is one
                        thing to fix, not six rows of identical text. */}
                    {g.n > 1 && <span className="issue-mult">×{g.n}</span>}
                    <span className="issue-detail">{g.reason}</span>
                  </div>
                  {g.shots.length > 0 && (
                    <div className="issue-shots">
                      <span className="issue-shots-label">镜头</span>
                      {g.shots.join("、")}
                    </div>
                  )}
                  {g.fix && (
                    <div className="issue-fix">
                      <span className="issue-fix-label">怎么改</span>
                      {g.fix}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {open && steps.length > 0 && <Steps steps={steps} />}

        {/* Artifacts come after the thinking: the sketch of this panel is
            思考 → 提示词 → 图/视频, and a reader follows the same order the agent
            worked in. For a rejection the reasons stay above both, because they
            are the message. */}
        {open && artifacts.kind && <Artifacts data={artifacts} />}
      </div>
    </div>
  );
}

/** Render the stage-specific artifact. */
function Artifacts({ data }: { data: Record<string, any> }) {
  switch (data.kind) {
    case "segments":
      return (
        <div className="artifact">
          <div className="artifact-head">切分结果（{data.segments?.length || 0} 段）</div>
          {(data.segments || []).map((s: any) => (
            <div className="seg-row" key={s.id}>
              <span className="seg-title">{s.title || s.id}</span>
              <span className="seg-meta">{s.chars} 字</span>
              <span className="seg-preview">{s.preview}</span>
            </div>
          ))}
        </div>
      );

    case "constraints":
      return (
        <div className="artifact">
          <div className="artifact-head">
            历史约束 {data.era && `· ${data.era}`} {data.year_range && `（${data.year_range}）`}
          </div>
          <div className="kv-row">
            <span className="kv-label">必现</span>
            <span className="chips">
              {(data.required || []).map((x: string, i: number) => (
                <span className="chip ok" key={i}>{x}</span>
              ))}
              {!data.required?.length && <span className="muted">（无）</span>}
            </span>
          </div>
          <div className="kv-row">
            <span className="kv-label">禁现</span>
            <span className="chips">
              {(data.forbidden || []).map((x: string, i: number) => (
                <span className="chip fail" key={i}>{x}</span>
              ))}
              {!data.forbidden?.length && <span className="muted">（无）</span>}
            </span>
          </div>
          {(data.figures || []).length > 0 && (
            <div className="kv-row">
              <span className="kv-label">人物</span>
              <span className="chips">
                {data.figures.map((f: any, i: number) => (
                  <span className="chip" key={i}>{f.name}·{f.identity}</span>
                ))}
              </span>
            </div>
          )}
          {(data.hard_facts || []).length > 0 && (
            <div className="kv-row">
              <span className="kv-label">硬事实</span>
              <span className="chips">
                {data.hard_facts.map((f: string, i: number) => (
                  <span className="chip" key={i}>{f}</span>
                ))}
              </span>
            </div>
          )}
        </div>
      );

    case "shots": {
      // Both shapes are read. The payload used to carry the storyboard's own keys
      // (`shot_id` / `visual` / `duration`) while this panel read `id` /
      // `visual_description` / `seconds`, so a 24-shot board rendered as 24 rows of
      // empty chips. Stored runs still hold the old shape, so the fallbacks stay.
      const shots = (data.shots || []).map((s: any) => ({
        id: s.shot_id || s.id || "",
        seconds: Number(s.seconds ?? s.duration ?? 0),
        size: s.shot_size || s.shot_type || "",
        move: s.camera_move || s.camera_movement || "",
        narration: s.narration || "",
        visual: s.visual || s.visual_description || "",
        prompt: s.prompt || s.video_prompt || "",
        characters: s.characters || [],
        link: s.link_type || "",
        startState: s.start_state || "",
      }));
      const totalSeconds =
        Number(data.total_seconds) || shots.reduce((sum: number, s: any) => sum + s.seconds, 0);

      return (
        <div className="artifact">
          <div className="artifact-head">
            分镜表（{shots.length} 镜，{Math.round(totalSeconds)} 秒）
            {data.prompt_only && (
              <span className="chip warn" style={{ marginLeft: 8 }}>
                只出提示词
              </span>
            )}
          </div>
          {shots.map((s: any, i: number) => (
            <div className="shot-card" key={s.id || i}>
              <div className="shot-top">
                <span className="shot-id">{s.id}</span>
                {/* Chips are conditional: an empty grey pill ("—  s") made the
                    card look broken instead of simply lacking that detail. */}
                {s.size && <span className="chip">{s.size}</span>}
                {s.move && <span className="chip">{s.move}</span>}
                {s.seconds > 0 && <span className="chip">{Math.round(s.seconds)}s</span>}
                {/* Only the non-default relations are chipped. Every shot being
                    "连续动作" is the normal case; marking it on all of them would
                    bury the one card that actually cuts to a new scene. */}
                {LINK_LABELS[s.link] && s.link !== "CONTINUOUS_ACTION" && (
                  <span className="chip warn" title="与上一镜的衔接类型">
                    {LINK_LABELS[s.link]}
                  </span>
                )}
                {s.characters.length > 0 && (
                  <span className="chip ok">{s.characters.join("、")}</span>
                )}
              </div>
              {s.narration && (
                <div className="shot-line">
                  <span className="shot-k">旁白</span>
                  <span className="shot-v">{s.narration}</span>
                </div>
              )}
              {/* What the shot starts from. This is the line that makes a chain
                  checkable: it should read as the previous shot's ending, not as
                  a fresh introduction of the same person. */}
              {s.startState && (
                <div className="shot-line">
                  <span className="shot-k">开始状态</span>
                  <span className="shot-v">{s.startState}</span>
                </div>
              )}
              {/* 画面 and 提示词 describe the same shot: the prompt is the visual
                  description rewritten with lighting, lens and style appended. Two
                  paragraphs saying it twice read as duplicated content, so only one
                  is shown — the prompt, because it is the artefact people copy. The
                  plain description is used only when no prompt exists (older runs,
                  prompt-less storyboards). */}
              {s.visual && !s.prompt && (
                <div className="shot-line">
                  <span className="shot-k">画面</span>
                  <span className="shot-v mono">{s.visual}</span>
                </div>
              )}
              {s.prompt && (
                <div className="shot-prompt">
                  <div className="shot-prompt-head">
                    <span className="shot-k">提示词</span>
                    <span className="shot-prompt-note">含画面描述 · 可直接复制</span>
                    <CopyButton text={s.prompt} label="复制" />
                  </div>
                  <div className="shot-prompt-body mono">{s.prompt}</div>
                </div>
              )}
            </div>
          ))}
        </div>
      );
    }

    case "portraits":
      return (
        <div className="artifact">
          <div className="artifact-head">
            角色设定（{data.blocks?.length || 0} 人）
            {/* Say which path was taken. "No image" and "image failed" look the
                same in a grid of empty cards, and they need different actions. */}
            {data.prompt_only && (
              <span className="chip warn" style={{ marginLeft: 8 }}>
                {data.skip_reason || "只出提示词"}
              </span>
            )}
            {data.text_zh && (
              <CopyButton text={data.text_zh} label="复制全部中文" />
            )}
            {data.text_en && (
              <CopyButton text={data.text_en} label="复制全部英文" />
            )}
          </div>

          {/* Only render the image grid when there is something to render. */}
          {!data.prompt_only && (data.portraits || []).length > 0 && (
            <div className="portrait-grid">
              {(data.portraits || []).map((p: any, i: number) => (
                <div className="portrait-card" key={i}>
                  {p.url ? (
                    <AssetImage className="portrait-img" src={p.url} alt={p.name} prompt={p.prompt} />
                  ) : p.rendering ? (
                    <div className="portrait-missing">
                      <span className="spinner" />
                      <span>正在生成图片…</span>
                    </div>
                  ) : (
                    <div className="portrait-missing">未生成</div>
                  )}
                  <div className="portrait-name">{p.name}</div>
                </div>
              ))}
            </div>
          )}

          {(data.blocks || []).map((b: Record<string, any>, i: number) => (
            <PromptBlock block={b} key={i} />
          ))}
        </div>
      );

    case "assets": {
      // One group per asset kind, in the order a film is built: people, places,
      // things. A single flat list of 20 prompts with no headings was unreadable,
      // and the prompt is the product here — in 提示词模式 it is the ONLY product.
      const groups = [
        { title: "人物提示词", items: data.characters || [] },
        { title: "场景提示词", items: data.scenes || [] },
        { title: "道具提示词", items: data.props || [] },
      ];
      return (
        <div className="artifact">
          <div className="artifact-head">
            视觉资产
            {groups.map(
              (g: any) =>
                g.items.length > 0 && (
                  <span className="chip" key={g.title} style={{ marginLeft: 8 }}>
                    {g.title.replace("提示词", "")} {g.items.length}
                  </span>
                ),
            )}
            {/* Why some cards have no picture. "Not rendered" and "failed" look
                identical in an empty frame otherwise. */}
            {data.prompt_only && (
              <span className="chip warn" style={{ marginLeft: 8 }}>
                {data.skip_reason || "只出提示词"}
              </span>
            )}
            {/* Prompts are written before any picture exists now; the run draws
                them all afterwards in one serial queue. Saying so keeps an empty
                frame from reading as a failed render. */}
            {data.images_pending && (
              <span className="chip" style={{ marginLeft: 8 }}>
                参考图待全片统一生成
              </span>
            )}
          </div>

          {groups.every((g: any) => g.items.length === 0) && (
            <div className="muted" style={{ fontSize: 12 }}>
              没有素材产出：这一轮模型没有返回人物/场景/道具（多是输出被截断），可重跑该阶段。
            </div>
          )}

          {groups.map(
            (g: any) =>
              g.items.length > 0 && (
                <div className="asset-group" key={g.title}>
                  <div className="asset-group-head">{g.title}</div>
                  {g.items.map((it: any, i: number) => (
                    <div className="asset-item" key={it.id || i}>
                      <div className="asset-item-head">
                        <span className="asset-name">{it.name || it.id}</span>
                        {it.id && <span className="mono asset-id">{it.id}</span>}
                        {it.meta && <span className="asset-meta">{it.meta}</span>}
                        {it.prompt && <CopyButton text={it.prompt} label="复制提示词" />}
                      </div>
                      {it.appearance && <div className="asset-note">{it.appearance}</div>}
                      {it.prompt && <div className="asset-prompt mono">{it.prompt}</div>}
                      {/* The picture sits under its own prompt, so a reader can
                          check one against the other without scrolling. */}
                      {it.image && (
                        <AssetImage
                          src={it.image}
                          alt={it.name || it.id}
                          prompt={it.prompt}
                        />
                      )}
                    </div>
                  ))}
                </div>
              ),
          )}
        </div>
      );
    }

    case "clips":
      return (
        <div className="artifact">
          <div className="artifact-head">
            分镜视频（{data.clips?.filter((c: any) => c.ok).length || 0}/
            {data.clips?.length || 0} 成功）
            {data.prompt_only && (
              <span className="chip warn" style={{ marginLeft: 8 }}>
                只出提示词
              </span>
            )}
          </div>

          {/* Prompts above the grid: this stage's product is the prompt per shot,
              and in 提示词模式 it is the only product — the grid then reads as
              "waiting for a renderer", not as the result. */}
          {(data.prompts || []).length > 0 && (
            <div className="asset-group">
              <div className="asset-group-head">
                镜头提示词（{data.prompts.length}）
                <CopyButton
                  text={(data.prompts || [])
                    .map((p: any) => `${p.shot_id}\n${p.prompt}`)
                    .join("\n\n")}
                  label="复制全部"
                />
              </div>
              {(data.prompts || []).map((p: any, i: number) => (
                <div className="asset-item" key={i}>
                  <div className="asset-item-head">
                    <span className="asset-name mono">{p.shot_id}</span>
                    {p.seconds > 0 && <span className="asset-meta">{Math.round(p.seconds)}s</span>}
                    {p.prompt && <CopyButton text={p.prompt} label="复制" />}
                  </div>
                  <div className="asset-prompt mono">{p.prompt}</div>
                </div>
              ))}
            </div>
          )}

          <div className="clip-grid">
            {(data.clips || []).map((c: any, i: number) => (
              <div className="clip-card" key={i}>
                {c.ok && c.url ? (
                  <video className="clip-video" src={c.url} controls preload="metadata" />
                ) : c.rendering ? (
                  /* A render takes 45–70 s and longer when the gateway queue is
                     busy. Showing "未生成" during that window made a running job
                     look like a failed one; the spinner plus the shot index says
                     both what is happening and how far along the batch is. */
                  <div className="clip-pending">
                    <span className="spinner" />
                    <span>正在生成视频{c.progress ? `（${c.progress}）` : "…"}</span>
                  </div>
                ) : (
                  <div className="clip-missing">
                    {data.prompt_only
                      ? "未渲染（提示词模式）"
                      : c.error
                        ? c.error.slice(0, 60)
                        : "未生成"}
                  </div>
                )}
                <div className="clip-meta">
                  <span className="mono">{c.shot_id}</span>
                  {c.reference_used && <span className="chip ok">已用参考图</span>}
                  {/* How this shot connects to the one before it. Without these
                      two chips a break between clips is unexplainable: the
                      reader cannot tell "chained, model drifted" from "scene
                      change, supposed to jump". */}
                  {ANCHOR_LABELS[c.first_frame_kind] && (
                    <span className="chip" title="这一镜的首帧从哪来">
                      首帧：{ANCHOR_LABELS[c.first_frame_kind]}
                    </span>
                  )}
                  {LINK_LABELS[c.link_type] && (
                    <span className="chip" title="与上一镜的衔接类型">
                      {LINK_LABELS[c.link_type]}
                    </span>
                  )}
                  {c.seconds > 0 && <span className="muted">{Math.round(c.seconds)}s</span>}
                </div>
              </div>
            ))}
          </div>
        </div>
      );

    case "audio":
      return (
        <div className="artifact">
          <div className="artifact-head">
            配音（{data.clips?.filter((c: any) => c.ok).length || 0}/{data.clips?.length || 0}）
          </div>
          {(data.clips || []).filter((c: any) => c.ok).map((c: any, i: number) => (
            <div className="audio-row" key={i}>
              <span className="mono">{c.shot_id}</span>
              <audio src={c.url} controls preload="none" />
              <span className="muted">{c.seconds}s</span>
            </div>
          ))}
        </div>
      );

    case "film":
      return (
        <div className="artifact">
          <div className="artifact-head">成片 · {data.note || ""}</div>
          {data.url ? (
            <video className="film" src={data.url} controls preload="metadata" />
          ) : (
            <div className="muted">未生成成片</div>
          )}
        </div>
      );

    default:
      return null;
  }
}

/** Copies text to the clipboard and flashes a confirmation.
 *
 * Why not `navigator.clipboard` alone: it is unavailable on a non-secure origin,
 * and this app runs on plain http://127.0.0.1. The textarea fallback keeps the
 * button working there instead of silently doing nothing.
 */
function CopyButton({ text, label }: { text: string; label?: string }) {
  const [done, setDone] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
    setDone(true);
    window.setTimeout(() => setDone(false), 1400);
  }
  return (
    <button className="copy-btn" onClick={copy} title="复制到剪贴板">
      {done ? "已复制" : label || "复制"}
    </button>
  );
}

/** One bilingual prompt block: title, both versions, both negative lists. */
function PromptBlock({ block }: { block: Record<string, any> }) {
  return (
    <div className="prompt-block">
      <div className="prompt-block-head">
        <span className="prompt-title">{block.title}</span>
        {block.note && <span className="prompt-note">{block.note}</span>}
      </div>

      <div className="prompt-row">
        <span className="prompt-lang">中文（即梦）</span>
        <CopyButton text={`${block.zh}\n\n负面词：${block.negative_zh}`} label="复制中文" />
      </div>
      <div className="prompt-text">{block.zh}</div>
      <div className="prompt-neg">负面词：{block.negative_zh}</div>

      <div className="prompt-row">
        <span className="prompt-lang">English (MiniMax / Flux)</span>
        <CopyButton text={`${block.en}\n\nNegative: ${block.negative_en}`} label="复制英文" />
      </div>
      <div className="prompt-text mono">{block.en}</div>
      <div className="prompt-neg">Negative: {block.negative_en}</div>
    </div>
  );
}

function Steps({ steps }: { steps: Array<Record<string, any>> }) {
  return (
    <div className="trace-list">
      {steps.map((st, i) =>
        st.kind === "reasoning" ? (
          <div className="reasoning-block" key={i}>
            <div className="reasoning-head">
              <span className="step-icon">{STEP_ICON[st.kind] || "·"}</span>
              {st.title || "模型思考过程"}
              {st.elapsed_ms > 0 && <span className="step-time">{st.elapsed_ms}ms</span>}
            </div>
            <div className="step-hint">{STEP_EXPLAIN[st.kind]}</div>
            <div className="reasoning-body">{st.detail}</div>
          </div>
        ) : (
          <div className={`trace-step ${st.ok === false ? "bad" : ""}`} key={i}>
            <span className="step-icon">{STEP_ICON[st.kind] || "·"}</span>
            <div className="step-body">
              <div className="step-title">
                {st.title}
                {st.elapsed_ms > 0 && <span className="step-time">{st.elapsed_ms}ms</span>}
              </div>
              {STEP_EXPLAIN[st.kind] && (
                <div className="step-hint">{STEP_EXPLAIN[st.kind]}</div>
              )}
              {st.detail && <div className="step-detail">{st.detail}</div>}
            </div>
          </div>
        ),
      )}
    </div>
  );
}
