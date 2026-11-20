import { useEffect, useMemo, useRef, useState } from "react";
import "./AgentBigScreen.css";
import Logo from "./Logo";
import { formatElapsed, useTick } from "../elapsed";
import { readMotionMode, subscribeMotionMode, isMotionOff, type MotionMode } from "../motionMode";

type AgentVisualState = "idle" | "waiting" | "working" | "done" | "error" | "cancelled";

interface BigScreenTurn {
  role: string;
  agent: string;
  kind: string;
  content: string;
  payload?: Record<string, any>;
}

interface BigScreenWorker {
  agent: string;
  label?: string;
  started_at?: number;
  seconds?: number;
  progress?: { done?: number; total?: number };
}

interface BigScreenRun {
  run_id: string;
  status: string;
  error?: string;
  messages?: BigScreenTurn[];
  live?: Record<string, Array<Record<string, any>>>;
  working?: { agent?: string; label?: string; started_at?: number; seconds?: number };
  /** 同一时刻在跑的每一个 Agent。分段是并发跑的，`working` 只会留最后一个。 */
  workers?: BigScreenWorker[];
}

interface AgentBigScreenProps {
  run: BigScreenRun | null;
  title?: string;
  historyMode?: boolean;
  onClose: () => void;
}

interface PlanStageRef {
  id?: string;
  label?: string;
}

/** 逐段生产这一步里的一个分段。`stages` 是「环节 id → 状态」的实时表，
 *  后端每推进一步就重发整份计划（pipeline_six 的 push_plan），所以它比
 *  `PlanStep.status` 细得多——后者只有一个 done/failed，它细到每一步。 */
interface PlanSegment {
  id?: string;
  title?: string;
  status?: string;
  stages?: Record<string, string>;
  note?: string;
}

interface PlanStep {
  id?: string;
  title?: string;
  owner?: string;
  output?: string;
  depends?: string;
  status?: string;
  result?: string;
  summary?: string;
  /** 仅 produce 步携带：本步要走的环节表与正在跑的分段。 */
  stages?: PlanStageRef[];
  segments?: PlanSegment[];
  seg_done?: number;
  seg_total?: number;
}

const ROSTER = [
  { id: "director", name: "导演", role: "主 Agent", icon: "导", does: "理解需求、制定计划、调度子 Agent 并汇总结果", aliases: ["director"], input: "用户请求", output: "五个子 Agent", artifact: "任务单" },
  { id: "script_adaptor", name: "内容改编", role: "子 Agent", icon: "改", does: "编剧、史实考据与资料改编", aliases: ["script_adaptor"], input: "导演", output: "剧本分镜", artifact: "剧本稿" },
  { id: "storyboard", name: "剧本分镜", role: "子 Agent", icon: "镜", does: "拆镜、运镜和镜头提示词", aliases: ["storyboard"], input: "内容改编", output: "一致性", artifact: "分镜表" },
  { id: "consistency", name: "一致性", role: "子 Agent", icon: "校", does: "人物、场景、道具和声音资产", aliases: ["consistency"], input: "剧本分镜", output: "视频生成", artifact: "资产包" },
  { id: "video_gen", name: "视频生成", role: "子 Agent", icon: "制", does: "模型适配、视频与声音生成", aliases: ["video_gen"], input: "一致性", output: "质量审查", artifact: "成片" },
  { id: "quality_review", name: "质量审查", role: "子 Agent", icon: "审", does: "剧本、分镜、资产和成片审核", aliases: ["quality_review"], input: "各阶段产出", output: "导演／返工 Agent", artifact: "审查结论" },
] as const;

/** 生产环节 → 流水线上的第几站（与 ROSTER 子 Agent 的顺序对齐）。
 *  顶栏用「阶段 3/5」表示走到哪个 Agent，「剧本改编 2/4」表示这一段走到第几步，
 *  两者编号体系不同，得换算一次。 */
const STAGE_NO: Record<string, number> = {
  script_adaptor: 1,
  storyboard: 2,
  consistency: 3,
  video_gen: 4,
  quality_review: 5,
};

const STATUS_TEXT: Record<AgentVisualState, string> = {
  idle: "待命", waiting: "等待中", working: "执行中", done: "已完成", error: "异常", cancelled: "已取消",
};

const RUN_STATUS: Record<string, string> = {
  queued: "等待中", running: "运行中", waiting: "等待用户回复", done: "已完成", failed: "失败", cancelled: "已取消",
};

const PLAN_STATUS: Record<string, string> = {
  pending: "待开始", running: "执行中", done: "已完成", failed: "未完成", rework: "返工中", not_needed: "不需要",
};

/** 消息 kind 只用来给卡片上色，展开全文时译成人话放在标题旁。
 *  取值来自 pipeline_six 写入的 payload，没列到的一律叫"记录"，不猜。 */
const TIMELINE_KIND: Record<string, string> = {
  segments: "分段", script: "剧本", shots: "分镜", assets: "资产", clips: "成片",
  review: "审查", rework: "返工", error: "异常", plan: "计划", generic: "进展",
};

/** 导演收集资料时的验收位。名字跟后端 `director_search.SLOTS` 对齐，
 *  这里短一截，是因为要塞进 240px 宽的左栏。 */
const SEARCH_SLOTS = [
  { id: "when", label: "何时" },
  { id: "where", label: "何地" },
  { id: "who", label: "何人" },
  { id: "what", label: "发生" },
  { id: "why", label: "为何" },
] as const;

/** 底部「实时调度记录」里的一张卡。key 按消息在列表里的位置算，位置不动，key 就不动。 */
interface TimelineEvent {
  key: string;
  seq: number;
  agent: string;
  kind: string;
  text: string;
}

interface SearchTried {
  query?: string;
  origin?: string;
  found?: number;
  reason?: string;
  error_type?: string;
}

interface SearchPlan {
  tried?: SearchTried[];
  coverage?: { covered?: string[]; missing?: string[]; answered?: number; asked?: number; enough?: boolean };
  rounds?: number;
  in_progress?: boolean;
  surprise?: string;
}

/** Node centres, in percent of the canvas box.
 *
 * The SVG below uses the same percentages (multiplied out to its viewBox), so
 * the wires stay welded to the nodes at every window size. The previous build
 * hard-coded pixel paths against percentage-placed nodes, which is why the
 * lines drifted off the cards as soon as the screen resized.
 */
const NODE_AT: Record<string, { x: number; y: number }> = {
  director: { x: 50, y: 15 },
  script_adaptor: { x: 28, y: 44 },
  storyboard: { x: 72, y: 44 },
  consistency: { x: 20, y: 72 },
  video_gen: { x: 50, y: 72 },
  quality_review: { x: 80, y: 72 },
};

interface TopologyEdge {
  from: string;
  to: string;
  kind: "dispatch" | "flow" | "rework";
  c1?: { x: number; y: number };
  c2?: { x: number; y: number };
  /** Where the artifact tag sits, in percent. Hand-placed to keep it off the cards. */
  tag: { x: number; y: number };
}

/** The real pipeline: 导演派发到起点，五个子 Agent 依次接力，审查结论回流。
 *
 * Nodes are laid out as a Z so the chain reads left→right, then wraps down and
 * back — the same order the bottom stepper lists.
 */
const EDGES: TopologyEdge[] = [
  { from: "director", to: "script_adaptor", kind: "dispatch", c1: { x: 460, y: 190 }, tag: { x: 41, y: 29 } },
  { from: "script_adaptor", to: "storyboard", kind: "flow", tag: { x: 50, y: 44 } },
  { from: "storyboard", to: "consistency", kind: "flow", c1: { x: 620, y: 340 }, tag: { x: 53, y: 56 } },
  { from: "consistency", to: "video_gen", kind: "flow", tag: { x: 35, y: 72 } },
  { from: "video_gen", to: "quality_review", kind: "flow", tag: { x: 65, y: 72 } },
  { from: "quality_review", to: "director", kind: "rework", c1: { x: 980, y: 430 }, c2: { x: 980, y: 200 }, tag: { x: 76, y: 27 } },
];

const BOX_W = 1000;
const BOX_H = 620;

function svgPoint(id: string): { x: number; y: number } {
  const at = NODE_AT[id] ?? { x: 50, y: 50 };
  return { x: (at.x / 100) * BOX_W, y: (at.y / 100) * BOX_H };
}

function edgeGeometry(edge: TopologyEdge): string {
  const a = svgPoint(edge.from);
  const b = svgPoint(edge.to);
  if (edge.c1 && edge.c2) return `M${a.x} ${a.y} C${edge.c1.x} ${edge.c1.y} ${edge.c2.x} ${edge.c2.y} ${b.x} ${b.y}`;
  if (edge.c1) return `M${a.x} ${a.y} Q${edge.c1.x} ${edge.c1.y} ${b.x} ${b.y}`;
  return `M${a.x} ${a.y} L${b.x} ${b.y}`;
}

/** 派发光束的弯折点。导演到五个子 Agent 各一条，手调过弯，避开卡片正中间。 */
const BEAM_BEND: Record<string, { x: number; y: number }> = {
  script_adaptor: { x: 430, y: 180 },
  storyboard: { x: 590, y: 180 },
  consistency: { x: 400, y: 330 },
  video_gen: { x: 500, y: 300 },
  quality_review: { x: 600, y: 330 },
};

/** 导演把任务交出去的那一道光。不是常驻连线——常驻五条会把中栏画成蜘蛛网，
 *  所以只在派发的一瞬间出现，一道亮线从导演长到目标节点，然后淡掉。 */
function beamPath(to: string): string {
  const from = svgPoint("director");
  const target = svgPoint(to);
  const bend = BEAM_BEND[to] || { x: (from.x + target.x) / 2, y: (from.y + target.y) / 2 };
  return `M${from.x} ${from.y} Q${bend.x} ${bend.y} ${target.x} ${target.y}`;
}

/** 导演手上的两件工具。它们不是 Agent（不进 ROSTER、不参与状态机），
 *  只表示"导演在调谁"，所以单独画在导演左右两侧。 */
const TOOLS = [
  { id: "mcp", name: "MCP", note: "外部工具", icon: "M", at: { x: 11, y: 15 } },
  { id: "knowledge", name: "知识库", note: "本地资料", icon: "知", at: { x: 89, y: 15 } },
] as const;

function toolLinkPath(id: string): string {
  const tool = TOOLS.find((item) => item.id === id);
  const from = tool ? { x: (tool.at.x / 100) * BOX_W, y: (tool.at.y / 100) * BOX_H } : { x: 0, y: 0 };
  const to = svgPoint("director");
  return `M${from.x} ${from.y} L${to.x} ${to.y}`;
}

function canonicalAgent(id: string): (typeof ROSTER)[number] | undefined {
  return ROSTER.find((agent) => agent.aliases.some((alias) => alias === id));
}

function latestTurnFor(agentId: string, turns: BigScreenTurn[]): BigScreenTurn | undefined {
  const agent = ROSTER.find((item) => item.id === agentId);
  if (!agent) return undefined;
  for (let index = turns.length - 1; index >= 0; index -= 1) {
    if (agent.aliases.some((alias) => alias === turns[index].agent)) return turns[index];
  }
  return undefined;
}

function cleanText(value: unknown, fallback = ""): string {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  return text || fallback;
}

function truncate(value: string, length: number): string {
  return value.length > length ? `${value.slice(0, length)}…` : value;
}

function FullscreenIcon() {
  return <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M6 2H2v4M10 2h4v4M6 14H2v-4M10 14h4v-4" stroke="currentColor" strokeWidth="1.4" /></svg>;
}

function CloseIcon() {
  return <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="m3 3 10 10M13 3 3 13" stroke="currentColor" strokeWidth="1.4" /></svg>;
}

export default function AgentBigScreen({ run, title, historyMode = false, onClose }: AgentBigScreenProps) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const [selectedAgent, setSelectedAgent] = useState("director");
  /** 底部记录点开的全文（整份快照，不只是 key——理由见 activity 下面那段注释）。 */
  const [openEvent, setOpenEvent] = useState<TimelineEvent | null>(null);
  const turns = run?.messages || [];

  /** 光流动效：偏好由设置页「常规」写入（motionMode.ts），这里只读不写。
   *
   *  系统关了动画（Windows「视觉效果→最佳性能」、无障碍减少动效）时浏览器
   *  会报告 prefers-reduced-motion: reduce；默认模式是「强制开」，所以这台机器
   *  上光流照常跑，除非用户在设置里改成跟随系统或关闭。 */
  const [motionMode, setMotionMode] = useState<MotionMode>(readMotionMode);
  const [systemReduced, setSystemReduced] = useState(() => Boolean(window.matchMedia?.("(prefers-reduced-motion: reduce)").matches));
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = (e: MediaQueryListEvent) => setSystemReduced(e.matches);
    mq.addEventListener("change", onChange);
    const unsubscribe = subscribeMotionMode(setMotionMode);
    return () => {
      mq.removeEventListener("change", onChange);
      unsubscribe();
    };
  }, []);
  const motionOff = isMotionOff(motionMode, systemReduced);

  const activeIds = useMemo(() => {
    const ids = new Set<string>();
    Object.entries(run?.live || {}).forEach(([agent, steps]) => {
      if (steps?.length) ids.add(canonicalAgent(agent)?.id || agent);
    });
    if (run?.working?.agent) ids.add(canonicalAgent(run.working.agent)?.id || run.working.agent);
    return ids;
  }, [run?.live, run?.working?.agent]);

  /** What each busy agent is doing right now, from the streaming trace. */
  const liveAction = useMemo(() => {
    const map = new Map<string, string>();
    Object.entries(run?.live || {}).forEach(([agent, steps]) => {
      const last = steps?.[steps.length - 1];
      if (!last) return;
      map.set(canonicalAgent(agent)?.id || agent, cleanText(last.title, cleanText(last.detail, "")));
    });
    return map;
  }, [run?.live]);

  const agentStates = useMemo(() => {
    const states = new Map<string, AgentVisualState>();
    for (const agent of ROSTER) {
      const latest = latestTurnFor(agent.id, turns);
      let state: AgentVisualState = "waiting";
      if (!run) state = "idle";
      else if (run.status === "cancelled") state = "cancelled";
      else if (activeIds.has(agent.id)) state = "working";
      else if (latest && ["error", "rework"].includes(latest.kind)) state = "error";
      else if (latest?.kind === "review" && latest.payload?.approved === false) state = "error";
      // 导演在跑就等于在编排：光凭"最后一条消息是计划"判定完成，会让它在
      // 整个运行期间顶着"已完成"，而这恰好是最该表现"正在调度"的位置。
      else if (agent.id === "director" && ["queued", "running", "waiting"].includes(run.status)) state = "working";
      else if (latest) state = "done";
      else if (agent.id === "director" && run.status === "failed") state = "error";
      else if (agent.id === "director" && run.status === "done") state = "done";
      states.set(agent.id, state);
    }
    return states;
  }, [activeIds, run, turns]);

  /** 并发在跑的每一个 Agent，各自一条从导演拉出来的金光。
   *  导演自己不算——它那条是"我在干活"，不是"我把活交出去"。 */
  const dispatchIds = useMemo(
    () => ROSTER.filter((agent) => agent.id !== "director" && agentStates.get(agent.id) === "working").map((agent) => agent.id),
    [agentStates],
  );

  /** 当前在跑的每一行。后端按 agent 记账，分段并发时会同时出现好几个。 */
  const workers = useMemo<BigScreenWorker[]>(() => {
    const list = run?.workers?.length ? run.workers : run?.working?.agent ? [run.working] : [];
    return list.filter((worker) => worker?.agent);
  }, [run?.workers, run?.working]);

  const nowSec = useTick(workers.length > 0);

  const plan = useMemo(() => {
    for (let index = turns.length - 1; index >= 0; index -= 1) {
      if (turns[index].kind === "plan") return turns[index].payload || {};
    }
    return {} as Record<string, any>;
  }, [turns]);
  const planSteps: PlanStep[] = Array.isArray(plan.steps) ? plan.steps : [];

  /** 导演检索时的账本。后端每查完一个词就原地覆写一条消息，这里取最后一份。 */
  const searchPlan = useMemo<SearchPlan | null>(() => {
    for (let index = turns.length - 1; index >= 0; index -= 1) {
      const found = turns[index].payload?.search_plan;
      if (found) return found as SearchPlan;
    }
    return null;
  }, [turns]);
  const triedQueries = searchPlan?.tried || [];
  const coveredSlots = searchPlan?.coverage?.covered || [];
  /** 收集资料这一段：有计划、还没排出任务队列、且计划仍标着进行中。 */
  const collecting = Boolean(searchPlan?.in_progress) && planSteps.length === 0;

  const selected = ROSTER.find((agent) => agent.id === selectedAgent) || ROSTER[0];
  const selectedState = agentStates.get(selected.id) || "idle";
  const children = ROSTER.slice(1);
  const counts = {
    working: children.filter((agent) => agentStates.get(agent.id) === "working").length,
    waiting: children.filter((agent) => ["waiting", "idle"].includes(agentStates.get(agent.id) || "idle")).length,
    done: children.filter((agent) => agentStates.get(agent.id) === "done").length,
    error: children.filter((agent) => agentStates.get(agent.id) === "error").length,
  };
  const completed = ROSTER.filter((agent) => agentStates.get(agent.id) === "done").length;
  const totalPlan = Number(plan.total) || planSteps.length;
  const donePlan = Number(plan.done) || planSteps.filter((step) => step.status === "done").length;

  /** 「逐段生产」这一步的切片：环节表 + 每个分段走到哪一格。 */
  const produce = useMemo(() => {
    const step = planSteps.find((item) => String(item.id) === "produce");
    return {
      step,
      stages: Array.isArray(step?.stages) ? step?.stages || [] : [],
      segments: Array.isArray(step?.segments) ? step?.segments || [] : [],
    };
  }, [planSteps]);

  /** 整条流水线走到百分之多少。
   *
   *  上一版是 plan.done / plan.total，而计划只有三步（取材 / 分段 / 逐段生产）：
   *  任务一进生产就固定在 67%，跑完才跳到 100%。而生产占掉整条流水线绝大部分
   *  时间——等于进度条在真正干活的那一段是死的。
   *
   *  这里改成数格子：取材一格、分段一格、生产的「每一段 × 每个环节」各一格。
   *  格子状态取自计划消息里的 segments[].stages，后端每推进一步重发一次计划，
   *  所以它随运行推进；正在跑的那格算半格，长环节里条子也有动静。格子数是数
   *  出来的，不做时长加权——权重是估的，格子是实的。
   */
  const pipeline = useMemo(() => {
    const stageCount = produce.stages.length || 4;
    // 生产还没列出分段时先按一段估：否则前两步走完就顶到 100%，而生产还没开始。
    const segCount = produce.segments.length || (produce.step ? 1 : 0);
    const total = 2 + segCount * stageCount;
    const credit = (status?: string): number =>
      status === "done" || status === "not_needed" ? 1 : status === "running" || status === "rework" ? 0.5 : 0;
    let done = 0;
    if (planSteps.length) {
      done += credit(planSteps.find((item) => String(item.id) === "material")?.status);
      done += credit(planSteps.find((item) => String(item.id) === "segment")?.status);
      for (const seg of produce.segments) {
        if (produce.stages.length) {
          for (const stage of produce.stages) done += credit(seg.stages?.[String(stage.id)]);
        } else {
          // 老格式的计划没有环节表，退回按整段计。
          done += credit(seg.status) * stageCount;
        }
      }
    }
    return { done, total, stageCount, pct: total > 0 ? Math.round((done / total) * 100) : 0 };
  }, [planSteps, produce]);

  const overall = run?.status === "done"
    ? 100
    : planSteps.length
      // 还没跑完就不给 100：最后一格是成片合成，它不在计划清单里。
      ? Math.min(99, pipeline.pct)
      : Math.round((completed / ROSTER.length) * 100);

  /** 生产正停在哪个环节。顶栏用它比「阶段 4/5」具体：后者只说第几个 Agent，
   *  前者说这一段的第几步。 */
  const produceCursor = useMemo(() => {
    for (const seg of produce.segments) {
      const map = seg.stages || {};
      for (let index = 0; index < produce.stages.length; index += 1) {
        const stage = produce.stages[index];
        const state = map[String(stage.id)];
        if (state === "running" || state === "rework") {
          return {
            seg: seg.id || "",
            id: String(stage.id),
            label: stage.label || String(stage.id),
            index: index + 1,
            total: produce.stages.length,
          };
        }
      }
    }
    return null;
  }, [produce]);

  /** 失败时停在哪个环节。只在 run 真的失败时算：计划可能先在合成阶段被判 failed，
   *  而任务仍在跑（成片合成不在计划清单里），那时说「中断于」是错的。
   *
   *  先找标成 failed 的环节；没有的话取这一段的第一个没走完的环节——审查不过被退回
   *  时，环节上是 `rework`，不是 `failed`，只认 failed 会让徽标退回「第几阶段」，
   *  反而看不出卡在哪。 */
  const failedAt = useMemo(() => {
    if (run?.status !== "failed") return null;
    let firstBlocked: { seg: string; label: string } | null = null;
    for (const seg of produce.segments) {
      const map = seg.stages || {};
      for (let index = 0; index < produce.stages.length; index += 1) {
        const stage = produce.stages[index];
        const state = map[String(stage.id)];
        const label = stage.label || String(stage.id);
        if (state === "failed") return { seg: seg.id || "", label };
        if (!firstBlocked && state && state !== "done") firstBlocked = { seg: seg.id || "", label };
      }
    }
    return firstBlocked;
  }, [produce, run?.status]);

  /** Which of the five stages the run is standing on — the answer to "到哪一步了". */
  const pendingIndex = children.findIndex((agent) => agentStates.get(agent.id) !== "done");
  const stageNo = pendingIndex === -1 ? children.length : pendingIndex + 1;

  /** 顶栏那枚阶段徽标。收集资料时是检索账目，生产时是「第几段的第几步」，失败时
   *  说明停在哪里——「阶段 5/5」配一个「失败」读起来像全部做完才失败。 */
  /** 生产已经收尾、任务还在跑：这时只剩成片合成，而合成不在计划清单里
   *  （计划自己那条 beyond 说明写着）。不这么说，徽标会退回「阶段 2/5」——
   *  那是按消息倒推的，生产跑完之后它就不准了。 */
  const finishing = Boolean(produce.step)
    && ["done", "failed"].includes(String(produce.step?.status || ""))
    && ["queued", "running", "waiting"].includes(String(run?.status || ""));

  const stageBadge = !run
    ? ""
    : collecting
      ? `资料收集 · 已查 ${triedQueries.length} 个词`
      : run.status === "failed"
        ? failedAt
          ? `中断于 ${failedAt.label}`
          : `中断于 第 ${stageNo} 阶段`
        : finishing
          ? "收尾 · 成片合成"
          : produceCursor
            ? `阶段 ${STAGE_NO[produceCursor.id] || stageNo}/${children.length} · ${produceCursor.label} ${produceCursor.index}/${produceCursor.total}`
            : `阶段 ${stageNo}/${children.length}`;

  const screenTitle = title?.trim() || "等待新任务";
  const directorTurns = turns.filter((turn) => turn.agent === "director" && turn.kind !== "plan");
  /** 决策全文，不截断：左卡正是要"看完"的地方。上一版在这里砍到 92 字，
   *  屏幕上看起来就像被谁裁掉了半句。 */
  const directorDecision = cleanText(directorTurns[directorTurns.length - 1]?.content, run ? "导演正在分析任务并等待子 Agent 返回" : "等待用户提交新任务");
  /** 顶栏那一行只有 40 来个汉字的宽度，节点提示也只有两行，它们用截断版。 */
  const directorDecisionBrief = truncate(directorDecision, 92);
  const currentAgent = canonicalAgent(run?.working?.agent || "");
  const currentWork = cleanText(run?.working?.label, currentAgent ? currentAgent.does : run ? "正在推进任务" : "尚未开始");

  /** 导演正在调哪个工具。工具名只出现在 trace 的 title 里，所以按关键词认：
   *  知识库有专属名字，剩下的外部调用一律算 MCP。 */
  const activeTool = useMemo<"mcp" | "knowledge" | null>(() => {
    if (canonicalAgent(run?.working?.agent || "")?.id !== "director") return null;
    const steps = run?.live?.director || [];
    for (let index = steps.length - 1; index >= 0; index -= 1) {
      const text = `${steps[index]?.title || ""} ${steps[index]?.detail || ""}`;
      if (/search_knowledge|查知识库/.test(text)) return "knowledge";
      if (/search_web|fetch_web_page|search_feishu|fetch_feishu|联网|查飞书/i.test(text)) return "mcp";
    }
    return null;
  }, [run?.live, run?.working?.agent]);

  const elapsedOf = (worker: BigScreenWorker): number => {
    const started = Number(worker.started_at) || 0;
    return started > 0 ? Math.max(0, Math.floor(nowSec - started)) : Number(worker.seconds) || 0;
  };

  /** 某个 Agent 自己的计时。并发时每个节点各走各的秒。 */
  const timerFor = (agentId: string): string => {
    const worker = workers.find((item) => (canonicalAgent(item.agent)?.id || item.agent) === agentId);
    return worker ? formatElapsed(elapsedOf(worker)) : "";
  };

  /** 每一行自己的进度——这才叫"子任务的进度"。
   *
   *  资料收集阶段的总量是五个验收位，覆盖率是程序算的；子 Agent 用后端写进
   *  心跳的阶段位置（第几步 / 共几步）。两样都没有就不给数字，宁可留空也不编。 */
  const progressOf = (worker: BigScreenWorker): { pct: number; text: string } | null => {
    const agent = canonicalAgent(worker.agent);
    if (agent?.id === "director" && collecting) {
      return {
        pct: Math.round((coveredSlots.length / SEARCH_SLOTS.length) * 100),
        text: `覆盖 ${coveredSlots.length}/${SEARCH_SLOTS.length}`,
      };
    }
    const done = Number(worker.progress?.done);
    const total = Number(worker.progress?.total);
    if (total > 0) return { pct: Math.min(100, Math.round((done / total) * 100)), text: `${done}/${total}` };
    return null;
  };

  const activity = useMemo(() => {
    /** 只挡掉用户消息和计划快照，其余全留。
     *
     *  原来这里取「最后 12 条」：每来一条新消息，窗口就整体左移一格，正在看的
     *  卡片被挤出窗口，连点开的全文浮层也一起消失（浮层是按 key 在这份列表里
     *  找的，找不到就不渲染）。现在不裁窗口——卡片按序排，容器横向滚动，新消息
     *  只往右追加，不再动已有的卡。 */
    const useful = turns.filter((turn) => turn.role !== "user" && turn.kind !== "plan");
    return useful.map((turn, index) => ({
      key: `${index}:${turn.agent}:${turn.kind}`,
      seq: index + 1,
      agent: canonicalAgent(turn.agent)?.name || turn.agent || "系统",
      kind: turn.kind,
      text: cleanText(turn.content, "已更新任务状态"),
    }));
  }, [turns]);

  /** 点开的那条记录存整份快照（state 在组件顶部）。只存 key 的写法有个后患：
   *  那条消息一旦离开窗口（或 kind 被后续改写），`activity.find` 就查不到，浮层
   *  会在下一次轮询时无声关闭——用户正读着的一段文字突然没了，看起来像「刷新把
   *  东西刷掉了」。 */

  /** 列表里同一条消息被后续轮询改写（追加了几个字）时，跟着更新浮层里的文字。 */
  useEffect(() => {
    if (!openEvent) return;
    const fresh = activity.find((event) => event.key === openEvent.key);
    if (fresh && fresh.text !== openEvent.text) setOpenEvent(fresh);
  }, [activity, openEvent]);

  /** 底栏横向滚动：只有用户本来就停在最右端时才自动跟到最新一条。
   *  否则他往回翻旧记录，新消息一来就被拽走，等于看不住。 */
  const timelineRef = useRef<HTMLDivElement | null>(null);
  const timelinePinned = useRef(true);
  useEffect(() => {
    const el = timelineRef.current;
    if (el && timelinePinned.current) el.scrollLeft = el.scrollWidth;
  }, [activity.length]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || document.fullscreenElement) return;
      // Esc 先收浮层，再关大屏——否则读全文时想退出，一下把整个大屏关了。
      if (openEvent) {
        setOpenEvent(null);
        return;
      }
      onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, openEvent]);

  async function toggleFullscreen() {
    if (!document.fullscreenElement) await rootRef.current?.requestFullscreen?.();
    else await document.exitFullscreen?.();
  }

  function taskFor(agentId: string): string {
    const state = agentStates.get(agentId) || "idle";
    const latest = latestTurnFor(agentId, turns);
    if (state === "working") {
      const live = liveAction.get(agentId);
      if (live) return truncate(live, 54);
      if (run?.working?.agent && canonicalAgent(run.working.agent)?.id === agentId) return truncate(currentWork, 54);
      if (agentId !== "director") return "正在执行";
    }
    if (state === "error") return truncate(cleanText(latest?.content, run?.error || "等待导演重新调度"), 54);
    if (state === "done") return truncate(cleanText(latest?.content, "结果已经回传导演"), 54);
    if (state === "cancelled") return "本次任务已取消";
    return agentId === "director" ? directorDecisionBrief : "等待前置结果或导演派发";
  }

  /** 有没有人在干活。不能只看子 Agent——导演自己检索时没有子 Agent 在跑，
   *  但计时器必须走，否则大屏在最长的一段等待里显示 --:--。 */
  const anyWorking = workers.length > 0 || counts.working > 0;
  /** 顶栏那一个"当前耗时"取最久的那个：并发时它是这一批里跑得最长的时间。 */
  const longestRunning = workers.reduce((max, worker) => Math.max(max, elapsedOf(worker)), 0);
  const timerText = anyWorking ? formatElapsed(longestRunning) : "";

  return (
    <div className={`agent-screen-backdrop${motionOff ? " motion-off" : ""}`} role="dialog" aria-modal="true" aria-label="多 Agent 协作大屏">
      <div className="agent-screen" ref={rootRef}>
        <header className="agent-screen-head">
          <div className="agent-screen-brand"><Logo size={44} /><div><h2>多 Agent 调度大屏</h2><p>编排 · 调度 · 执行 · 回传 · 返工</p></div></div>
          <div className="agent-screen-task">
            <div className="agent-screen-task-title">
              <strong>{screenTitle}</strong>
              <div className="task-badges">
                <span className={`task-status run-${run?.status || "idle"}`}>{historyMode ? "历史回放" : run ? RUN_STATUS[run.status] || run.status : "待命"}</span>
                {run ? <span className="task-stage">{stageBadge}</span> : null}
              </div>
            </div>
            <div className={`agent-screen-progress ${run?.status === "failed" ? "is-failed" : ""} ${run?.status === "done" ? "is-done" : ""}`}><span>全部</span><i><b style={{ width: `${overall}%` }} /></i><em>{overall}%</em></div>
          </div>
          <div className="agent-screen-actions"><button className="btn ghost small" onClick={toggleFullscreen}><FullscreenIcon />全屏</button><button className="btn ghost small" onClick={onClose}><CloseIcon />关闭</button></div>
        </header>

        <div className="agent-screen-metrics">
          <span><b>{workers.length || counts.working}</b> 并发执行</span><span><b>{counts.waiting}</b> 等待调度</span><span><b>{counts.done}</b> 已完成</span><span className={counts.error ? "metric-error" : ""}><b>{counts.error}</b> 异常／返工</span>
          <span className={`metric-timer ${timerText ? "is-live" : ""}`}><b>{timerText || "--:--"}</b> 当前耗时</span>
          <span className="metric-current">当前：{currentAgent ? `${currentAgent.name} · ${currentWork}` : run ? directorDecisionBrief : "等待任务"}</span>
        </div>

        <div className="agent-screen-body">
          <aside className="current-work-panel">
            <div className="screen-section-head"><strong>当前工作情况</strong><span>任务视角</span></div>
            <div className="current-work-content">
              <div className="work-status-card">
                <div className="work-status-head">
                  <span className={`run-state run-${run?.status || "idle"}`}>{run ? RUN_STATUS[run.status] || run.status : "待命"}</span>
                  <span className="work-scope">当前子任务</span>
                  {timerText ? <span className="work-timer">{timerText}</span> : null}
                </div>
                <h3>{truncate(screenTitle, 42)}</h3>
                {/* 有子任务就列出来，没有就把「正在推进什么」放同一块地方。
                    原来这两态是两个不同的元素（列表 ↔ 段落），每换一次 React 就
                    重建一次，卡片高度也跟着跳；设备在轮询里来回切的时候，下面那些
                    卡片会被顶着上下动，看着像被"刷掉"了。 */}
                <div className={`worker-list ${workers.length ? "" : "is-empty"}`}>
                  {workers.length ? workers.map((worker) => {
                    const agent = canonicalAgent(worker.agent);
                    const progress = progressOf(worker);
                    return (
                      <div className="worker-row" key={worker.agent}>
                        <span className="worker-icon">{agent?.icon || "·"}</span>
                        <span className="worker-copy">
                          <strong>{agent?.name || worker.agent}</strong>
                          <small>{worker.label || agent?.does || ""}</small>
                        </span>
                        <span className="worker-meta">
                          {formatElapsed(elapsedOf(worker))}
                          {progress ? <em>{progress.text}</em> : null}
                        </span>
                        <i className={`worker-bar ${progress ? "" : "is-indeterminate"}`}>
                          <b style={{ width: `${progress ? progress.pct : 40}%` }} />
                        </i>
                      </div>
                    );
                  }) : <p className="worker-idle">{currentWork}</p>}
                </div>
              </div>
              {/* 运行卡钉在顶部（谁在跑、跑多久一直要看），下面这一段独立滚动：
                  决策全文、统计、资料收集台账都可能比面板高，不能靠截断应付。 */}
              <div className="current-work-scroll">
              <div className="director-decision-card"><div><span>导</span><strong>导演当前决策</strong></div><p>{directorDecision}</p></div>
              <div className="work-count-grid"><div><b>{counts.working}</b><span>执行中</span></div><div><b>{counts.waiting}</b><span>等待中</span></div><div><b>{counts.done}</b><span>已完成</span></div><div className={counts.error ? "has-error" : ""}><b>{counts.error}</b><span>异常</span></div></div>
              <div className="plan-queue">
                {searchPlan && !planSteps.length ? (
                  <div className="search-ledger">
                    <div className="subsection-title">
                      <strong>资料收集</strong>
                      <span>{searchPlan.rounds ? `第 ${searchPlan.rounds} 轮 · ` : ""}已查 {triedQueries.length} 个词</span>
                    </div>
                    <div className="coverage-row">
                      {SEARCH_SLOTS.map((slot) => (
                        <span className={`coverage-chip ${coveredSlots.includes(slot.id) ? "is-covered" : ""}`} key={slot.id}>{slot.label}</span>
                      ))}
                    </div>
                    <div className="ledger-list">
                      {triedQueries.map((item, index) => {
                        const hits = Number(item.found) || 0;
                        const outcome = item.error_type ? "fail" : hits > 0 ? "hit" : "miss";
                        return (
                          <div className={`ledger-row ledger-${outcome}`} key={`${item.query || index}`} title={item.reason || ""}>
                            <span className="ledger-mark">{outcome === "fail" ? "!" : outcome === "hit" ? "✓" : "×"}</span>
                            <span className="ledger-query">{item.query}</span>
                            <em>{outcome === "fail" ? "没查成" : outcome === "hit" ? `命中 ${hits} 段` : "未命中"}</em>
                          </div>
                        );
                      })}
                      {searchPlan.in_progress ? <div className="ledger-row ledger-pending"><span className="ledger-mark" /><span className="ledger-query">正在查下一个词…</span></div> : null}
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="subsection-title"><strong>导演任务编排</strong><span>{donePlan}/{totalPlan || 0}</span></div>
                    {planSteps.length ? (
                      <div className="queue-list">
                        {planSteps.slice(0, 6).map((step, index) => (
                          <div className={`queue-step queue-${step.status || "pending"}`} key={step.id || index} title={[step.title, step.summary || step.result].filter(Boolean).join(" — ")}>
                            <span className="queue-index">{String(index + 1).padStart(2, "0")}</span>
                            <span className="queue-copy">
                              <strong>{step.title || "未命名任务"}</strong>
                              <small>{step.owner || "导演"}{step.depends ? ` · 等待 ${step.depends}` : ""}{step.output ? ` · 产出 ${step.output}` : ""}</small>
                            </span>
                            <em>{PLAN_STATUS[step.status || "pending"] || step.status}</em>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="queue-preview">
                        <p>导演还没有生成计划。提交需求后会按这个顺序展开：</p>
                        <ul>
                          <li><span>01</span><div><strong>拆解需求</strong><small>产出任务单与验收标准</small></div></li>
                          <li><span>02</span><div><strong>子 Agent 依次执行</strong><small>改编 → 分镜 → 资产 → 成片</small></div></li>
                          <li><span>03</span><div><strong>结果回传汇总</strong><small>每个产物落到导演手里</small></div></li>
                          <li><span>04</span><div><strong>审查不过则返工</strong><small>退回对应环节重做</small></div></li>
                        </ul>
                      </div>
                    )}
                  </>
                )}
              </div>
              {run?.error && <div className="current-blocker"><strong>当前阻塞</strong><p>{run.error}</p></div>}
              </div>
            </div>
          </aside>

          <section className="topology-panel">
            <div className="screen-section-head"><strong>Agent 调度拓扑</strong><span>实线：调度与接续　虚线：返工回流</span></div>
            <div className="topology-canvas">
              <svg className="topology-lines" viewBox={`0 0 ${BOX_W} ${BOX_H}`} preserveAspectRatio="none" aria-hidden="true">
                <defs>
                  <marker id="dispatch-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" /></marker>
                  <marker id="flow-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" /></marker>
                </defs>
                {TOOLS.map((tool) => (
                  <g key={`link-${tool.id}`}>
                    <path className={`tool-link ${activeTool === tool.id ? "is-active" : ""}`} d={toolLinkPath(tool.id)} />
                    {activeTool === tool.id && (
                      <circle className="task-packet" r="6"><animateMotion dur="1.2s" repeatCount="indefinite" path={toolLinkPath(tool.id)} /></circle>
                    )}
                  </g>
                ))}
                {EDGES.map((edge, edgeIndex) => {
                  const d = edgeGeometry(edge);
                  const targetState = agentStates.get(edge.to) || "idle";
                  const state = edge.kind === "rework" ? (counts.error > 0 ? "error" : "idle") : targetState;
                  /** 线上跑的光。原来是一个 `<circle>` + animateMotion，跟着 viewBox
                   *  一起被非等比拉伸，圆点压成扁椭圆，速度还得按像素猜；现在换成
                   *  沿路径走的光段，用 stroke-dashoffset 推着走。
                   *
                   *  dash 的长度单位是 viewBox 单位（1000×620），不是像素——Chromium
                   *  不会把 CSS 里写的 dash 值按 pathLength 归一化（那是 SVG 属性才有的
                   *  待遇），所以这里一律给绝对数值，每种状态的周期都能整除位移量 -120，
                   *  循环才不会每圈"跳"一下。
                   *
                   *  强弱按目的地状态分：执行中最亮最快（一深一浅的米白段在金色底线上
                   *  跑，一眼就是"正在流"），已完成换绿慢一档，没轮到的最淡最慢——它在
                   *  说"管子通着"，不是在说"正在流"。
                   *
                   *  返工线不铺光段：它本身就是会动的红色虚线，再压一层光流只是互相打
                   *  架；这里只留一条线，动效归它自己。
                   *
                   *  导演到某个节点已经在跑光束时就跳过这一条：同一段走廊上叠两束
                   *  不同速度的光，看着像打架。 */
                  const flowState = edge.kind === "rework" ? null : state;
                  const beamCovered = edge.kind === "dispatch" && dispatchIds.includes(edge.to);
                  const flow = flowState && flowState !== "cancelled" && !beamCovered ? flowState : null;
                  return (
                    <g key={`${edge.from}-${edge.to}`}>
                      <path className={`topology-edge edge-${edge.kind} line-${state}`} d={d} markerEnd={`url(#${edge.kind === "rework" ? "flow-arrow" : "dispatch-arrow"})`} />
                      {flow && <path className={`edge-stream stream-${flow}`} d={d} style={{ animationDelay: `${-(edgeIndex * 0.42).toFixed(2)}s` }} />}
                    </g>
                  );
                })}
                {dispatchIds.map((id) => {
                  const path = beamPath(id);
                  return (
                    <g className="dispatch-link" key={`dispatch-${id}`}>
                      {/* 不用 pathLength：光束的 dash 走 CSS，Chromium 不会给它做归一化，
                          写了反而让 58/242 这种尺寸被理解成"整条线长 1"。 */}
                      <path className="dispatch-glow" d={path} />
                      <path className="dispatch-track" d={path} />
                      {/* 一对反向光流，读作"一段正在进行的对话"：金色长条从导演流向
                          子 Agent（派活、追问），浅绿短节从子 Agent 流回导演（回话、
                          交产物）。两束周期不同，才不会像机械往返。 */}
                      <path className="dispatch-comet" d={path} />
                      <path className="dispatch-reply" d={path} />
                    </g>
                  );
                })}
              </svg>
              {EDGES.map((edge) => {
                const source = ROSTER.find((agent) => agent.id === edge.from);
                return <span className={`edge-artifact artifact-${edge.kind}`} style={{ left: `${edge.tag.x}%`, top: `${edge.tag.y}%` }} key={`tag-${edge.from}-${edge.to}`}>{source?.artifact}</span>;
              })}
              <AgentNode agent={ROSTER[0]} state={agentStates.get("director") || "idle"} task={taskFor("director")} selected={selectedAgent === "director"} onSelect={setSelectedAgent} at={NODE_AT.director} timer={timerFor("director")} collecting={collecting} main />
              {children.map((agent) => (
                <AgentNode key={agent.id} agent={agent} state={agentStates.get(agent.id) || "idle"} task={taskFor(agent.id)} selected={selectedAgent === agent.id} onSelect={setSelectedAgent} at={NODE_AT[agent.id] || { x: 50, y: 50 }} timer={timerFor(agent.id)} />
              ))}
              {TOOLS.map((tool) => (
                <div
                  className={`tool-node ${activeTool === tool.id ? "is-active" : ""}`}
                  style={{ top: `${tool.at.y}%`, left: `${tool.at.x}%` }}
                  key={`tool-${tool.id}`}
                  title={`${tool.name} · ${tool.note}｜${activeTool === tool.id ? "导演正在调用" : "空闲"}`}
                >
                  <span className="tool-icon">{tool.icon}</span>
                  <span className="tool-copy">
                    <strong>{tool.name}</strong>
                    <small>{activeTool === tool.id ? "调用中" : tool.note}</small>
                  </span>
                </div>
              ))}
              <div className="stage-strip">
                {children.map((agent) => {
                  const state = agentStates.get(agent.id) || "idle";
                  const isCurrent = currentAgent?.id === agent.id;
                  return (
                    <div className={`stage-step stage-${state} ${isCurrent ? "is-current" : ""}`} key={agent.id}>
                      <i />
                      <span>{agent.name}</span>
                      {timerFor(agent.id) ? <em>{timerFor(agent.id)}</em> : null}
                    </div>
                  );
                })}
              </div>
            </div>
          </section>

          <aside className="agent-status-board">
            <div className="screen-section-head"><strong>Agent 实时执行</strong><span>{counts.working ? `${counts.working} 个执行中` : "全员待命"}</span></div>
            <div className="status-agent-list">
              {ROSTER.map((agent) => {
                const state = agentStates.get(agent.id) || "idle";
                return (
                  <button className={`status-agent state-${state} ${agent.id === "director" ? "master" : ""} ${selectedAgent === agent.id ? "selected" : ""}`} key={agent.id} onClick={() => setSelectedAgent(agent.id)} title={taskFor(agent.id)}>
                    <span className="status-agent-icon">{agent.icon}</span>
                    <span className="status-agent-copy"><strong>{agent.name}<em>{agent.role}</em></strong><small>{taskFor(agent.id)}</small></span>
                    <span className={`status-agent-state state-${state}`}>{STATUS_TEXT[state]}{timerFor(agent.id) ? <em>{timerFor(agent.id)}</em> : null}</span>
                  </button>
                );
              })}
            </div>
            <div className="selected-agent-detail">
              <div className="selected-head"><span>{selected.icon}</span><strong>{selected.name} · {selected.role}</strong><em className={`state-${selectedState}`}>{STATUS_TEXT[selectedState]}</em></div>
              <p>{selected.does}</p>
              <dl>
                <dt>输入来自</dt><dd>{selected.input}</dd>
                <dt>预计交给</dt><dd>{selected.output}</dd>
                <dt>交付产物</dt><dd>{selected.artifact}</dd>
                <dt>当前工作</dt><dd>{taskFor(selected.id)}</dd>
                {liveAction.get(selected.id) && <><dt>实时动作</dt><dd className="detail-live">{liveAction.get(selected.id)}</dd></>}
                {timerFor(selected.id) ? <><dt>已运行</dt><dd className="detail-live">{timerFor(selected.id)}</dd></> : null}
              </dl>
            </div>
          </aside>
        </div>

        <footer className="dispatch-timeline">
          <div className="timeline-title"><strong>实时调度记录</strong><span>点任一条看全文</span></div>
          <div
            className="timeline-events"
            ref={timelineRef}
            onScroll={(event) => {
              const el = event.currentTarget;
              timelinePinned.current = el.scrollLeft + el.clientWidth >= el.scrollWidth - 32;
            }}
          >
            {activity.length ? activity.map((event) => (
              <button
                type="button"
                className={`timeline-event event-${event.kind} ${openEvent?.key === event.key ? "is-open" : ""}`}
                key={event.key}
                title={event.text}
                onClick={() => setOpenEvent(openEvent?.key === event.key ? null : event)}
              >
                <span>{String(event.seq).padStart(2, "0")}</span>
                <strong>{event.agent}</strong>
                <p>{event.text}</p>
              </button>
            )) : (
              <div className="timeline-empty">
                <strong>还没有调度记录</strong>
                <small>任务开始后，导演派发、Agent 执行、结果回传和返工都会按顺序落在这里。</small>
              </div>
            )}
          </div>
          {openEvent ? (
            <div className="timeline-detail" role="dialog" aria-label="调度记录全文">
              <div className="timeline-detail-head">
                <span>{String(openEvent.seq).padStart(2, "0")}</span>
                <strong>{openEvent.agent}</strong>
                <em>{TIMELINE_KIND[openEvent.kind] || "记录"}</em>
                <button type="button" className="btn ghost small" onClick={() => setOpenEvent(null)}>收起</button>
              </div>
              <p>{openEvent.text}</p>
            </div>
          ) : null}
        </footer>
      </div>
    </div>
  );
}

function AgentNode({ agent, state, task, selected, onSelect, at, timer, collecting = false, main = false }: {
  agent: (typeof ROSTER)[number];
  state: AgentVisualState;
  task: string;
  selected: boolean;
  onSelect: (id: string) => void;
  at: { x: number; y: number };
  timer: string;
  collecting?: boolean;
  main?: boolean;
}) {
  return (
    <button
      className={`topology-node ${main ? "main-agent" : "child-agent"} state-${state} ${selected ? "selected" : ""}`}
      style={{ top: `${at.y}%`, left: `${at.x}%` }}
      onClick={() => onSelect(agent.id)}
      title={`${agent.name} · ${STATUS_TEXT[state]}｜${task}`}
    >
      <span className="node-icon">{agent.icon}</span>
      <span className="node-content"><strong>{agent.name}</strong><em>{agent.role}</em><small>{task}</small></span>
      <span className="node-meta">
        <span className="node-state">{collecting ? <span className="node-spinner" /> : <i />}{collecting ? "收集中" : STATUS_TEXT[state]}</span>
        {state === "working" && timer ? <span className="node-timer">{timer}</span> : null}
      </span>
    </button>
  );
}
