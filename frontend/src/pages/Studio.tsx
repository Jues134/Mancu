import { useCallback, useEffect, useRef, useState, type PointerEvent } from "react";
import TurnRow, { AGENT_ABOUT, STEP_EXPLAIN, STEP_ICON } from "../components/TurnRow";
import AgentBigScreen from "../components/AgentBigScreen";
import { PlanDock, PlanDockButton } from "../components/PlanDock";
import { type PlanCardData } from "../components/PlanCard";
import { agentInitial } from "../agentLabel";
import Logo from "../components/Logo";
import Settings from "./Settings";
import { targetSecondsFor } from "../duration";
import { formatElapsed, useElapsedSeconds } from "../elapsed";
import "../styles.css";

/**
 * Documentary workbench — the six-agent roster.
 *
 * The left column is the *roster*, not a task list. Six agents exist; each shows
 * one of four states (待命 / 工作中 / 完成 / 返工). A run dispatches work to
 * whichever agents it needs, so "at once" is visible as several segments' rows lit
 * up — which a task-card list obscured, because one agent's parallel shards looked
 * like several different agents.
 *
 * The centre is the conversation: agent utterances in order, with the director's
 * dispatches and the review rejections interleaved. Rejections are shown inline
 * rather than in a log, because the self-loop is the thing this project claims and
 * it should be visible where it happens.
 *
 * Voice and composition are internal steps of the video path rather than
 * separate agents.
 */

/**
 * The six-agent roster, as the production spec defines it.
 *
 * `media` marks the two stages that can render output as well as write a prompt:
 * the consistency stage renders portraits, the video stage renders shots. `paused`
 * is gone — every agent in this roster is wired. Voice and composition are
 * internal steps of the video path now, not separate agents, so they no longer
 * appear here.
 */
const AGENTS: Array<{
  id: string;
  name: string;
  does: string;
  group: string;
  media?: "images" | "videos";
}> = [
  { id: "director", name: "导演", does: "理解需求、制定计划、调度、协调返工", group: "调度" },
  { id: "script_adaptor", name: "内容改编", does: "编剧 + 历史考据 + 资料改编", group: "准备" },
  { id: "storyboard", name: "剧本分镜", does: "拆镜、运镜、镜头提示词", group: "准备" },
  { id: "consistency", name: "一致性", does: "人物/场景/道具/声音资产", group: "准备", media: "images" },
  { id: "video_gen", name: "视频生成", does: "模型适配、视频与声音生成", group: "制作", media: "videos" },
  { id: "quality_review", name: "质量审查", does: "剧本/分镜/资产/视频审核", group: "审查" },
];

const GROUPS = ["调度", "准备", "制作", "审查"];

/**
 * Review modes, as a fallback for a server that does not report them.
 *
 * `backend/app/doc/review_mode.py` decides what each mode blocks — it is a rule
 * table, not a word in a prompt — and `/api/doc/capabilities` ships that list so
 * the picker never advertises behaviour the pipeline lacks. This copy only keeps
 * the control rendered when the page is served by an older backend (or the
 * capability call fails); when the payload is present it wins.
 */
const REVIEW_MODES_FALLBACK = [
  { id: "STRICT", label: "严格审核", summary: "逐项核对史实、原始资料与硬性要求；P0–P2 都会打回重做" },
  { id: "NORMAL", label: "普通审核", summary: "交付标准（默认）；P0、P1 打回重做，P2、P3 只记录" },
  { id: "FAST", label: "快速审核", summary: "出草稿用；只有 P0 打回重做，P1 及以下只记录" },
];

/** The settings gear. Inline SVG for the same reasons as the rail icons. */
function GearIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <circle cx="8" cy="8" r="2.2" stroke="currentColor" strokeWidth="1.3" />
      <path
        d="M8 1.6v1.8M8 12.6v1.8M1.6 8h1.8M12.6 8h1.8M3.5 3.5l1.3 1.3M11.2 11.2l1.3 1.3M12.5 3.5l-1.3 1.3M4.8 11.2l-1.3 1.3"
        stroke="currentColor" strokeWidth="1.2"
      />
    </svg>
  );
}

/** The collaboration screen entry: a frame with the director/child hierarchy. */
function BigScreenIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <rect x="1.8" y="2.4" width="12.4" height="8.4" rx="1.2" stroke="currentColor" strokeWidth="1.2" />
      <path d="M5.2 13.6h5.6M8 10.8v2.8M4.5 7.9 6.7 6l1.8 1.2 2.8-2.6" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  );
}

/**
 * Icons for the rail controls.
 *
 * Inline SVG rather than an icon font or emoji: the app ships as one self-contained
 * bundle, and a webfont would add a request plus a flash of missing glyphs. Emoji
 * render differently on every platform, which is why "＋ 新对话" looked like a
 * different control on Windows than elsewhere.
 *
 * Each is `currentColor` so hover and active states colour it without a second
 * asset.
 */
function RailIcon({ name }: { name: "agents" | "history" | "new" | "collapse" }) {
  const common = { width: 13, height: 13, viewBox: "0 0 16 16", fill: "none", "aria-hidden": true };

  if (name === "agents") {
    // Three stacked nodes joined by lines: a roster, not a person.
    return (
      <svg {...common}>
        <circle cx="4" cy="4" r="2" stroke="currentColor" strokeWidth="1.3" />
        <circle cx="4" cy="12" r="2" stroke="currentColor" strokeWidth="1.3" />
        <circle cx="12" cy="8" r="2" stroke="currentColor" strokeWidth="1.3" />
        <path d="M6 4.6 10 7.4M6 11.4 10 8.6" stroke="currentColor" strokeWidth="1.3" />
      </svg>
    );
  }
  if (name === "history") {
    // A clock with an arrow: "past", not "settings".
    return (
      <svg {...common}>
        <circle cx="8" cy="8" r="5.4" stroke="currentColor" strokeWidth="1.3" />
        <path d="M8 4.8V8l2.4 1.6" stroke="currentColor" strokeWidth="1.3" />
      </svg>
    );
  }
  if (name === "new") {
    return (
      <svg {...common}>
        <path d="M8 3v10M3 8h10" stroke="currentColor" strokeWidth="1.6" />
      </svg>
    );
  }
  // Collapse: chevron pointing at the direction the rail will move.
  return (
    <svg {...common}>
      <path d="M9.5 3.5 5 8l4.5 4.5" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  );
}



/** Run statuses in plain words, rather than the backend's slugs. */
const STATUS_LABEL: Record<string, string> = {
  queued: "等待中",
  running: "运行中",
  waiting: "等你回复",
  done: "已完成",
  partial: "部分完成",
  failed: "失败",
  cancelled: "已取消",
};

/** 还在推进、还能取消的状态。其余都是终态——一条终态的任务才叫"历史"。 */
const LIVE_STATUSES = ["queued", "running", "waiting"];
const isLive = (status?: string) => Boolean(status) && LIVE_STATUSES.includes(status as string);

type AgentState = "idle" | "working" | "done" | "rework";

interface Turn {
  role: string;
  agent: string;
  kind: string;
  content: string;
  payload: Record<string, any>;
}

interface RunState {
  run_id: string;
  status: string;
  error: string;
  has_video: boolean;
  video_url: string;
  messages: Turn[];
  summary: Record<string, any>;
  material_check: Record<string, any>;
  final_cut: Record<string, any>;
  //: Live traces of agents still working, keyed by agent id. Present only while
  //: the run is in flight; this is what lets reasoning appear as it is written.
  live?: Record<string, Array<Record<string, any>>>;
  //: Who is working right now, and since when. The backend writes this around
  //: every model call and every render, because with streaming switched off a
  //: single call produces no events for a minute at a time — without this the
  //: conversation looks frozen rather than busy.
  working?: { agent?: string; label?: string; started_at?: number; seconds?: number };
  /** Whether the backend is streaming model output (设置 → 调用方式). Reported
   *  because "no live steps" has two causes — the switch is off, or the model
   *  returns no reasoning channel — and the placeholder row has to name the
   *  right one. Absent from older backends, hence optional. */
  streaming?: boolean;
}

/** One row in the history panel: the summary the list endpoint returns. */
interface HistoryRun {
  id: string;
  status: string;
  title: string;
  preview: string;
  chars: number;
  messages: number;
  created_at: string;
  /** Pipeline identifier reported by the API. */
  workflow_id?: string;
}

/** Human label for a run's pipeline, so the list does not show two rows that
 *  look identical while only one opens to prompts and media. */
function workflowLabel(id: string): string {
  if (id === "doc_pipeline") return "纪录片";
  return "";
}

/** Buckets a history list into 今天 / 昨天 / 更早.
 *
 * Grouping by label rather than by an offset keeps the section headings stable
 * across a day boundary — "昨天" always means the previous calendar day, not
 * "24 hours ago", which would put an evening run in the wrong bucket all day.
 */
function groupByDay(runs: HistoryRun[]): Array<{ label: string; runs: HistoryRun[] }> {
  const today = new Date();
  const dayOf = (iso: string) => {
    const d = new Date(iso);
    return `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`;
  };
  const yesterday = new Date(today);
  yesterday.setDate(today.getDate() - 1);

  const buckets: Record<string, HistoryRun[]> = {};
  for (const r of runs) {
    const key = dayOf(r.created_at);
    const label =
      key === dayOf(today.toISOString())
        ? "今天"
        : key === dayOf(yesterday.toISOString())
          ? "昨天"
          : "更早";
    (buckets[label] = buckets[label] || []).push(r);
  }
  return ["今天", "昨天", "更早"]
    .filter((l) => buckets[l]?.length)
    .map((label) => ({ label, runs: buckets[label] }));
}


export default function Studio() {
  const [run, setRun] = useState<RunState | null>(null);
  const [runId, setRunId] = useState("");
  /**
   * 「新开一个对话」只是把中栏切回输入态，不等于放弃正在跑的任务。
   *
   * 以前它会把 `runId` 一起清空，而轮询、任务板、协作大屏都只认 `runId`：
   * 任务在后台继续跑，前端却失去了它的 id，于是三处的数字全都停在按下那一刻
   * （任务板和大屏"不动了"），再想找回来只能去历史列表里点——那里又把它当
   * 历史打开，顶部还挂一条"正在查看历史记录 · 运行中"的自相矛盾横条。
   *
   * 现在分开：`runId` 继续跟着那个任务跑，`newConversation` 只决定中栏画什么。
   */
  const [newConversation, setNewConversation] = useState(false);
  /* 正在看的那条历史；空串表示中栏看的是当前任务。 */
  const [viewingHistory, setViewingHistory] = useState("");
  const [chapter, setChapter] = useState("");
  /**
   * The length this material needs, estimated from its own volume.
   *
   * There is no longer a picker. A fixed target made the storyboard pad or
   * compress content to reach it, which fights the whole point of adapting a
   * script. The backend still receives a number because it uses it to size the
   * token budget and to report how far the plan drifted — but that number now
   * follows the text, not the other way round.
   *
   * 4.5 characters/second is the midpoint of the 3.5–5.5 range the storyboard
   * prompt itself documents; the estimate is the length the narration alone
   * would take. Action and transitions add more, which is why the result is
   * described as a floor rather than a promise.
   */
  // Media switches, decided before the run starts. Off by default: a prompt costs
  // nothing, and spending credit should be a deliberate act.
  const [genImages, setGenImages] = useState(false);
  const [genVideos, setGenVideos] = useState(false);
  // How strictly the next run is reviewed. Persisted because this is a standing
  // preference ("I always want strict"), not a per-task decision; each run stores
  // the mode it actually used, so old records keep their own value.
  const [reviewMode, setReviewMode] = useState<string>(() => {
    try {
      const saved = window.localStorage.getItem("studio.reviewMode");
      if (saved === "STRICT" || saved === "NORMAL" || saved === "FAST") return saved;
    } catch {
      /* storage can be blocked; the default is a fine answer */
    }
    return "FAST";
  });
  const [busy, setBusy] = useState(false);
  // Always on. The 「暂停刷新」 toggle is gone — it stopped the display, never the
  // task, which is a confusing thing to offer next to a running job.
  const polling = true;
  const [cancelling, setCancelling] = useState(false);
  const [draft, setDraft] = useState("");
  const [caps, setCaps] = useState<any>(null);
  const [message, setMessage] = useState<{ kind: "ok" | "err" | "info"; text: string } | null>(null);
  const streamRef = useRef<HTMLDivElement | null>(null);

  // Left rail: which panel is showing, and whether the drawer is open.
  //
  // Two orthogonal pieces of state. `railTab` picks the content; `railCollapsed`
  // decides whether it is visible at all. Keeping them apart is what makes the
  // collapsed strip still able to switch tabs: collapsing does not reset the
  // choice, so reopening lands the user back where they were.
  const [railTab, setRailTab] = useState<"agents" | "history">("agents");
  const [railCollapsed, setRailCollapsed] = useState(false);

  /**
   * Rail width in pixels, dragged by the handle on its right edge.
   *
   * `null` means "use the stylesheet's default". That matters: the width must
   * resize with the window until the user expresses a preference, and a number
   * frozen at mount time would not. Once dragged, the explicit value wins.
   */
  const [railWidth, setRailWidth] = useState<number | null>(null);

  /**
   * Whether the settings dialog is open.
   *
   * Settings used to be a separate route, which meant leaving the workbench to
   * change a model — losing sight of a running task and of the conversation
   * being read. As a dialog it opens over the workbench and the run keeps going
   * behind it, which is how this app behaves elsewhere already.
   */
  const [settingsOpen, setSettingsOpen] = useState(false);
  // Read-only visualization of the same run. Closing it never changes the task.
  const [bigScreenOpen, setBigScreenOpen] = useState(false);
  /**
   * 右上角任务板开没开。`null` 表示「按默认」——有清单就展开。
   *
   * 用三态而不是布尔，是为了让「用户手动收起」这件事不被自动展开覆盖：一个
   * 布尔值必须在「第一张清单出现」的那个瞬间被写一次，而那次写入管不到下一轮
   * 任务；换成 null 之后，默认行为只由「有没有清单」决定，用户的动作只覆盖他
   * 自己这一轮。
   */
  const [planDock, setPlanDock] = useState<boolean | null>(null);
  const [settingsDirty, setSettingsDirty] = useState(false);
  const [settingsBusy, setSettingsBusy] = useState(false);
  const closeSettings = useCallback(() => {
    if (settingsBusy) return;
    if (settingsDirty && !window.confirm("有未保存的设置，确定放弃修改并关闭？")) return;
    setSettingsOpen(false);
    setSettingsDirty(false);
  }, [settingsDirty, settingsBusy]);
  // Escape closes the dialog. Registered on the window rather than the panel so
  // it works no matter which control has focus.
  useEffect(() => {
    if (!settingsOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        closeSettings();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [settingsOpen, closeSettings]);

  /** Whether the rail is being dragged right now. */
  const [railDragging, setRailDragging] = useState(false);

  //: Widths the drag works in, and the width the rail collapses to.
  //:
  //: `RAIL_COLLAPSE` sits *above* `RAIL_MIN` on purpose. Dragging past it snaps
  //: into the collapsed layout instead of shrinking to an unreadable sliver: a
  //: 140px rail cannot show a label and looks broken rather than compact. The gap
  //: between the two is the hysteresis band — the drag must come back out past
  //: `RAIL_MIN` before the rail expands again, so a pointer resting on the
  //: boundary does not flicker between the two layouts.
  const RAIL_COLLAPSE = 120;
  const RAIL_MIN = 160;
  const RAIL_MAX = 420;

  /** Begin a drag from the handle at the rail's right edge. */
  function onRailResizeStart(event: PointerEvent<HTMLDivElement>) {
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = (event.currentTarget.parentElement as HTMLElement)?.offsetWidth
      ?? (railCollapsed ? 52 : 220);
    setRailDragging(true);

    // Listeners go on `window`, not the handle: a fast drag leaves the handle
    // behind, and pointer capture on a 6px strip is fiddly and loses the drag on
    // a window boundary.
    const onMove = (e: PointerEvent) => {
      const raw = startWidth + (e.clientX - startX);

      // Dragging narrow enough snaps to the collapsed layout. The width is not
      // stored in that case: the collapsed strip has its own fixed size, and
      // keeping a number here would make "expand" resume from a sliver.
      if (raw < RAIL_COLLAPSE) {
        if (!railCollapsed) {
          setRailCollapsed(true);
          setRailWidth(null);
        }
        return;
      }

      // Coming back out: expand as soon as the pointer passes RAIL_MIN, so the
      // rail reappears with content rather than as a 121px empty column.
      if (raw >= RAIL_MIN) {
        if (railCollapsed) setRailCollapsed(false);
        setRailWidth(Math.min(RAIL_MAX, raw));
        return;
      }

      // Between the two thresholds: stay in whatever state we are in. This is the
      // hysteresis band that stops the layouts flickering at the boundary.
      if (!railCollapsed) setRailWidth(Math.min(RAIL_MAX, Math.max(RAIL_MIN, raw)));
    };

    const onUp = () => {
      setRailDragging(false);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      // Restore the cursor: it is set on body during the drag so it does not flicker
      // as the pointer crosses the stage.
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    document.body.style.cursor = "col-resize";
    // Without this, a drag selects the text of every element it passes over.
    document.body.style.userSelect = "none";
  }

  /** Width to render: the user's choice, or the stylesheet default. */
  const railStyle = railWidth === null ? undefined : { width: `${railWidth}px`, flexBasis: `${railWidth}px` };

  /** Collapsing while on a tab keeps that tab; expanding must never land on a
   *  blank drawer, so it is a plain toggle of the same two states. */
  function toggleRail() {
    setRailCollapsed((v) => !v);
  }

  /**
   * Start a fresh conversation: clear the centre pane back to the input form.
   *
   * What this does NOT do is stop anything. `runId` is what the poll loop follows,
   * so dropping it only detaches the *view* — the pipeline keeps executing on the
   * server and the run stays in history. That is the same rule the rest of this
   * component follows: looking away must never interrupt work.
   *
   * Which is exactly why a running task needs a confirmation first: the user is
   * about to lose sight of work that is still in flight, and getting it back
   * means finding it in the history list afterwards. A silent wipe would look
   * like the task had been cancelled.
   */
  function startNewConversation() {
    if (runId && isLive(run?.status)) {
      const ok = window.confirm(
        "有一个任务正在运行。新开对话只是把这个界面切回输入框，任务本身会继续在后台跑，顶部任务板和协作大屏照旧刷新；中栏会留一条「回到任务」的入口。\n\n确定要新开吗？",
      );
      if (!ok) return;
    }

    // 只清画面，不清任务：`runId` 是轮询、任务板、大屏唯一的锚，清掉它等于
    // 把正在跑的任务从界面上摘走（表现就是它们全都"不动了"）。
    // 任务已经结束的，没什么可留的，照旧整个归零。
    if (!isLive(run?.status)) {
      setRunId("");
      setRun(null);
    }
    setViewingHistory("");
    setNewConversation(true);
    setDraft("");
    setMessage(null);
    // The chapter box is deliberately left alone: a user composing a new task
    // and then changing their mind should not lose what they typed.
    setRailTab("agents");
    setRailCollapsed(false);
  }

  /* 顶部提示曾经永不退场：`setMessage` 只在开新任务时清空，于是
     「运行已结束（cancelled），无需取消」会一整轮占着横条。它现在待在常驻
     提示带里，永久占地更明显——成功类和信息类 6 秒后自己退场；失败类留着，
     错误要停在原地等用户读完。 */
  useEffect(() => {
    if (!message || message.kind === "err") return;
    const timer = window.setTimeout(() => setMessage(null), 6000);
    return () => window.clearTimeout(timer);
  }, [message]);

  useEffect(() => {
    fetch("/api/doc/capabilities")
      .then((r) => r.json())
      .then(setCaps)
      .catch(() => setCaps(null));
  }, []);

  const load = useCallback(async (id: string) => {
    try {
      const r = await fetch(`/api/doc/runs/${id}`);
      if (r.ok) setRun(await r.json());
    } catch {
      /* keep previous view */
    }
  }, []);

  /** 结果消息上的「继续生成」：只重跑没通过的段，后端自己知道重跑哪些。 */
  const retryRun = useCallback(
    async (id: string) => {
      try {
        const r = await fetch(`/api/doc/runs/${id}/retry`, { method: "POST" });
        if (r.ok) {
          // 续跑把这一条变回"在跑的任务"。如果它是以历史身份打开的，就得同时
          // 收回历史身份，否则又会出现"历史记录 · 运行中"、没有取消键、大屏
          // 停在那里的同一套问题。
          if (viewingHistory === id) {
            setViewingHistory("");
            setRunId(id);
            setNewConversation(false);
          }
          await load(id);
        }
      } catch {
        /* network hiccups leave the button available for another click */
      }
    },
    [load, viewingHistory],
  );

  // ---------------- history ----------------
  //
  // Two pieces of state, not one: `history` is the list, `historyOnlyDone` the
  // filter. Keeping them separate means toggling the filter does not discard the
  // already-fetched rows, so the switch feels instant.
  //
  // `viewingHistory` 声明在上面与 `runId` 一起：`retryRun` 的依赖数组在渲染时
  // 就会读它，放在这里会落到暂时性死区（Cannot access before initialization）。
  const [history, setHistory] = useState<HistoryRun[]>([]);
  const [historyOnlyDone, setHistoryOnlyDone] = useState(true);

  const fetchHistory = useCallback(async (onlyDone: boolean) => {
    try {
      const qs = onlyDone ? "?status=done&limit=40" : "?limit=40";
      const r = await fetch(`/api/doc/runs${qs}`);
      if (r.ok) setHistory((await r.json()).runs || []);
    } catch {
      /* leave the previous list */
    }
  }, []);

  // Load the list once on mount, and again whenever the filter changes.
  //
  // This effect must sit BELOW the declaration of `fetchHistory`: React runs the
  // component top to bottom, so an effect created above a `const` that is defined
  // further down reads it while it is still in its temporal dead zone and throws
  // "Cannot access 'fetchHistory' before initialization". That throw happens during
  // render, before anything is painted, which is why the whole page came up blank.
  useEffect(() => {
    void fetchHistory(historyOnlyDone);
  }, [fetchHistory, historyOnlyDone]);

  /** Make a task the user just started show up in the history list.
   *
   * Two reasons it did not: the list was only fetched on mount and on a filter
   * change, so a new run stayed invisible until the page was reloaded; and the
   * default filter is 「仅已完成」, which a run that started a second ago can never
   * satisfy. Being told 「新任务已开始」 with nothing in the list to point at reads
   * as a failed submit.
   *
   * So the filter is released once rather than the row being faked as done: the
   * user is then looking at all runs, which is where their new one actually is.
   */
  function revealNewRun() {
    if (historyOnlyDone) setHistoryOnlyDone(false);
    else void fetchHistory(false);
  }

  /** Delete one history entry, after confirming.
   *
   * Confirmation is required because the delete is permanent: the rows and the
   * generated files are removed, and there is no undo.
   */
  async function deleteHistory(id: string) {
    if (!window.confirm("删除这条历史对话？相关消息和生成的图片/视频都会一起删除，无法恢复。")) {
      return;
    }
    try {
      const r = await fetch(`/api/doc/runs/${id}`, { method: "DELETE" });
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        throw new Error(j.detail || `删除失败（HTTP ${r.status}）`);
      }
      setHistory((prev) => prev.filter((x) => x.id !== id));
      // If the deleted run is the one on screen — or the active one — clear the
      // view rather than leave a conversation visible that no longer exists.
      if (viewingHistory === id || runId === id) {
        setViewingHistory("");
        setRun(null);
        if (runId === id) {
          setRunId("");
          setNewConversation(false);
        }
      }
    } catch (e) {
      setMessage({ kind: "err", text: e instanceof Error ? e.message : "删除失败" });
    }
  }

  /** Open one past run.
   *
   * A running task keeps running: the poll loop is keyed off `runId`, which is not
   * touched here. Viewing history is a read, not a switch of the active run.
   *
   * 例外是「还在跑的那一条」：它不是历史，点它要回到任务本身。当成历史打开
   * 会让三样东西一起消失——轮询（大屏和任务板就此停摆）、底部输入框、取消键，
   * 而顶部那条横条会写着"正在查看历史记录 · 运行中"，把一件事同时说成两件。
   */
  async function openHistory(id: string) {
    const row = history.find((h) => h.id === id);
    if (row && isLive(row.status)) {
      setNewConversation(false);
      setViewingHistory("");
      setRunId(id);
      await load(id);
      return;
    }
    setViewingHistory(id);
    await load(id);
  }

  /** Stop viewing a past run.
   *
   * With an active task: return to it, which means reloading it because the
   * history view has overwritten `run`. Without one: there is nothing to
   * return to, so clear the pane back to the input form instead of loading "".
   */
  function backToCurrent() {
    setNewConversation(false);
    setViewingHistory("");
    if (runId) {
      void load(runId);
    } else {
      setRun(null);
    }
  }

  // Re-attach to a task that is still executing when this page (re)loads.
  //
  // `runId` lives in memory only, so a refresh used to drop the user back to the
  // input form while the pipeline kept running. That was survivable when cancel
  // sat in the top bar; now that the composer owns it, losing the run meant
  // losing the only way to stop it. The newest live run wins, and an explicit
  // `setRunId` (from starting or opening one) is never overridden.
  useEffect(() => {
    let disposed = false;
    (async () => {
      try {
        const r = await fetch("/api/doc/runs?limit=10");
        if (!r.ok) return;
        const data = await r.json();
        const live = (data.runs || []).find(
          (x: { status?: string }) =>
            x.status === "running" || x.status === "queued" || x.status === "waiting",
        );
        if (!disposed && live) setRunId((current) => current || live.id);
      } catch {
        /* nothing running, or the list is unavailable — the form is the fallback */
      }
    })();
    return () => {
      disposed = true;
    };
  }, []);

  //: Which run's terminal state the history list was last refreshed for. The
  //: poll effect has other dependencies (viewingHistory among them) that can
  //: re-run it after the run has already stopped; without this, each such re-run
  //: costs another list request for a state we have already synced.
  const historySyncedRef = useRef("");

  // Poll the ACTIVE run only while the conversation on screen is that run.
  //
  // The poll writes `run`, and opening a past run also writes `run` — so without
  // this guard the loop would overwrite the history view every 1.5 s. Note it
  // pauses *display*, not the task: the pipeline keeps executing server-side, which
  // is the behaviour confirmed earlier ("看历史不该影响正在跑的任务").
  useEffect(() => {
    if (!runId || viewingHistory) return;
    void load(runId);
    // 终态就停。以前这个定时器只依赖 runId，于是任务停止/失败之后还在每 1.5 秒
    // 问一次——后台什么都没发生，日志里却是一屏 200，看起来像任务没停。
    // 只信「这一条 run」的状态。刚点完开始时 `run` 还停在上一任务的终态上，
    // 直接拿它的 status 判断会立刻判定为已结束，新任务于是再也不刷新。
    const status = run?.run_id === runId ? run?.status || "" : "";
    // waiting 也要继续轮询：回答之后状态会变，而且用户可能在另一个窗口回答。
    if (status && !["running", "queued", "waiting"].includes(status)) {
      // 这一刻任务结束了：历史列表里那一条要从「运行中」变成结果，在「仅已完成」
      // 筛选下也到这时才该出现。之前列表只在挂载和切筛选时拉取，所以它一直挂着
      // 旧状态，直到用户重载页面。同一条 run 的同一个终态只补一次。
      const stamp = `${runId}:${status}`;
      if (historySyncedRef.current !== stamp) {
        historySyncedRef.current = stamp;
        void fetchHistory(historyOnlyDone);
      }
      return;
    }
    const t = window.setInterval(() => void load(runId), 1500);
    return () => window.clearInterval(t);
  }, [runId, load, viewingHistory, run?.status, fetchHistory, historyOnlyDone]);

  // Follow the stream only when the user is already at the bottom; otherwise a
  // poll would yank the view away while they read history.
  useEffect(() => {
    const el = streamRef.current;
    if (!el) return;
    const d = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (d < 80) el.scrollTop = el.scrollHeight;
  }, [run?.messages.length]);

  /** Remember the mode for next time; the run itself records its own copy. */
  function changeReviewMode(value: string) {
    setReviewMode(value);
    try {
      window.localStorage.setItem("studio.reviewMode", value);
    } catch {
      /* storage is optional — losing the memory is not worth an error */
    }
  }

  /** The run request body.
   *
   *  Both submit paths (「开始生成」 and the composer's 「新任务」) go through this,
   *  because they drifted twice already: `review_mode` was wired into one and not
   *  the other, and then the two media switches — the boxes were ticked, the run
   *  still wrote prompts only, and the pipeline recorded `generate_videos: false`
   *  with nothing pointing at the missing field.
   */
  function runPayload(text: string) {
    return {
      chapter_text: text,
      // 按**这段**文字算，不是按工作台上那段。从输入框再提交一段新文本时，
      // 旧写法会把上一段的估算带过去。
      target_seconds: targetSecondsFor(text),
      max_rework: 1,
      review_mode: reviewMode,
      // The switches decide whether this run also renders. Prompts are produced
      // either way, so a run with both off still yields the deliverable.
      generate_images: genImages,
      generate_videos: genVideos,
    };
  }

  /** Submit a new task from the composer; does not disturb a running one. */
  async function submit() {
    const text = draft.trim();
    if (!text || busy) return;
    // 这里以前挡「不足 150 字」，把「帮我做一个关于党的诞生的视频」这类请求直接
    // 拦在门外——恰恰是 RAG 唯一能帮上忙的那种输入。材料够不够由后端判断
    // （够长 / 能识别成主题请求），前端再抄一份规则只会两边越走越远。
    setBusy(true);
    setMessage(null);
    try {
      const r = await fetch("/api/doc/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(runPayload(text)),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || "启动失败");
      setDraft("");
      setRunId(j.run_id);
      setNewConversation(false);
      // 提交时如果正停在某条历史记录上，`viewingHistory` 还指着旧的那条，
      // 轮询会以为"用户在读历史"而拒绝刷新——新任务于是冻住不动。
      setViewingHistory("");
      revealNewRun();
      setMessage({ kind: "ok", text: "新任务已开始（上一个任务仍独立运行）" });
    } catch (e: unknown) {
      setMessage({ kind: "err", text: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
    }
  }

  /** Ask the backend to stop the current run at its next checkpoint. */
  async function cancel() {
    if (!runId || cancelling) return;
    setCancelling(true);
    try {
      const r = await fetch(`/api/doc/runs/${runId}/cancel`, { method: "POST" });
      const j = await r.json();
      setMessage({
        // 这个接口的 ok:false 只有一种含义：任务已经结束，没什么可取消的。
        // 那不是失败，涂成红色会让用户以为取消出了错——用中性色，并让它
        // 自己退场，别在顶部常驻一条已经过去的事。
        kind: j.ok ? "ok" : "info",
        text: j.ok ? j.note || "已请求取消" : j.reason || "无法取消",
      });
      await load(runId);
    } catch (e: unknown) {
      setMessage({ kind: "err", text: e instanceof Error ? e.message : String(e) });
    } finally {
      setCancelling(false);
    }
  }

  async function start() {
    if (!chapter.trim()) {
      setMessage({
        kind: "err",
        text: "先写一段材料，或一句「帮我做一个关于××的视频，1 分钟左右」",
      });
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const r = await fetch("/api/doc/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(runPayload(chapter)),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || "启动失败");
      setRunId(j.run_id);
      setNewConversation(false);
      setViewingHistory("");
      revealNewRun();
    } catch (e: unknown) {
      setMessage({ kind: "err", text: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
    }
  }

  const turns: Turn[] = run?.messages || [];

  /** 列表里哪一行代表"用户此刻正在看的东西"。
   *
   * 只标 `viewingHistory` 是不够的：点一条还在跑的任务时走的是"回到任务"分支，
   * `viewingHistory` 会被清空，于是那条刚被点开的任务在列表里没有任何标记——
   * 用户看不出自己在看哪一条。当前任务和正在读的历史都算"当前"，两者互斥
   * （`viewingHistory` 有值时主区显示的是历史），所以一个 id 就够。 */
  const historyCurrentId = viewingHistory || (newConversation ? "" : runId);

  /* 最新一张任务清单。后端用同一个 msg_key 原地覆写，所以「最后一条 plan」就是
     当前状态，不需要合并历史。 */
  const planCard: PlanCardData | null = (() => {
    for (let i = turns.length - 1; i >= 0; i -= 1) {
      if (turns[i].kind === "plan") return turns[i].payload as PlanCardData;
    }
    return null;
  })();
  /* 换一轮任务时忘掉上一轮的开合决定：上一轮用户收起了板，这一轮不该跟着收起。 */
  useEffect(() => {
    setPlanDock(null);
  }, [runId]);
  const planDockOpen = planCard ? planDock ?? true : false;
  // Media flags are captured when a run is created; while one is in flight the
  // switches only queue a change for the next one.
  // waiting 也算「进行中」：任务没结束，只是卡在一次提问上。此时仍然禁用提交，
  // 否则用户会在没回答的情况下又开一条新任务，原来那条永远停在等回答。
  const runActive = isLive(run?.status);

  /* Who is working, and for how long. Read here rather than inside the JSX map so
     the one-second tick re-renders a leaf component, not the conversation. */
  const workingAgent = run?.working?.agent || "";
  const workingSeconds = useElapsedSeconds(run?.working);
  const workingName = AGENTS.find((a) => a.id === workingAgent)?.name || workingAgent;
  // An agent that is already streaming its reasoning has a LiveTurn of its own;
  // a second "正在工作" row for the same agent would read as two parallel jobs.
  // A live-trace key can exist with no steps yet (the agent announced itself but
  // has not flushed a reasoning snippet). Treating "key exists" as "has something
  // to show" left that gap blank: LiveTurn skips empty step lists, and this flag
  // suppressed the placeholder that would otherwise cover it.
  const workingHasLiveTurn = ((run?.live || {})[workingAgent] || []).length > 0;
  const check = run?.material_check || {};

  // Picker options: the server's list when it is at least as complete as the one
  // this build knows about, else the local copy. A payload missing a mode would
  // leave the select blank for that value — which reads as a broken control
  // rather than as a stale one — so the local list is the safer default. Extra
  // modes the server knows about are still shown.
  const serverModes: Array<{ id: string; label: string; summary: string }> =
    Array.isArray(caps?.review_modes) ? caps.review_modes : [];
  const reviewModes = REVIEW_MODES_FALLBACK.every((m) => serverModes.some((s) => s.id === m.id))
    ? serverModes
    : REVIEW_MODES_FALLBACK;
  const reviewModeSummary = reviewModes.find((m) => m.id === reviewMode)?.summary || "";

  /**
   * Which run the centre pane is showing, or "" for the input form.
   *
   * `runId` is the ACTIVE task — the one a poll loop follows and that the
   * composer starts new work from. `viewingHistory` is which past run the user
   * opened. They are different axes, and the centre pane must key off "is there
   * anything to show", not off "is a task active".
   *
   * The bug this fixes: the pane was gated on `runId && ...`, so opening a
   * history entry from a freshly loaded page — where no task had been started
   * yet and `runId` was "" — set `viewingHistory` and loaded the data, but
   * neither branch rendered: the intro form showed (because `!runId`) and the
   * conversation did not. The data was there the whole time; only the condition
   * was wrong.
   *
   * `newConversation` is the third state: the user asked for a blank canvas while
   * a task is still running. `runId` stays (that is what keeps the poll, the plan
   * dock and the big screen alive), but the pane shows the input form.
   */
  const shownRunId = viewingHistory || (newConversation ? "" : runId);

  /** Short title for the big screen; the full user material stays in the conversation. */
  const bigScreenTitle = (() => {
    const userTurn = turns.find((turn) => turn.role === "user");
    const text = String(userTurn?.content || chapter || draft || "等待新任务").replace(/\s+/g, " ").trim();
    return text.length > 54 ? `${text.slice(0, 54)}…` : text;
  })();

  /** Derive each agent's state from the messages, not from a separate field. */
  function stateOf(agentId: string): AgentState {
    const aliases: Record<string, string[]> = {
      script_adaptor: ["script_adaptor"],
      consistency: ["consistency"],
      video_gen: ["video_gen"],
      quality_review: ["quality_review"],
    };
    const names = aliases[agentId] || [agentId];

    const mine = turns.filter((t) => names.includes(t.agent));
    if (!mine.length) return "idle";
    const last = mine[mine.length - 1];
    if (last.kind === "error") return "rework";
    if (last.kind === "rework") return "rework";
    if (last.kind === "review" && !last.payload?.approved) return "rework";
    // A stage message with no later activity means the agent finished its turn.
    return run?.status === "running" && mine.length && last.kind === "stage" ? "done" : "done";
  }

  return (
    <div
      className={`app workbench-app ${railCollapsed ? "rail-collapsed" : ""}`}
      // 顶栏左侧那段要和左轨连成一块红墙，所以顶栏得知道轨道现在多宽
      // （默认 220，用户可拖拽，折叠时 52）。用变量传，避免两处硬编码。
      style={{ "--rail-w": railCollapsed ? "52px" : `${railWidth ?? 220}px` } as any}
    >
      <header className="topbar">
        <div className="brand">
          {/* 门楣只留片名。原来那句「6 个 Agent · 文本 → 人物视频」是把内部
              结构写给用户看；任务摘要（几段几镜头）在 220px 的墙里也放不下，
              溢出会跨到纸面上，所以顶栏只保留两个字，进度看任务板。 */}
          <Logo size={28} />
          <h1>漫簇</h1>
        </div>
        <div className="topbar-right">
          {/* 「暂停刷新」 and a separate 「取消任务」 used to live here. Both are
              gone: pausing the display of a run is not something anyone asked for
              twice, and cancelling belongs next to the thing it acts on — the
              composer, where the user's hand already is. */}
          {/* 任务板的把手。清单本身固定在右上角，这里只是它的开关——没有这个
              按钮，收起之后就没有入口了。 */}
          {planCard && (
            <PlanDockButton
              card={planCard}
              open={planDockOpen}
              onToggle={() => setPlanDock((v) => !(v ?? true))}
            />
          )}
          <button
            className="btn ghost small"
            onClick={() => setBigScreenOpen(true)}
            title="打开多 Agent 协作大屏"
            aria-label="打开多 Agent 协作大屏"
            style={{ gap: 5 }}
          >
            <BigScreenIcon />
            协作大屏
          </button>
          <button
            className="btn ghost small"
            onClick={() => setSettingsOpen(true)}
            title="打开设置"
            aria-label="打开设置"
            style={{ gap: 5 }}
          >
            <GearIcon />
            设置
          </button>
        </div>
      </header>

      {/* 常驻在右上角，压在对话区上。不占布局宽度：清单是随时要瞄一眼的东西，
          让它挤掉正文等于用一份计划换掉正在看的内容。 */}
      {planCard && planDockOpen && (
        <div id="plan-dock">
          <PlanDock card={planCard} onClose={() => setPlanDock(false)} />
        </div>
      )}

      <div className={`workbench ${railCollapsed ? "rail-collapsed" : ""} ${railDragging ? "rail-dragging" : ""}`}>
        {/* ---------------- left: tabbed drawer ----------------
            The roster and the history list used to be stacked in one scrolling
            column, so reaching a conversation meant scrolling past the agent list.
            They are now two tabs over a single drawer, and the drawer collapses. */}
        <aside className="agent-rail" style={railCollapsed ? undefined : railStyle}>
          <div className="rail-tabs">
            {/* Icon always, label when there is room. The label is hidden by CSS
                at narrow widths rather than removed here, so the text stays in the
                DOM for screen readers and for the tooltip. */}
            <button
              className={`rail-tab ${railTab === "agents" ? "active" : ""}`}
              onClick={() => {
                setRailTab("agents");
                setRailCollapsed(false);
              }}
              title="智能体"
              aria-pressed={railTab === "agents"}
            >
              <RailIcon name="agents" />
              <span className="rail-tab-label">智能体</span>
              <span className="rail-tab-count">{AGENTS.length}</span>
            </button>
            <button
              className={`rail-tab ${railTab === "history" ? "active" : ""}`}
              onClick={() => {
                setRailTab("history");
                setRailCollapsed(false);
              }}
              title="历史对话"
              aria-pressed={railTab === "history"}
            >
              <RailIcon name="history" />
              <span className="rail-tab-label">历史对话</span>
              {history.length > 0 && <span className="rail-tab-count">{history.length}</span>}
            </button>
            {/* Far right of the same row, per the sketch. */}
            <button
              className="rail-toggle"
              onClick={toggleRail}
              title={railCollapsed ? "展开目录" : "收起目录"}
              aria-label={railCollapsed ? "展开目录" : "收起目录"}
              aria-expanded={!railCollapsed}
            >
              <span style={{ display: "inline-flex", transform: railCollapsed ? "rotate(180deg)" : "none" }}>
                <RailIcon name="collapse" />
              </span>
            </button>
          </div>

          {/* 新建任务不再挤在页签行里：三个标签加一个动作挤在 220px 里
              只能缩成单字，而它又不是页签，是动作。单独一行反而更清楚。 */}
          <div className="rail-actions">
            <button
              className="rail-new"
              onClick={startNewConversation}
              title="新建任务"
              aria-label="新建任务"
            >
              <RailIcon name="new" />
              <span className="rail-new-label">新建任务</span>
            </button>
          </div>

          {!railCollapsed && (
            <div className="rail-body">
              {railTab === "agents" && (
                <>
                  {GROUPS.map((group) => (
                    <div className="rail-group" key={group}>
                      <div className="rail-stage">{group}</div>
                      {AGENTS.filter((a) => a.group === group).map((a) => {
                        const st = stateOf(a.id);
                        return (
                          <div className={`rail-item ${st}`} key={a.id}>
                            <span className="rail-dot" />
                            <div className="rail-text">
                              <span className="rail-label">
                                {a.name}
                              </span>
                              <span className="rail-sub">
                                {st === "working" ? "工作中" : st === "done" ? "完成"
                                  : st === "rework" ? "返工" : !shownRunId ? "待开始" : a.does}
                              </span>
                            </div>
                            {/* Only media stages get a switch. Checking it means
                                "write the prompt AND render"; leaving it clear
                                means "prompt only".
                                In 提示词模式 the switch is disabled rather than
                                hidden: it exists, it is simply not this machine's
                                current mode, and a control that vanishes makes a
                                user hunt for a setting that is not there. */}
                            {a.media && (
                              <label
                                className={`rail-switch ${caps?.prompt_only ? "off" : ""}`}
                                title={
                                  caps?.prompt_only
                                    ? caps.media_note || "提示词模式：只出提示词，不渲染画面"
                                    : runActive
                                        // The flags are read when the run is created, so
                                        // ticking one mid-run changes nothing — saying
                                        // "下一轮" is what keeps a live-looking control
                                        // from promising something it cannot do.
                                        ? "本次任务已开始，改动只对下一个任务生效"
                                        : "勾选后同时生成，不勾选只出提示词"
                                }
                              >
                                <input
                                  type="checkbox"
                                  checked={a.media === "images" ? genImages : genVideos}
                                  onChange={(e) => {
                                    const on = e.target.checked;
                                    if (a.media === "images") {
                                      setGenImages(on);
                                      return;
                                    }
                                    setGenVideos(on);
                                  }}
                                  disabled={busy || Boolean(caps?.prompt_only)}
                                />
                                <span>{a.media === "images" ? "出图" : "出视频"}</span>
                                {runActive && <span className="rail-switch-note">下一轮</span>}
                              </label>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  ))}

                  {/* Two distinct states, two different colours. Prompt-only is
                      the configured mode, not a fault: reporting it in red taught
                      the user to read a normal run as broken. Red is reserved for
                      full mode with nothing to call. */}
                  {caps?.prompt_only && (
                    <div className="memory-box">
                      <div className="memory-head">提示词模式</div>
                      <div className="memory-line">
                        {caps.media_note || "只产出提示词与校验，不生成图片/视频"}
                      </div>
                    </div>
                  )}

                  {caps && !caps.prompt_only && !caps.can_generate && (
                    <div className="memory-box" style={{ borderColor: "rgba(248,113,113,.3)" }}>
                      <div className="memory-head" style={{ color: "#fca5a5" }}>生成能力未就绪</div>
                      <div className="memory-line">{caps.reason || "缺少供应商 Key"}</div>
                    </div>
                  )}
                </>
              )}

              {railTab === "history" && (
                <div className="history-box">
                  <div className="history-head">
                    <span>历史对话</span>
                    {/* Filter: a first screen full of failed debug runs is noise,
                        so the default view is finished runs only. The toggle is
                        explicit rather than hidden, because failed attempts are
                        also worth reviewing. */}
                    <label className="history-filter" title="勾选只看已完成">
                      <input
                        type="checkbox"
                        checked={historyOnlyDone}
                        onChange={(e) => {
                          setHistoryOnlyDone(e.target.checked);
                          void fetchHistory(e.target.checked);
                        }}
                      />
                      <span>只看完成</span>
                    </label>
                  </div>

                  {history.length === 0 && <div className="history-empty">暂无记录</div>}

                  {groupByDay(history).map((group) => (
                    <div className="history-group" key={group.label}>
                      <div className="history-group-label">{group.label}</div>
                      {group.runs.map((h) => (
                        <div
                          className={`history-item ${historyCurrentId === h.id ? "current" : ""}`}
                          key={h.id}
                          onClick={() => void openHistory(h.id)}
                          title={h.preview}
                        >
                          <div className="history-text">
                            <div className="history-title">{h.title}</div>
                            <div className="history-meta">
                              {/* 还在跑的这一条不是"历史"：点它等于回到那个任务。
                                  没有这个标记时，列表里一行动着的任务和一堆结束的
                                  任务长得一样，点进去的效果也看不出来。 */}
                              {isLive(h.status) && <span className="history-live">运行中</span>}
                              {h.workflow_id && (
                                <span className={`history-wf ${h.workflow_id === "doc_pipeline" ? "doc" : ""}`}>
                                  {workflowLabel(h.workflow_id)}
                                </span>
                              )}
                              {h.messages} 条消息 · {h.chars} 字 ·{" "}
                              {new Date(h.created_at).toLocaleTimeString("zh-CN", {
                                hour: "2-digit",
                                minute: "2-digit",
                              })}
                            </div>
                          </div>
                          {/* Stop propagation: the row also opens the run, and
                              deleting should not also navigate to the run being
                              deleted. */}
                          <button
                            className="history-del"
                            title="删除这条历史"
                            onClick={(e) => {
                              e.stopPropagation();
                              void deleteHistory(h.id);
                            }}
                          >
                            ✕
                          </button>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Resize handle on the right edge, present in both states.
              It must stay while collapsed: dragging the narrow rail back out is
              how a user expands it by hand, and hiding the handle there would
              leave the collapse button as the only way back. */}
          <div
            className="rail-resize"
            onPointerDown={onRailResizeStart}
            role="separator"
            aria-orientation="vertical"
            aria-label="拖动调整目录宽度"
            title="拖动可调整宽度"
          />
        </aside>

        {/* ---------------- centre ---------------- */}
        <main className="stage">
          {/* 常驻提示带：状态横条和历史模式横条都放在滚动区之外。

              历史横条原来在 `.conversation` 里面，是那个滚动容器的第一个子
              元素——打开历史时它看着像固定，滚一下就没了，而它恰恰是用来提醒
              "你看的不是当前任务"的，最不该消失。两条横条现在同处一层，滚动
              消息时一起不动；左右内边距与 `.conversation` 相同，所以它们的
              左边缘和正文对齐。 */}
          {(message || (shownRunId && viewingHistory) || (newConversation && runId)) && (
            <div className="stage-notices">
              {message && (
                <div className={`banner dismissible ${message.kind === "err" ? "fail" : message.kind}`}>
                  <span>{message.text}</span>
                  <button
                    className="banner-close"
                    onClick={() => setMessage(null)}
                    title="关闭这条提示"
                    aria-label="关闭提示"
                  >
                    ✕
                  </button>
                </div>
              )}

              {/* 新开对话后仍然存在的那个任务：告诉用户它还活着，并给一个回去的
                  入口。没有这一条，"新建任务"就等于把正在跑的东西藏起来——任务板
                  和大屏在刷新，中栏却是一张空表单，看起来像任务丢了。 */}
              {newConversation && runId && (
                <div className="banner info resume-banner">
                  <span>
                    上一个任务
                    {isLive(run?.status) ? "仍在后台运行" : "已结束"}
                    {run?.status ? `（${STATUS_LABEL[run.status] || run.status}）` : ""}
                  </span>
                  <button className="btn ghost small" onClick={backToCurrent}>
                    回到任务
                  </button>
                </div>
              )}

              {/* History mode banner: it must be obvious that the conversation
                  below is a past run, not the active one — otherwise a user could
                  read an old result and think the current task produced it. */}
              {shownRunId && viewingHistory && (
                <div className="banner info history-banner">
                  <span>
                    正在查看历史记录（{new Date(
                      (history.find((h) => h.id === viewingHistory)?.created_at) || Date.now(),
                    ).toLocaleString("zh-CN")}
                    ）
                    {run?.status && ` · ${STATUS_LABEL[run.status] || run.status}`}
                  </span>
                  {/* Two exits, because the right one depends on whether a task
                      is actually running: with one, returning means going back
                      to it; without one, it means going back to the input form. */}
                  {runId ? (
                    <button className="btn ghost small" onClick={backToCurrent}>
                      回到当前任务
                    </button>
                  ) : (
                    <button className="btn ghost small" onClick={backToCurrent}>
                      关闭历史
                    </button>
                  )}
                </div>
              )}
            </div>
          )}

          {!shownRunId && (
            <div className="intro project-intro">
              <section className="project-hero" aria-labelledby="project-intro-title">
                <div className="project-copy">
                  <span className="project-index">01 / PROJECT</span>
                  <p className="project-kicker">新建政治人物纪录片</p>
                  <h2 id="project-intro-title">
                    让史料成为影像，
                    <br />
                    让每一步都有据可查。
                  </h2>
                  <p className="project-description">
                    输入人物、事件或历史材料。导演负责拆解任务，内容改编进行史实核验，
                    分镜、一致性、视频生成与质量审查依次推进。
                  </p>
                </div>

                <div className="project-form">
                  <label className="project-form-label" htmlFor="project-material">
                    纪录片主题或原始材料
                  </label>
                  <textarea
                    id="project-material"
                    className="input chapter-input project-material"
                    placeholder={
                      "粘贴历史正文；也可以只写一句主题，例如\n"
                      + "「帮我做一个关于党的诞生的视频，时长大概 1 分钟左右」\n"
                      + "库里有就直接用，没有会去权威网站检索并入库。"
                    }
                    value={chapter}
                    onChange={(e) => setChapter(e.target.value)}
                    rows={8}
                  />

                  <div className="project-form-actions">
                    {/* Review mode stays beside the launch action because it changes
                        which review severities can stop this specific run. */}
                    <label className="review-mode project-review" title={reviewModeSummary}>
                      <span className="review-mode-label">审核模式</span>
                      <select
                        className="review-select"
                        value={reviewMode}
                        onChange={(e) => changeReviewMode(e.target.value)}
                        disabled={busy}
                        aria-label="审核模式"
                      >
                        {reviewModes.map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.label}
                          </option>
                        ))}
                      </select>
                    </label>

                    <button className="btn primary project-start" onClick={start} disabled={busy}>
                      {busy ? <span className="spinner" /> : <><span>立即制作</span><span aria-hidden="true">→</span></>}
                    </button>
                  </div>

                  <p className="project-review-note">{reviewModeSummary}</p>
                </div>

                <div className="project-mark" aria-hidden="true">
                  <span>纪录片生产系统</span>
                  <strong>MANcu · 01</strong>
                </div>
              </section>

              <footer className="project-workflow" aria-label="制作流程">
                <div className="project-workflow-title">
                  <span>WORKFLOW</span>
                  <strong>六个 Agent，一条可审查的制作路线</strong>
                </div>
                <div className="project-workflow-steps" aria-hidden="true">
                  <span>导演</span><i /><span>改编</span><i /><span>分镜</span><i />
                  <span>一致性</span><i /><span>生成</span><i /><span>审查</span>
                </div>
              </footer>

              <p className="project-footnote">
                时长由内容、对白、动作与转场需求自动估算；只写主题时，系统会先检索并核对资料。
              </p>
            </div>
          )}

          {shownRunId && (
            <div className="conversation" ref={streamRef}>
              {check && check.ok === false && (
                <div className="banner fail">
                  材料提示：{check.advice || "事件密度偏高、缺少人物细节，画面可能偏静态"}
                </div>
              )}

              {turns.map((t, i) => (
                <TurnRow
                  key={i}
                  turn={t}
                  agentName={AGENTS.find((a) => a.id === t.agent)?.name || t.agent}
                  runId={runId}
                  canAnswer={run?.status === "waiting"}
                  onAnswered={() => runId && void load(runId)}
                  canRetry={run?.status === "done" || run?.status === "failed"}
                  onRetry={() => runId && void retryRun(runId)}
                />
              ))}

              {/* Live reasoning from agents still working. Rendered after the
                  finished messages, and keyed by agent so a stage that is
                  thinking right now shows its text growing in place. */}
              {Object.entries(run?.live || {}).map(([agent, steps]) => {
                // Skip agents whose trace holds nothing worth showing yet.
                if (!steps?.length) return null;
                return (
                  <LiveTurn
                    key={`live-${agent}`}
                    agent={agent}
                    name={AGENTS.find((a) => a.id === agent)?.name || agent}
                    steps={steps}
                    elapsed={agent === workingAgent ? workingSeconds : 0}
                  />
                );
              })}

              {/* The silent case. With streaming off there is no reasoning to
                  grow in place, so without this row a 60-second model call shows
                  nothing at all and reads as a hang. */}
              {workingAgent && !workingHasLiveTurn && (
                <WorkingTurn
                  name={workingName}
                  label={run?.working?.label || ""}
                  seconds={workingSeconds}
                  streaming={run?.streaming}
                />
              )}

              {run?.status === "failed" && <div className="banner fail">失败：{run.error}</div>}

              {run?.video_url && (
                <div className="turn agent">
                  <div className="turn-head">
                    <span className="turn-avatar">影</span>
                    <span className="turn-name">成片</span>
                    <span className="turn-tag">{run.final_cut?.note || ""}</span>
                  </div>
                  <div className="turn-body">
                    <video className="film" src={run.video_url} controls preload="metadata" />
                    <div className="film-actions">
                      <a className="btn ghost small" href={run.video_url} download>下载 MP4</a>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* 输入框跟着"中栏是不是在显示当前任务"走，不跟 runId 走：
              新开对话时 runId 还在（任务要留给任务板和大屏），但中栏是输入表单，
              两个输入框同时出现会让人不知道哪一个是当前的。 */}
          {runId && !newConversation && (
            <div className="composer">
              <div className="composer-hint muted">
                {/* 停在「等你回复」时既没在跑、也没结束：这时说「任务结束后发送」
                    是错的（没人会替它结束），而说「取消任务」也不知道取消的是什么。 */}
                {run?.status === "waiting"
                  ? "停在等你回复：先在上面选一项继续，或点右侧结束这次任务"
                  : runActive
                    ? "运行中：右侧按钮是「取消任务」；现在输入的文字会在任务结束后发送"
                    : `状态：${STATUS_LABEL[run?.status || ""] || run?.status || ""}`}
              </div>
              <div className="composer-row">
                <textarea
                  className="input composer-input"
                  placeholder={
                    run?.status === "waiting"
                      ? "想换一段材料，可以先写在这里；结束上面那次任务后会用它开新任务"
                      : runActive
                        ? "可以先写下下一段文本，任务结束后点「发送」"
                        : "粘贴新的文本，或写一句主题开始下一个任务"
                  }
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onKeyDown={(e) => {
                    // Enter submits only when nothing is running: while a task is
                    // in flight the same button cancels, so a stray Enter must not
                    // quietly start a second run behind the user's back.
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      if (!runActive) void submit();
                    }
                  }}
                  rows={1}
                />
                {/* Same picker as the intro form: both buttons start a run, so
                    they must not disagree about how strictly it is reviewed. */}
                <label className="review-mode" title={reviewModeSummary}>
                  <span className="review-mode-label">审核</span>
                  <select
                    className="review-select compact"
                    value={reviewMode}
                    onChange={(e) => changeReviewMode(e.target.value)}
                    disabled={busy}
                    aria-label="审核模式"
                  >
                    {reviewModes.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.label}
                      </option>
                    ))}
                  </select>
                </label>
                {/* One button, two meanings, because there is only one thing to do
                    with a composer while a run is in flight: stop it. Splitting
                    "start a new task" and "cancel" across two places made it
                    possible to have two runs going and no obvious way to stop
                    either. */}
                {runActive ? (
                  <button
                    className="btn cancel"
                    onClick={cancel}
                    disabled={cancelling}
                    title={
                      run?.status === "waiting"
                        ? "结束这次任务；停在等待回答的那一步不会再继续"
                        : "停止当前任务；已产出的内容会保留"
                    }
                  >
                    {cancelling ? (
                      <span className="spinner" />
                    ) : run?.status === "waiting" ? (
                      "结束任务"
                    ) : (
                      "取消任务"
                    )}
                  </button>
                ) : (
                  <button className="btn primary" onClick={submit} disabled={busy || !draft.trim()}>
                    {busy ? <span className="spinner" /> : "发送"}
                  </button>
                )}
              </div>
            </div>
          )}
        </main>
      </div>

      {bigScreenOpen && (
        <AgentBigScreen
          run={run}
          title={bigScreenTitle}
          historyMode={Boolean(viewingHistory)}
          onClose={() => setBigScreenOpen(false)}
        />
      )}

      {/* Settings, over the workbench. The backdrop closes on click; the panel
          itself stops propagation so a click inside does not dismiss it. */}
      {settingsOpen && (
        <div className="settings-backdrop" onClick={closeSettings}>
          <div
            className="settings-panel"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-label="设置"
          >
            <div className="settings-head">
              <h2>设置</h2>
              <button
                className="settings-close"
                onClick={closeSettings}
                disabled={settingsBusy}
                title="关闭"
                aria-label="关闭设置"
              >
                ✕
              </button>
            </div>
            <Settings
              reviewMode={reviewMode}
              onReviewModeSaved={changeReviewMode}
              onDirtyChange={setSettingsDirty}
              onBusyChange={setSettingsBusy}
            />
          </div>
        </div>
      )}
    </div>
  );
}


/** "谁在工作、在做什么、已经多久" — the answer to a silent run.
 *
 * Streaming off means a stage can be quiet for a full minute: no reasoning, no
 * steps, nothing to append. This row is the only thing on screen that changes,
 * so it has to be true (real elapsed, real agent) and it has to keep moving.
 *
 * The trailing note explains *why* nothing is appearing, so it has to get the
 * reason right. It used to say "streaming is off" unconditionally — but the
 * switch lives in the backend's provider config, which this component does not
 * read, and a model that returns no `reasoning_content` (every non-reasoning
 * model) produces no live steps either. With streaming on, that sentence was
 * simply false. The backend now reports the switch; if it is on and still
 * nothing has arrived, the note names the real cause instead of the wrong one —
 * but only after a few seconds, because the first reasoning flush is throttle to
 * twice a second and a young stage is not evidence of anything. */
function WorkingTurn({
  name,
  label,
  seconds,
  streaming,
}: {
  name: string;
  label: string;
  seconds: number;
  streaming?: boolean;
}) {
  const note =
    streaming === false
      ? "未开启流式输出，内容会在这一步完成后一次性出现。"
      : streaming === true && seconds >= 6
        ? "该模型未返回思考过程，内容会在这一步完成后一次性出现。"
        : "内容会在这一步完成后一次性出现。";
  return (
    <div className="turn agent live working-turn">
      <div className="turn-head">
        <span className="turn-avatar">{agentInitial(name)}</span>
        <span className="turn-name">{name}</span>
        <span className="pulse small" />
        <span className="turn-tag">正在工作</span>
        <span className="working-elapsed mono" title="已用时">
          {formatElapsed(seconds)}
        </span>
      </div>
      <div className="turn-body">
        <div className="turn-text muted">
          {label || "处理中"}…
          <span className="working-note">{note}</span>
        </div>
      </div>
    </div>
  );
}

/** A stage that is still running: its reasoning grows in place as it streams. */
function LiveTurn({
  agent,
  name,
  steps,
  elapsed = 0,
}: {
  agent: string;
  name: string;
  steps: Array<Record<string, any>>;
  /** Elapsed seconds while this agent is the one the backend reports as busy. */
  elapsed?: number;
}) {
  const reasoning = steps.filter((s) => s.kind === "reasoning");
  const others = steps.filter((s) => s.kind !== "reasoning");
  const latest = reasoning.length ? reasoning[reasoning.length - 1].detail : "";

  return (
    <div className="turn agent live">
      <div className="turn-head">
        <span className="turn-avatar">{agentInitial(name)}</span>
        <span className="turn-name">{name}</span>
        <span className="pulse small" />
        <span className="turn-tag">进行中</span>
        {elapsed > 0 && (
          <span className="working-elapsed mono" title="已用时">
            {formatElapsed(elapsed)}
          </span>
        )}
      </div>
      {/* 与完成后的卡片一致：先说清这个岗位是干嘛的。 */}
      {AGENT_ABOUT[agent] && <div className="turn-about">{AGENT_ABOUT[agent]}</div>}
      <div className="turn-body">
        {latest && (
          <div className="reasoning-block">
            <div className="reasoning-head">
              <span className="step-icon">思考</span>
              思考中…
            </div>
            <div className="step-hint">{STEP_EXPLAIN["reasoning"]}</div>
            <div className="reasoning-body">{latest}</div>
          </div>
        )}
        {others.map((st, i) => (
          <div className="trace-step" key={i}>
            <span className="step-icon">{STEP_ICON[st.kind] || "·"}</span>
            <div className="step-body">
              <div className="step-title">{st.title}</div>
              {STEP_EXPLAIN[st.kind] && (
                <div className="step-hint">{STEP_EXPLAIN[st.kind]}</div>
              )}
              {st.detail && <div className="step-detail">{st.detail}</div>}
            </div>
          </div>
        ))}
        {!latest && !others.length && <div className="turn-text muted">{agent} 正在处理…</div>}
      </div>
    </div>
  );
}
