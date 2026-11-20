import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
} from "react";
import { settingsApi } from "../api/client";
import KnowledgePanel from "../components/KnowledgePanel";
import LogPanel from "../components/LogPanel";
import { MCPSettings } from "../components/SettingsPanels";
import { readMotionMode, writeMotionMode, type MotionMode } from "../motionMode";
import "../styles.css";

interface ProviderPublic {
  id: string;
  preset: string;
  label: string;
  base_url: string;
  model: string;
  vision_model: string;
  image_model: string;
  video_model: string;
  /** "" for a hosted provider, "gateway" for a local ComfyUI job queue. */
  kind?: string;
  timeout: number;
  temperature: number;
  has_key: boolean;
  key_hint: string;
  /** ok | missing | placeholder | suspicious — a shape check, not authentication. */
  key_verdict: string;
  /** Only present while editing; the stored key never travels to the browser. */
  api_key?: string;
}

interface ProviderDraft extends ProviderPublic {
  api_key: string;
}

function ProviderBusy({ label }: { label: string }) {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const started = Date.now();
    const timer = window.setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, []);
  return <><span className="provider-orbit" aria-hidden="true"><i /></span><span role="status">{label}… {seconds}秒</span></>;
}

interface AgentBinding {
  id: string;
  label: string;
  /** Capability chips declared by the backend, e.g. {key:"image", label:"需生图模型"}. */
  needs?: Array<{ key: string; label: string }>;
  resolved_provider: string;
}

interface Preset {
  id: string;
  label: string;
  base_url: string;
  model: string;
  vision_model: string;
  note: string;
}





/** Functional tabs; changing tabs preserves every draft in memory. */
const SECTIONS: Array<{
  id: string;
  label: string;
  icon: "key" | "agents" | "sliders" | "book" | "logs";
}> = [
  // 常规放第一位：打开设置最常动的是流式开关和审核模式，不该翻到底下找。
  { id: "call", label: "常规", icon: "sliders" },
  { id: "providers", label: "供应商", icon: "key" },
  { id: "binding", label: "Agent 配置", icon: "agents" },
  { id: "knowledge", label: "RAG 知识库", icon: "book" },
  { id: "mcp", label: "MCP 工具", icon: "agents" },
  // Placed last on purpose: it is the one tab nobody opens on a good day.
  { id: "logs", label: "日志", icon: "logs" },
];

/**
 * Nav icons, inline SVG for the same reasons as the rail: one self-contained
 * bundle, no webfont request, and identical rendering on every platform.
 */
function SettingsIcon({ name }: { name: "key" | "agents" | "sliders" | "book" | "logs" }) {
  const common = { width: 13, height: 13, viewBox: "0 0 16 16", fill: "none", "aria-hidden": true };
  if (name === "key") {
    return (
      <svg {...common}>
        <circle cx="6" cy="6" r="3.2" stroke="currentColor" strokeWidth="1.3" />
        <path d="M8.3 8.3 13 13M11 10.6l1.6-1.6" stroke="currentColor" strokeWidth="1.3" />
      </svg>
    );
  }
  if (name === "agents") {
    return (
      <svg {...common}>
        <circle cx="4" cy="4" r="2" stroke="currentColor" strokeWidth="1.3" />
        <circle cx="4" cy="12" r="2" stroke="currentColor" strokeWidth="1.3" />
        <circle cx="12" cy="8" r="2" stroke="currentColor" strokeWidth="1.3" />
        <path d="M6 4.6 10 7.4M6 11.4 10 8.6" stroke="currentColor" strokeWidth="1.3" />
      </svg>
    );
  }
  if (name === "book") {
    return (
      <svg {...common}>
        <path d="M2.4 3.2h4.1c.6 0 1.1.5 1.1 1.1v8.4c0-.6-.5-1.1-1.1-1.1H2.4zM13.6 3.2H9.5c-.6 0-1.1.5-1.1 1.1v8.4c0-.6.5-1.1 1.1-1.1h4.1z"
          stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
      </svg>
    );
  }
  if (name === "logs") {
    // A card with ruled lines: the shape of a log file, not a generic document.
    return (
      <svg {...common}>
        <rect x="2.8" y="2.4" width="10.4" height="11.2" rx="1.6" stroke="currentColor" strokeWidth="1.2" />
        <path d="M5.2 5.6h5.6M5.2 8h5.6M5.2 10.4h3.4" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
      </svg>
    );
  }
  return (
    <svg {...common}>
      <path d="M2 4.5h12M2 8h12M2 11.5h12" stroke="currentColor" strokeWidth="1.3" />
      <circle cx="6" cy="4.5" r="1.4" fill="currentColor" />
      <circle cx="10" cy="8" r="1.4" fill="currentColor" />
      <circle cx="7" cy="11.5" r="1.4" fill="currentColor" />
    </svg>
  );
}

/**
 * Settings — one page, five sections.
 *
 * Ordering follows what a new user can act on, cheapest first:
 *   ① 免费模型   — three zero-cost routes side by side: keyless endpoints,
 *                  the channel relay, the operator's shared model
 *   ② 供应商     — your own paid/custom endpoints
 *   ③ 分配与回退 — which agent uses which provider
 *
 * The relay is deliberately INSIDE this section rather than a separate page:
 * "how do I get a free model" has three answers, and splitting one of them onto
 * another route hid it from anyone who did not already know it existed.
 */
/**
 * What leaving the vision model empty actually costs.
 *
 * Two different consequences, and only one of them is a fallback:
 *
 *   · The pipeline does NOT fail over this. `LLMConfig.effective_vision_model`
 *     falls back to the text model, so a request still goes out — but a text
 *     model has no image input, so the director cannot actually look at the frame
 *     it is asked about. The visible symptom is a verdict written from the
 *     prompt's description of the picture rather than from the picture, which
 *     reads as a normal answer and is therefore the more dangerous outcome.
 *
 *   · The two agents that need it are named, because "有 Agent 需要" beside the
 *     label does not say which, and a user cannot judge the risk of a blank
 *     field without knowing who depends on it.
 *
 * Stated where the field is, not in a global banner: the decision is made here.
 */
function visionHint(value: string): string {
  if (value) return "用于看图的 Agent（导演、质量审查）会使用这个模型。";
  return (
    "留空时不会报错，但会退回到文本模型——文本模型没有图像输入，" +
    "导演与质量审查就只能依据文字描述评审，无法真正看图。" +
    "如果需要按画面评审，请填一个多模态模型（如 glm-4v-plus）。"
  );
}

/**
 * How to describe a provider's key in the header chip.
 *
 * Three states, and the middle one is the one that was missing: a value that is
 * present but obviously fake. Reporting it as "configured" left the user staring
 * at a green chip and a red run failure with no way to connect the two.
 */
function keyChip(p: ProviderPublic): { tone: string; label: string; title: string } {
  // A key being typed right now wins: the user needs to see that their edit is
  // recognised before they save.
  if (p.api_key) {
    return { tone: "ok", label: "已填 Key（未保存）", title: "填写后需点「保存设置」才生效" };
  }
  if (!p.has_key) {
    return { tone: "warn", label: "未配置 Key", title: "这个供应商没有 Key，无法调用" };
  }
  if (p.key_verdict === "placeholder") {
    return {
      tone: "fail",
      label: "Key 是占位符",
      title: `保存的 Key 看起来是占位值（${p.key_hint}），不是真实密钥。请粘贴真实 Key 后保存。`,
    };
  }
  if (p.key_verdict === "suspicious") {
    return {
      tone: "warn",
      label: "Key 过短，请确认",
      title: `保存的 Key 只有 ${p.key_hint}，明显短于任何供应商的真实密钥。请核对。`,
    };
  }
  return { tone: "ok", label: `已配置 ${p.key_hint}`, title: `已保存 Key ${p.key_hint}` };
}

/**
 * A model field that accepts both a scanned suggestion and a typed name.
 *
 * Why not a plain `<select>`: model catalogues come from a live scan of the
 * provider, and a scan fails whenever the key is wrong, expired, or the endpoint
 * refuses `/models` — a 401 leaves the list empty. A `<select>` with an empty list
 * is unusable: the user cannot pick a model AND cannot type one, so the only path
 * forward is to fix the key before doing anything else. But the model name is
 * known to the user (they bought `glm-4v-plus`, the vendor's page says so), and
 * making them wait on a scan that depends on the very key they are trying to
 * configure is backwards.
 *
 * `<datalist>` is the right primitive: the browser offers the scanned ids as
 * suggestions while the text field stays free-form. It degrades to a plain input
 * where datalist is unsupported, which is still usable.
 */
function ModelField({
  label,
  value,
  onChange,
  options,
  placeholder = "输入模型名，或从列表中选择",
  hint,
  onScan,
  scanning,
  scanError,
}: {
  label: ReactNode;
  value: string;
  onChange: (next: string) => void;
  options: Array<{ id: string; reason?: string }>;
  placeholder?: string;
  hint?: string;
  onScan?: () => void;
  scanning?: boolean;
  scanError?: string;
}) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  /** 用户正在输入（而不是刚打开列表）。只有这时才按输入内容过滤。 */
  const [typing, setTyping] = useState(false);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  // 比较前先收空白：粘贴进来的模型名常带一个看不见的尾随空格，而列表是按
  // 子串比的——「big-pickle 」不会命中「big-pickle」，界面上却一字不差。
  const typed = value.trim();
  const known = options.some((o) => o.id.trim().toLowerCase() === typed.toLowerCase());

  // 打开列表时先给全部，只有真的在打字才过滤。
  //
  // 之前是「框里有值就过滤」，于是选过一次之后框里正好是完整 id，再打开就只剩
  // 那一条——用户看到的像是「扫描到 33 个，下拉只有 1 个」。清空输入框能看到
  // 全部，但没人会想到要这么做。
  const visible =
    typing && typed
      ? options.filter((o) => o.id.toLowerCase().includes(typed.toLowerCase()))
      : options;
  // 一个都匹配不上时，把列表前几个摆出来。原来这里只有一句「没有匹配」，
  // 用户既不知道为什么，也没有下一步——而答案往往就在那 33 个名字里。
  const sample = typing && typed && visible.length === 0 ? options.slice(0, 8) : [];

  // Close on an outside click. Registered on the document because the list floats
  // over other fields, and a click on the hint below must dismiss it rather than
  // leave it hanging over the form.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  /**
   * Keyboard control, because a mouse-only dropdown over a text field is worse
   * than the native one it replaces.
   *
   *   ↓ / ↑   move through the filtered list
   *   Enter   accept the highlighted entry
   *   Esc     close without changing the value
   */
  function onKeyDown(e: ReactKeyboardEvent<HTMLInputElement>) {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      if (!open) {
        setOpen(true);
        return;
      }
      if (!visible.length) return;
      e.preventDefault();
      const step = e.key === "ArrowDown" ? 1 : -1;
      setActive((i) => {
        const next = i + step;
        if (next < 0) return visible.length - 1;
        if (next >= visible.length) return 0;
        return next;
      });
    } else if (e.key === "Enter" && open && active >= 0 && visible[active]) {
      e.preventDefault();
      onChange(visible[active].id);
      setOpen(false);
      setTyping(false);
      setActive(-1);
    } else if (e.key === "Escape") {
      setOpen(false);
      setActive(-1);
    }
  }

  return (
    <div className="field">
      <label>{label}</label>
      <div className="model-field" ref={wrapRef}>
        <div className="model-combo">
          <input
            className="input"
            value={value}
            placeholder={placeholder}
            onChange={(e) => {
              onChange(e.target.value);
              setOpen(true);
              setTyping(true);
              setActive(-1);
            }}
            onFocus={() => {
              setOpen(true);
              // 聚焦是「我要挑一个」，不是「我在搜索」：先给全量列表。
              setTyping(false);
            }}
            // 离开时收掉首尾空白：模型名是要原样发给供应商的，多一个空格就是
            // 另一个名字，「找不到模型」会以 4xx 的形式在生成阶段才炸出来。
            onBlur={() => {
              if (value !== value.trim()) onChange(value.trim());
            }}
            onKeyDown={onKeyDown}
            spellCheck={false}
            autoComplete="off"
            role="combobox"
            aria-expanded={open}
            aria-autocomplete="list"
          />
          {/* The chevron doubles as the touch target: tapping a text field does
              not always raise a keyboard, and without it a phone user has no way
              to reveal the list. */}
          <button
            type="button"
            className="model-combo-toggle"
            onClick={() => {
              setOpen((v) => !v);
              // 点箭头等于「打开清单」，不是搜索：把过滤撤掉，否则只会看到当前
              // 这一个模型，看起来像扫描结果丢了。
              setTyping(false);
              setActive(-1);
            }}
            aria-label="展开模型列表"
            tabIndex={-1}
          >
            ▾
          </button>

          {open && (
            <div className="model-combo-list" role="listbox">
              {visible.length === 0 ? (
                <>
                  <div className="model-combo-empty">
                    {options.length === 0
                      ? "还没有扫描到模型。可以直接输入模型名，或点「扫描模型」。"
                      : `没有匹配「${typed}」：这次扫描回来的 ${options.length} 个模型里没有它。`}
                  </div>
                  {sample.map((o) => (
                    <button
                      type="button"
                      role="option"
                      aria-selected={false}
                      className="model-combo-item"
                      key={o.id}
                      title={o.reason}
                      // 与下面的列表同一个理由：onClick 会先触发 input 的 blur，
                      // 列表在点击落地前就关了。
                      onMouseDown={(e) => {
                        e.preventDefault();
                        onChange(o.id);
                        setOpen(false);
                        setTyping(false);
                        setActive(-1);
                      }}
                    >
                      <span className="model-combo-id">{o.id}</span>
                      {o.reason && <span className="model-combo-reason">{o.reason}</span>}
                    </button>
                  ))}
                </>
              ) : (
                visible.map((o, i) => (
                  <button
                    type="button"
                    role="option"
                    aria-selected={i === active}
                    className={`model-combo-item ${i === active ? "active" : ""} ${
                      o.id === value ? "selected" : ""
                    }`}
                    key={o.id}
                    title={o.reason}
                    // `onMouseDown` rather than `onClick`: the input's blur would
                    // close the list before the click lands.
                    onMouseDown={(e) => {
                      e.preventDefault();
                      onChange(o.id);
                      setOpen(false);
                      setTyping(false);
                      setActive(-1);
                    }}
                    onMouseEnter={() => setActive(i)}
                  >
                    <span className="model-combo-id">{o.id}</span>
                    {o.reason && <span className="model-combo-reason">{o.reason}</span>}
                  </button>
                ))
              )}
            </div>
          )}
        </div>
        {onScan && (
          <button
            className="btn ghost small"
            onClick={onScan}
            disabled={scanning}
            title="按当前地址与 Key 扫描可用模型"
          >
            {scanning ? <span className="spinner" /> : "扫描"}
          </button>
        )}
      </div>
      {/* A typed value the scan did not return is not an error: the scan may have
          failed, or the model may be new. Saying nothing is better than a warning
          the user cannot act on. */}
      {value && !known && !scanError && options.length > 0 && (
        <div className="field-note muted">
          这个名字不在扫描结果里（共 {options.length} 个）。保存后会原样发给供应商，
          名字不对会在调用时报错——展开下拉可以看列表里有哪些。
        </div>
      )}
      {scanError && <div className="field-note err">{scanError}</div>}
      {hint && <div className="field-note muted">{hint}</div>}
    </div>
  );
}

/** Where the scanned model catalogue is remembered between dialog openings. */
const CATALOGUE_KEY = "mancu.settings.catalogues";

export default function Settings({
  reviewMode = "FAST", onReviewModeSaved, onDirtyChange, onBusyChange,
}: {
  reviewMode?: string;
  onReviewModeSaved?: (mode: string) => void;
  onDirtyChange?: (dirty: boolean) => void;
  onBusyChange?: (busy: boolean) => void;
}) {
  const [reviewDraft, setReviewDraft] = useState(reviewMode);
  const [savedReview, setSavedReview] = useState(reviewMode);
  const [baseline, setBaseline] = useState("");
  const [visited, setVisited] = useState<string[]>(["providers"]);
  const [providers, setProviders] = useState<ProviderDraft[]>([]);
  const [agents, setAgents] = useState<AgentBinding[]>([]);
  const [assignments, setAssignments] = useState<Record<string, string>>({});
  const [fallbacks, setFallbacks] = useState<Record<string, string[]>>({});
  const [defaultProvider, setDefaultProvider] = useState("");
  // 是否流式输出。默认开：思考一分钟的阶段应当能看到进度。
  const [stream, setStream] = useState(true);
  // 大屏光流动效。默认「强制开」；纯前端偏好，改动即时生效、不进后端保存。
  const [motionMode, setMotionMode] = useState<MotionMode>(readMotionMode);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [visionAgents, setVisionAgents] = useState<string[]>([]);
  /**
   * provider id → models grouped by capability. Filled by "扫描模型".
   *
   * Persisted to localStorage, because the dialog unmounts when it closes and a
   * React state survives only as long as the component. Closing and reopening
   * used to wipe the whole catalogue: every model dropdown shrank back to the one
   * saved value plus "（不配置）", which reads as "the scan found almost nothing"
   * even right after a scan that found plenty.
   *
   * A scan costs a network round trip and needs a working key, so throwing the
   * result away on a window close is the expensive half of the bug.
   */
  const [catalogues, setCatalogues] = useState<
    Record<string, Record<string, Array<{ id: string; reason: string }>>>
  >(() => {
    try {
      const raw = window.localStorage.getItem(CATALOGUE_KEY);
      const parsed = raw ? JSON.parse(raw) : {};
      return parsed && typeof parsed === "object" ? parsed : {};
    } catch {
      // A corrupt or unavailable store must not stop the dialog from opening:
      // the catalogue is a convenience, not required state.
      return {};
    }
  });
  /** provider id → scan error, shown inline rather than as a silent empty list. */
  const [scanErrors, setScanErrors] = useState<Record<string, string>>({});

  // Persist the catalogue whenever it changes.
  //
  // An effect rather than a write inside `scanProvider`: the scan updates the
  // catalogue on both the success and the failure path, and a call site would
  // miss one of them. localStorage can throw when the quota is full or storage is
  // blocked, and a settings dialog must not die over a convenience cache.
  useEffect(() => {
    try {
      window.localStorage.setItem(CATALOGUE_KEY, JSON.stringify(catalogues));
    } catch {
      /* not fatal: the list simply does not survive this dialog closing */
    }
  }, [catalogues]);
  const [scanning, setScanning] = useState<string>("");
  /**
   * Every pickable model, grouped by capability, each tagged with its provider.
   *
   * The table needs this because an agent's requirement is a *capability*, not a
   * vendor: "视觉生成 needs a video model" must offer the video models that
   * exist, each labelled with where it lives.
   */
  const [modelOptions, setModelOptions] = useState<
    Record<string, Array<{ value: string; display: string; provider_id: string; model: string }>>
  >({});
  /**
   * Provider-first grouping for the assignment dropdown.
   *
   * Shape: [{provider_id, provider_label, models:[{value, model, capability, label}]}]
   *
   * Kept separate from `modelOptions` because the two answer different questions:
   * `modelOptions` asks "is there an image model anywhere?", while this asks
   * "what does this supplier offer?" — the dropdown needs the latter.
   */
  const [providerGroups, setProviderGroups] = useState<
    Array<{
      provider_id: string;
      provider_label: string;
      models: Array<{ value: string; model: string; capability: string; label: string }>;
    }>
  >([]);


  // ---- relay state (embedded, no longer a separate page) ----

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState("");
  /** 哪些供应商的 Key 当前以明文显示。默认全部遮住，逐条记住。 */
  const [revealedKeys, setRevealedKeys] = useState<Record<string, boolean>>({});
  const [message, setMessage] = useState<{ kind: "ok" | "err"; text: string } | null>(null);
  /** Why the last save failed, shown inline beside the save button. */
  const [failure, setFailure] = useState("");
  const [activeSection, setActiveSection] = useState<string>(SECTIONS[0].id);
  const mainRef = useRef<HTMLDivElement | null>(null);
  const modelSection = ["providers", "binding", "call"].includes(activeSection);
  const draftSnapshot = JSON.stringify({ providers, assignments, fallbacks, defaultProvider, stream });
  const modelDirty = (!!baseline && draftSnapshot !== baseline) || reviewDraft !== savedReview;

  useEffect(() => { onDirtyChange?.(modelDirty); }, [modelDirty, onDirtyChange]);
  useEffect(() => { onBusyChange?.(saving); }, [saving, onBusyChange]);

  function jumpTo(id: string) {
    setActiveSection(id);
    setVisited(prev => prev.includes(id) ? prev : [...prev, id]);
    if (mainRef.current) mainRef.current.scrollTop = 0;
  }

  const load = useCallback(async () => {
    try {
      // Only what this page renders: providers, their bindings and fallbacks.
      // The free-model catalogue and channel relay were removed from this page,
      // so fetching their endpoints here would be dead weight on every load.
      const [cfg, pre] = await Promise.all([settingsApi.get(), settingsApi.presets()]);
      setProviders(
        (cfg.providers as unknown as ProviderPublic[]).map((p) => ({ ...p, api_key: "" })),
      );
      setBaseline(JSON.stringify({
        providers: (cfg.providers as unknown as ProviderPublic[]).map(p => ({ ...p, api_key: "" })),
        assignments: cfg.assignments, fallbacks: cfg.fallbacks ?? {},
        defaultProvider: cfg.default_provider, stream: cfg.stream !== false,
      }));
      setAssignments(cfg.assignments);
      setFallbacks(cfg.fallbacks ?? {});
      setDefaultProvider(cfg.default_provider);
      setStream(cfg.stream !== false);
      setAgents(cfg.agents);
      setModelOptions(cfg.model_options || {});
      setProviderGroups(cfg.provider_groups || []);
      setPresets(pre.presets as unknown as Preset[]);
      setVisionAgents(pre.vision_agents);
      // Returned so the caller can compare what it asked for with what the
      // server actually stored. Without that comparison a dropped field looks
      // exactly like the user's click being ignored.
      return cfg;
    } catch (e: unknown) {
      setMessage({ kind: "err", text: e instanceof Error ? e.message : String(e) });
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // ---------------------------------------------------------------- free models

  // ---------------------------------------------------------------- relay

  /** Add the relay as a provider so agents can bind to it in one click. */

  // ---------------------------------------------------------------- providers

  function addProvider(presetId: string) {
    const preset = presets.find((p) => p.id === presetId);
    const id = `p${Date.now().toString(36)}`;
    setProviders((prev) => [
      ...prev,
      {
        id,
        preset: presetId,
        label: preset?.label ?? presetId,
        base_url: preset?.base_url ?? "",
        model: preset?.model ?? "",
        vision_model: preset?.vision_model ?? "",
  image_model: preset?.image_model ?? "",
  video_model: preset?.video_model ?? "",
        timeout: 60,
        temperature: 0.4,
        has_key: false,
        key_hint: "",
        api_key: "",
      },
    ]);
    if (!defaultProvider) setDefaultProvider(id);
  }

  function updateProvider(id: string, patch: Partial<ProviderDraft>) {
    setProviders((prev) => prev.map((p) => (p.id === id ? { ...p, ...patch } : p)));
    // 地址或 Key 一改，上一次扫描的目录就不再属于这个供应商。留着它比清空更糟：
    // 下拉里给出的是上一个中转站的模型名，选中后必然失败，而界面看起来一切正常。
    if ("base_url" in patch || "api_key" in patch) {
      setCatalogues((prev) => {
        if (!(id in prev)) return prev;
        const next = { ...prev };
        delete next[id];
        return next;
      });
      setScanErrors((prev) => {
        if (!(id in prev)) return prev;
        const next = { ...prev };
        delete next[id];
        return next;
      });
    }
  }

  /**
   * 收掉地址和 Key 的首尾空白。
   *
   * 从聊天窗口、文档里复制出来的值常常带着换行或空格，肉眼看不出来，但拼出来的
   * URL 已经不是一个地址了——「扫描失败: Failed to fetch」就是这么来的（后端只
   * 在保存时 strip，而测试连接、扫描模型用的都是还没保存的草稿值）。
   *
   * 只在离开输入框时收：边输边收会把「http://a.com/ v1」中间的意图也吃掉，
   * 而且用户按空格的瞬间字符消失会以为输入法坏了。
   */
  function trimProviderField(id: string, field: "base_url" | "api_key") {
    setProviders((prev) =>
      prev.map((p) => {
        if (p.id !== id) return p;
        const value = p[field];
        if (typeof value !== "string") return p;
        const clipped = value.trim();
        return clipped === value ? p : { ...p, [field]: clipped };
      }),
    );
  }

  function removeProvider(id: string) {
    setProviders((prev) => prev.filter((p) => p.id !== id));
    setAssignments((prev) => {
      const next = { ...prev };
      for (const k of Object.keys(next)) if (next[k] === id) delete next[k];
      return next;
    });
    setFallbacks((prev) => {
      const next: Record<string, string[]> = {};
      for (const [agent, ids] of Object.entries(prev)) {
        const kept = ids.filter((x) => x !== id);
        if (kept.length) next[agent] = kept;
      }
      return next;
    });
    if (defaultProvider === id) setDefaultProvider("");
  }

  async function saveProviders() {
    setSaving(true);
    setMessage(null);
    // Clear the previous attempt's notice: a stale "保存失败" left beside the
    // button while a new save is in flight reads as the new one failing.
    setFailure("");
    try {
      await settingsApi.save({
        providers: providers.map((p) => ({
          id: p.id,
          preset: p.preset,
          label: p.label,
          // 再收一次：用户可能填完直接点保存，中间没有离开输入框。
          base_url: p.base_url.trim(),
          api_key: p.api_key.trim(),
          model: p.model,
          vision_model: p.vision_model,
          image_model: p.image_model ?? "",
          video_model: p.video_model ?? "",
          // Sent back on every save. Without it the round trip silently erased a
          // gateway marker, and the run then used the hosted client against an
          // address that does not speak that protocol.
          kind: p.kind ?? "",
          timeout: p.timeout,
          temperature: p.temperature,
        })),
        assignments,
        default_provider: defaultProvider,
        fallbacks,
        stream,
      });
      const saved = await load();
      if (!saved) throw new Error("配置已提交，但重新读取失败，请重试确认保存结果。");
      setFailure("");
      // Compare what was sent with what came back. If the flag was dropped, the
      // save still succeeded — everything else is stored — so this must not be
      // reported as a failure, but the checkbox must not silently snap back to
      // 「开」either. That revert is exactly what an older backend process does:
      // its payload model has no `stream` field, so Pydantic discards it.
      if (saved && typeof saved.stream === "boolean" && saved.stream !== stream) {
        const text = stream
          ? "已保存，但「流式输出」没能开启：后端返回的设置里没有这一项。重启服务后即可保存。"
          : "已保存，但「流式输出」没能关闭：后端返回的设置里没有这一项。重启服务后即可保存。";
        setFailure(text);
        setMessage({ kind: "err", text });
        return;
      }
      onReviewModeSaved?.(reviewDraft);
      setSavedReview(reviewDraft);
      setMessage({ kind: "ok", text: "已保存，下一个任务使用新的设置。" });
    } catch (e: unknown) {
      const text = e instanceof Error ? e.message : String(e);
      // Stored in its own state rather than only in `message`: the banner at the
      // top of the dialog is not where the user is looking after clicking save,
      // and the save bar renders this beside the button too.
      setFailure(text);
      setMessage({ kind: "err", text });
    } finally {
      setSaving(false);
    }
  }

  async function test(id: string) {
    setTesting(id);
    setMessage(null);
    setScanErrors((prev) => {
      if (!(id in prev)) return prev;
      const next = { ...prev };
      delete next[id];
      return next;
    });
    try {
      const p = providers.find((x) => x.id === id);
      if (!p) throw new Error("供应商已删除");
      const res = await settingsApi.test({
        ...(p.has_key ? { provider_id: id } : {}),
        base_url: p.base_url.trim(), api_key: p.api_key.trim(),
        model: p.model, timeout: p.timeout,
      });
      if (res.ok) {
        setMessage({ kind: "ok", text: `连接成功：${res.model} 返回「${res.reply}」` });
      } else {
        const text = res.error ?? "连接失败";
        setMessage({ kind: "err", text });
        // 顶部那条横幅在滚动到第二三个供应商时已经看不见了，所以同一句话
        // 也落在按钮下面：点了按钮就要在按钮附近得到答复。
        setScanErrors((prev) => ({ ...prev, [id]: text }));
      }
    } catch (e: unknown) {
      const text = e instanceof Error ? e.message : String(e);
      setMessage({ kind: "err", text });
      setScanErrors((prev) => ({ ...prev, [id]: text }));
    } finally {
      setTesting("");
    }
  }

  /**
   * Scan one provider's catalogue.
   *
   * A saved provider is rescanned by id (its key stays server-side); an unsaved
   * one sends base_url + the draft key so a key can be checked before saving.
   */
  async function scanProvider(p: ProviderDraft) {
    setScanning(p.id);
    setScanErrors((prev) => ({ ...prev, [p.id]: "" }));
    try {
      const res = await settingsApi.scanModels({
        ...(p.has_key ? { provider_id: p.id } : {}),
        base_url: p.base_url.trim(), api_key: p.api_key.trim(), timeout: p.timeout,
      });
      if (!res.ok) {
        setScanErrors((prev) => ({ ...prev, [p.id]: res.error || "扫描失败" }));
        setCatalogues((prev) => ({ ...prev, [p.id]: {} }));
        return;
      }
      setCatalogues((prev) => ({ ...prev, [p.id]: res.groups || {} }));
      setMessage({ kind: "ok", text: `扫描完成：${p.label} 返回 ${res.total ?? 0} 个模型` });
      // Fill an empty field with the first model of that capability, so a scan
      // immediately produces a usable configuration instead of a list to click.
      const groups = res.groups || {};
      const patch: Partial<ProviderDraft> = {};
      const first = (cap: string) => (groups[cap] || [])[0]?.id;
      if (!p.model && first("text")) patch.model = first("text");
      if (!p.vision_model && first("vision")) patch.vision_model = first("vision");
      if (!p.image_model && first("image")) patch.image_model = first("image");
      if (!p.video_model && first("video")) patch.video_model = first("video");
      if (Object.keys(patch).length) updateProvider(p.id, patch);
    } catch (err) {
      setScanErrors((prev) => ({
        ...prev,
        [p.id]: err instanceof Error ? err.message : "扫描失败",
      }));
    } finally {
      setScanning("");
    }
  }

  function toggleFallback(agentId: string, providerId: string) {
    setFallbacks((prev) => {
      const ids = prev[agentId] ?? [];
      const next = ids.includes(providerId)
        ? ids.filter((x) => x !== providerId)
        : [...ids, providerId];
      const out = { ...prev };
      if (next.length) out[agentId] = next;
      else delete out[agentId];
      return out;
    });
  }

  if (loading) {
    return (
      <div className="settings-dialog">
        <div className="settings-nav" />
        <div className="settings-main">
          <p className="muted">正在加载设置…</p>
        </div>
      </div>
    );
  }

  return (
    /* The shell is a dialog, not a page: settings open over the workbench so the
       user does not lose the conversation they were reading. The side nav mirrors
       the panels below so jumping between them does not require scrolling past
       two hundred lines of provider fields. */
    <div className="settings-dialog">
      <nav className="settings-nav" aria-label="设置分类">
        <div className="settings-nav-title">设置</div>
        {SECTIONS.map((s) => (
          <button
            key={s.id}
            type="button"
            aria-current={activeSection === s.id ? "page" : undefined}
            aria-controls={`settings-view-${["providers", "binding", "call"].includes(s.id) ? "models" : s.id}`}
            className={`settings-nav-item ${activeSection === s.id ? "active" : ""}`}
            onClick={() => jumpTo(s.id)}
          >
            <span className="settings-nav-icon"><SettingsIcon name={s.icon} /></span>
            <span>{s.label}</span>
          </button>
        ))}
      </nav>

      <div className="settings-workspace">
      <div className="settings-view" id="settings-view-models" hidden={!modelSection}>
      <div className="settings-main" ref={mainRef}>
        <fieldset className="settings-fields" disabled={saving}>
        {message && (
          <div className={`banner ${message.kind === "ok" ? "ok" : "fail"}`}>{message.text}</div>
        )}


      {/* ================= ① 供应商 ================= */}
      <section className="panel" id="sec-providers" hidden={activeSection !== "providers"}>
        <h2>供应商</h2>
        <p className="hint">
          统一管理模型服务的地址、Key 和可用模型；在「Agent 配置」中分配给不同角色。已保存的 Key 不会以明文回传。
        </p>

        {providers.length === 0 && (
          <p className="muted" style={{ marginBottom: 12 }}>
            还没有供应商。请使用下方的供应商预设添加配置。
          </p>
        )}

        {providers.map((p) => {
          const needsVision = visionAgents.some(
            (a) => assignments[a] === p.id || (!assignments[a] && defaultProvider === p.id),
          );
          return (
            <div className="provider-card" key={p.id}>
              <div className="provider-head">
                <input
                  className="input provider-label"
                  value={p.label}
                  onChange={(e) => updateProvider(p.id, { label: e.target.value })}
                  placeholder="显示名称"
                />
                {/* The chip reports what the key actually is, not merely that the
                    field is non-empty. It used to show a green "已配置" for any
                    stored value — including the scratch `t-key` left behind by a
                    test — while every run failed with "尚未配置 LLM". */}
                <span className={`chip ${keyChip(p).tone}`} title={keyChip(p).title}>
                  <span className="dot" />
                  {keyChip(p).label}
                </span>
                <button
                  className="btn ghost small"
                  onClick={() => test(p.id)}
                  disabled={Boolean(testing)}
                  aria-busy={testing === p.id}
                >
                  {/* 只留一个转圈看起来像"按钮内容没了"。带着动作词，用户才知道
                      它在做什么；25 秒的等待也就有了意义。 */}
                  {testing === p.id ? (
                    <ProviderBusy label="测试中" />
                  ) : "测试连接"}
                </button>
                {/* One scan fills the four model dropdowns below. Placed next to
                    "测试连接" because it is the other key-check action. */}
                <button
                  className="btn ghost small"
                  onClick={() => scanProvider(p)}
                  disabled={Boolean(scanning) || !p.base_url}
                  aria-busy={scanning === p.id}
                  title="列出这个 Key 可用的所有模型，按能力分组"
                >
                  {scanning === p.id ? (
                    <ProviderBusy label="扫描中" />
                  ) : "扫描模型"}
                </button>
                <button className="btn ghost small danger" onClick={() => removeProvider(p.id)}>
                  删除
                </button>
              </div>

              {/* 结果要落在按钮附近。原来这条提示在卡片最底部，而卡片有四个
                  模型下拉那么长——点了上面的按钮，答复出现在三百像素以外，
                  读起来就是"什么都没发生"。 */}
              {scanErrors[p.id] && (
                <div className="hint provider-action-error" style={{ color: "var(--fail)" }}>
                  失败：{scanErrors[p.id]}
                </div>
              )}

              <div className="row2">
                <div className="field">
                  <label>接口地址（OpenAI 兼容）</label>
                  <input
                    className="input"
                    value={p.base_url}
                    onChange={(e) => updateProvider(p.id, { base_url: e.target.value })}
                    onBlur={() => trimProviderField(p.id, "base_url")}
                    placeholder="https://api.example.com/v1"
                    spellCheck={false}
                  />
                </div>
                <div className="field">
                  <label>API Key</label>
                  <div className="input-affix">
                    <input
                      className="input"
                      type={revealedKeys[p.id] ? "text" : "password"}
                      value={p.api_key}
                      onChange={(e) => updateProvider(p.id, { api_key: e.target.value })}
                      onBlur={() => trimProviderField(p.id, "api_key")}
                      placeholder={p.has_key ? "已保存，留空则不修改" : "sk-..."}
                      spellCheck={false}
                      autoComplete="off"
                    />
                    <button
                      type="button"
                      className="input-affix-btn"
                      onClick={() =>
                        setRevealedKeys((prev) => ({ ...prev, [p.id]: !prev[p.id] }))
                      }
                      // 按钮只切显示方式，不改值：要能看见自己粘了什么，
                      // 但不能因为看了一眼就把明文留在界面上。
                      title={revealedKeys[p.id] ? "隐藏 Key" : "显示 Key"}
                      aria-label={revealedKeys[p.id] ? "隐藏 API Key" : "显示 API Key"}
                      aria-pressed={!!revealedKeys[p.id]}
                    >
                      {revealedKeys[p.id] ? (
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M3 3l18 18" />
                          <path d="M10.6 10.6a2 2 0 002.8 2.8" />
                          <path d="M9.9 5.2A10.6 10.6 0 0112 5c5 0 9 4.5 9 7 0 .9-.6 2.1-1.6 3.2" />
                          <path d="M6.2 6.5C4.2 8 3 9.9 3 12c0 2.5 4 7 9 7 1.3 0 2.5-.3 3.6-.8" />
                        </svg>
                      ) : (
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M3 12s3.5-7 9-7 9 7 9 7-3.5 7-9 7-9-7-9-7z" />
                          <circle cx="12" cy="12" r="2.6" />
                        </svg>
                      )}
                    </button>
                  </div>
                </div>
              </div>

              <div className="row2">
                <ModelField
                  label="文本模型"
                  value={p.model}
                  onChange={(v) => updateProvider(p.id, { model: v })}
                  options={catalogues[p.id]?.text || []}
                  hint="留空表示该供应商不提供文本模型。"
                />
                <ModelField
                  label={
                    <>
                      视觉模型
                      {needsVision && (
                        <span className="chip warn" style={{ marginLeft: 6 }}>
                          有 Agent 需要
                        </span>
                      )}
                    </>
                  }
                  value={p.vision_model}
                  onChange={(v) => updateProvider(p.id, { vision_model: v })}
                  options={catalogues[p.id]?.vision || []}
                  hint={visionHint(p.vision_model)}
                />
              </div>

              {/* Media models live on their own row: a text provider and a media
                  provider are usually separate accounts, and mixing the fields
                  invites typing a text model name into an image field. */}
              <div className="field-row">
                <ModelField
                  label={
                    <>
                      生图模型
                      {p.image_model ? null : <span className="chip">可留空</span>}
                    </>
                  }
                  value={p.image_model}
                  onChange={(v) => updateProvider(p.id, { image_model: v })}
                  options={catalogues[p.id]?.image || []}
                  hint="文生图。留空表示该供应商不提供生图。"
                />
                <ModelField
                  label={
                    <>
                      生视频模型
                      {p.video_model ? null : <span className="chip">可留空</span>}
                    </>
                  }
                  value={p.video_model}
                  onChange={(v) => updateProvider(p.id, { video_model: v })}
                  options={catalogues[p.id]?.video || []}
                  hint="图生视频。留空表示该供应商不提供生视频。"
                />
              </div>

              {/* Scan feedback: how many models were found and in which groups.
                  A scan that finds nothing must say so — an empty dropdown with
                  no explanation reads as "this provider has no models".
                  失败提示不在这里：它跟着按钮走，见上面。 */}
              {catalogues[p.id] && Object.keys(catalogues[p.id]).length > 0 && (
                <div className="hint">
                  扫描到{" "}
                  {Object.entries(catalogues[p.id])
                    .filter(([, list]) => list.length > 0)
                    .map(([cap, list]) => {
                      const name =
                        cap === "text"
                          ? "文本"
                          : cap === "vision"
                            ? "视觉"
                            : cap === "image"
                              ? "生图"
                              : "生视频";
                      return `${name} ${list.length}`;
                    })
                    .join("，")}
                  。下拉里选一个模型即可。
                </div>
              )}

              {(p.image_model || p.video_model) && (
                <div className="hint" style={{ marginTop: -4 }}>
                  这个供应商具备媒体能力，生成阶段会优先用它。模型名必须与供应商一致，
                  否则报「不提供模型」。
                </div>
              )}
            </div>
          );
        })}

        <div className="preset-row">
          <span className="muted">添加供应商：</span>
          {presets.map((preset) => (
            <button
              key={preset.id}
              className="preset-btn"
              onClick={() => addProvider(preset.id)}
              title={preset.note}
            >
              + {preset.label}
            </button>
          ))}
        </div>
      </section>

      {/* ================= ② 调用方式 ================= */}
      <section className="panel" id="sec-call" hidden={activeSection !== "call"}>
        <h2>常规</h2>
        <p className="hint">
          设置模型输出方式、审核模式与界面动效；图片与视频生成开关仍在工作台中选择。
        </p>

        <label className="call-switch">
          <input
            type="checkbox"
            checked={stream}
            onChange={(e) => setStream(e.target.checked)}
          />
          <span className="call-switch-text">
            <b>流式输出</b>
            <span className="call-switch-note">
              {stream
                ? "开启：思考过程实时滚动。代价是回复若在 max_tokens 处被截断，流式那一遍会白跑一次，再用非流式重试。"
                : "关闭：等模型一次说完再返回，界面在这段时间没有进度。好处是截断时只重试一次，不会多花一遍时间和 token。"}
            </span>
          </span>
        </label>
        <div className="field settings-review-field">
          <label htmlFor="settings-review-mode">审核模式</label>
          <select id="settings-review-mode" className="input" value={reviewDraft} onChange={e => setReviewDraft(e.target.value)}>
            <option value="STRICT">严格审核</option>
            <option value="NORMAL">普通审核</option>
            <option value="FAST">快速审核</option>
          </select>
          <p className="hint">{reviewDraft === "STRICT" ? "P0–P2 问题会触发返工，P3 只记录。" : reviewDraft === "FAST" ? "仅 P0 问题阻断，适合快速查看草稿。" : "P0、P1 问题触发返工，P2、P3 只记录。"}</p>
        </div>
        <div className="field settings-review-field">
          <label htmlFor="settings-motion-mode">大屏光流动效</label>
          <select
            id="settings-motion-mode"
            className="input"
            value={motionMode}
            onChange={(e) => {
              const next = e.target.value as MotionMode;
              setMotionMode(next);
              writeMotionMode(next);
            }}
          >
            <option value="on">开启（默认）</option>
            <option value="system">跟随系统</option>
            <option value="off">关闭</option>
          </select>
          <p className="hint">
            {motionMode === "on"
              ? "调度大屏连线上的光流始终播放。即使系统关闭了动画效果也照常显示，改动立即生效。"
              : motionMode === "system"
                ? "系统要求减少动效时（如开启了「最佳性能」或无障碍选项）自动关闭光流；否则正常播放。"
                : "关闭大屏连线上的流动光效，只保留静态连线。适合投屏或低性能机器。"}
          </p>
        </div>
      </section>

      {/* ================= ③ 分配与回退 ================= */}
      <section className="panel" id="sec-binding" hidden={activeSection !== "binding"}>
        <h2>Agent 配置</h2>
        <p className="hint">
          主模型是这个智能体默认使用的供应商。勾选的「回退」会在主模型失败时依次尝试，
          只对可恢复的失败生效（Key 失效、额度用尽、超时、服务端错误）；
          内容审核拒绝不会重试，因为换个模型结果一样。
        </p>

        <div className="field">
          <label>默认供应商（未指定时使用）</label>
          <select
            className="input"
            value={defaultProvider}
            onChange={(e) => setDefaultProvider(e.target.value)}
          >
            <option value="">（不设置）</option>
            {providers.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label || p.id}
              </option>
            ))}
          </select>
        </div>

        <table className="bind-table">
          <thead>
            <tr>
              <th>智能体</th>
              <th>主模型</th>
              <th>失败时回退</th>
            </tr>
          </thead>
          <tbody>
            {agents.map((agent) => {
              const bound = assignments[agent.id] ?? "";
              const resolved = bound || defaultProvider;
              const target = providers.find((p) => p.id === resolved);
              // Capabilities come from the backend; fall back to the vision list
              // for an older payload so the chip never disappears silently.
              const needs =
                agent.needs ??
                (visionAgents.includes(agent.id)
                  ? [{ key: "vision", label: "需视觉模型" }]
                  : []);

              // A declared need with no model on the resolved provider is a real
              // gap: generation would fail at call time, so mark it here.
              const gapOf = (key: string): boolean => {
                if (!target) return false;
                if (key === "vision") return !target.vision_model;
                if (key === "image") return !target.image_model;
                if (key === "video") return !target.video_model;
                return false;
              };
              return (
                <tr key={agent.id}>
                  <td>
                    <span className="agent-cell">{agent.label}</span>
                    {needs.map((need) => (
                      <span
                        key={need.key}
                        className={`chip ${gapOf(need.key) ? "fail" : "warn"}`}
                        style={{ marginLeft: 6 }}
                        title={
                          gapOf(need.key)
                            ? `${need.label}：当前供应商未配置该能力`
                            : need.label
                        }
                      >
                        {need.label}
                        {gapOf(need.key) ? " ⚠" : ""}
                      </span>
                    ))}
                  </td>
                  <td>
                    <select
                      className="input"
                      value={bound}
                      onChange={(e) =>
                        setAssignments((prev) => {
                          const next = { ...prev };
                          if (e.target.value) next[agent.id] = e.target.value;
                          else delete next[agent.id];
                          return next;
                        })
                      }
                    >
                      <option value="">
                        继承默认
                        {defaultProvider
                          ? `（${providers.find((p) => p.id === defaultProvider)?.label ?? ""}）`
                          : ""}
                      </option>
                      {/* One group per capability the agent needs. A model is
                          identified by "provider · name", so picking it selects
                          both the endpoint and the model in one action — the
                          previous list showed only vendors, leaving the model
                          implicit and often absent. */}
                      {/* Grouped by supplier, one heading each. This matches
                          how a user thinks: pick the account, then pick what that
                          account offers. Values stay `providerId:model` so the
                          resolver is unchanged; the capability is shown as a
                          suffix when a supplier offers more than one kind. */}
                      {providerGroups.map((g) => {
                        // Only the capabilities this agent needs, so the list does
                        // not offer a text model to an agent that only draws.
                        const wanted = agent.needs?.length
                          ? agent.needs.map((n) => n.key)
                          : ["text"];
                        const models = g.models.filter((m) =>
                          wanted.includes(m.capability),
                        );
                        if (!models.length) return null;
                        return (
                          <optgroup key={g.provider_id} label={g.provider_label}>
                            {models.map((m) => (
                              <option
                                key={`${g.provider_id}:${m.model}`}
                                value={`${g.provider_id}:${m.model}`}
                              >
                                {m.model}
                                {models.length > 1 ? `（${m.label}）` : ""}
                              </option>
                            ))}
                          </optgroup>
                        );
                      })}
                    </select>
                    {/* Per-capability gap note. `gapOf` covers vision, image and
                        video, so one row explains every reason the bound provider
                        cannot serve this agent — replacing the vision-only check
                        that referenced a variable this patch removed. */}
                    {needs.some((n) => gapOf(n.key)) && (
                      <div className="muted" style={{ color: "var(--warn)", marginTop: 3 }}>
                        {needs
                          .filter((n) => gapOf(n.key))
                          .map((n) =>
                            n.key === "image"
                              ? "该供应商没填生图模型"
                              : n.key === "video"
                                ? "该供应商没填生视频模型"
                                : "该供应商没填视觉模型",
                          )
                          .join("；")}
                      </div>
                    )}
                  </td>
                  <td>
                    {providers.filter((p) => p.id !== resolved).length === 0 ? (
                      <span className="muted">无其他供应商</span>
                    ) : (
                      <div className="fallback-checks">
                        {providers
                          .filter((p) => p.id !== resolved)
                          .map((p) => (
                            <label key={p.id} className="fallback-check">
                              <input
                                type="checkbox"
                                checked={(fallbacks[agent.id] ?? []).includes(p.id)}
                                onChange={() => toggleFallback(agent.id, p.id)}
                              />
                              {p.label || p.id}
                            </label>
                          ))}
                      </div>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

        </fieldset>
      </div>
      <footer className="settings-footer">
        <div
          className={`settings-save-status ${modelDirty && !failure && !saving ? "dirty" : ""}`}
          role="status"
          aria-live="polite"
        >
          {failure ? <span className="save-fail">保存失败：{failure}</span>
            : saving ? "正在保存…"
            : modelDirty ? "有未保存的修改"
            : message?.kind === "ok" ? message.text : "设置已加载"}
        </div>
        {!baseline
          ? <button className="btn ghost" onClick={() => void load()}>重新加载</button>
          : <button className="btn primary" onClick={saveProviders} disabled={saving || !modelDirty}>
              {saving ? "保存中…" : "保存设置"}
            </button>}
      </footer>
      </div>
      <div className="settings-view" id="settings-view-knowledge" hidden={activeSection !== "knowledge"}>
        {visited.includes("knowledge") && <KnowledgePanel />}
      </div>
      <div className="settings-view" id="settings-view-mcp" hidden={activeSection !== "mcp"}>
        {visited.includes("mcp") && (
          <MCPSettings active={activeSection === "mcp"} />
        )}
      </div>
      {/* Mounted lazily like the two above: the panel polls on mount, and loading
          it for someone who opened 设置 to change a key would be a wasted request. */}
      <div className="settings-view" id="settings-view-logs" hidden={activeSection !== "logs"}>
        {visited.includes("logs") && <LogPanel />}
      </div>
      </div>
    </div>
  );
}
