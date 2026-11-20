/**
 * RAG 面板。
 *
 * 排版按「一条知识长什么样」来，而不是按数据库字段：一、知识正文；二、关键
 * 知识点；三、来源与参考资料。这三段是这类内容被使用的顺序——正文写脚本，要点
 * 当分镜锚点，来源用以核对。所以详情页不做表单，只读。
 *
 * 两处刻意的克制：
 *
 * 1. **没有编辑**。条目要么来自检索（有来源可查），要么来自人工录入（人已确认）。
 *    允许改正文而不留痕，这条知识就失去了可追溯性，来源那一栏也就成了摆设。
 * 2. **待审核单独一栏**。查到的内容当场就能被这次运行使用，但它还没经过人判断。
 *    把「能用」和「已入库」画成同一个绿点，等于让机器替人做判断。
 */

import { useCallback, useEffect, useMemo, useState } from "react";

export interface KnowledgeSource {
  kind?: string;
  org?: string;
  title?: string;
  url?: string;
}

export interface KnowledgeEntry {
  id: string;
  title: string;
  category: string;
  period: string;
  subtitle: string;
  body: string;
  key_points: string[];
  sources: KnowledgeSource[];
  status: "pending" | "approved" | "rejected";
  status_label: string;
  /**
   * `card` 是 AI 从权威来源摘要出的一条主题卡片，`book` 是用户整本上传的资料。
   * 两者共用一条检索路径，但界面上的处理不同：卡片要看正文，书要看章节。
   */
  kind?: "card" | "book";
  kind_label?: string;
  /**
   * 整本资料的章节明细，只对 `book` 返回。
   *
   * `key_points` 只有章节标题，而这类书的章节标题常常就是「1921 年」——光看标题
   * 看不出收了哪几件事。`events` 是章内那些事件块的小标题。
   */
  chapters?: Array<{
    title: string;
    events: string[];
    event_total: number;
    blocks: number;
  }>;
  origin: string;
  query: string;
  note: string;
  uses: number;
  created_at: string;
  updated_at: string;
}

interface Stats {
  approved: number;
  pending: number;
  rejected: number;
  total: number;
  books?: number;
}

/** `POST /api/knowledge/documents` 的返回。`structure` 是入库前给用户看的切分结果。 */
interface UploadResult {
  entry: KnowledgeEntry;
  structure: {
    title: string;
    note: string;
    chapters: number;
    blocks: number;
    chars: number;
    samples: { chapter: string; blocks: number; headings: string[] }[];
  };
  index: {
    ok: boolean;
    chunks?: number;
    embedded?: number;
    removed?: number;
    model?: string;
    reason?: string;
    degraded?: boolean;
  };
  replaced: boolean;
  encoding: string;
  bytes: number;
}

/** `/api/knowledge-index/health` 的返回。`ready` 才是「现在真的能语义检索」。 */
interface VectorHealth {
  configured: boolean;
  /** `local` = 程序内置的 ONNX 模型，`cloud` = 配置了外部嵌入端点。 */
  provider?: string;
  model: string;
  chunks: number;
  /** 只有文本、没有向量的块数：上次嵌入失败留下的，字面能搜、语义搜不到。 */
  text_only_chunks?: number;
  indexed_entries: number;
  dim: number;
  ready: boolean;
  reason: string;
}

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  if (!res.ok) {
    let body = "";
    try {
      body = (await res.text()).slice(0, 300);
    } catch {
      /* 状态码本身就够说明问题 */
    }
    throw new Error(body || `HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

const STATUS_FILTERS = [
  { value: "", label: "全部" },
  { value: "approved", label: "已入库" },
  { value: "pending", label: "待审核" },
  { value: "rejected", label: "已放弃" },
];

/**
 * 首页与栏目列表不是文章。
 *
 * 后端已经不再抓这类页面（`websearch.is_navigation_url`），但库里可能还留着
 * 门槛加上之前生成的条目——那一条《邓小平与中国特色社会主义事业》的来源里就有
 * 「党的文献杂志投稿指南」和「办公人导航」。来源列表是人工核对的唯一入口，所以
 * 在这里标出来，而不是让人点开链接才发现。
 */
function indexLikeSource(url?: string): boolean {
  if (!url) return false;
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return false;
  }
  const path = parsed.pathname.replace(/\/+$/, "");
  if (!path) return true;
  return /(?:index|default)\.(?:html?|shtml)$/i.test(path);
}

/**
 * 正文超过这个长度就不再逐段渲染。
 *
 * 判断「是不是书」本来可以只看 `kind`，但只靠它有一个真实的退化路径：早期入库
 * 的记录没有 `kind` 字段（迁移前写下的行），或者某次接口改动把字段漏了，于是
 * 六万字的正文被 `split(/\n+/).map(<p>)` 一口气插成上千个节点，浏览器卡十几秒。
 *
 * 字数是一个不依赖任何字段的兜底：**没有任何一条知识该有六万字正文**。
 * 超过这个量的东西，要么是整本资料，要么是入库出了错，两种都不该全文渲染。
 */
const FULL_TEXT_LIMIT = 20000;

export default function KnowledgePanel() {
  const [entries, setEntries] = useState<KnowledgeEntry[]>([]);
  const [stats, setStats] = useState<Stats>({ approved: 0, pending: 0, rejected: 0, total: 0 });
  const [status, setStatus] = useState("");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<string>("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  // 资料类型筛选。默认「全部」，因为用户不该为了看到自己刚传的书先去切一个 tab。
  const [kind, setKind] = useState("");
  // 语义检索的实际状态。这一层最危险的失败是静默的：没配模型时照样返回结果
  // （只是走了主题检索），索引过期时召回变空也不报错。所以在面板上常驻显示，
  // 而不是等用户发现「明明库里有却搜不到」再去猜。
  const [vhealth, setVhealth] = useState<VectorHealth | null>(null);

  const load = useCallback(async (statusOverride?: string, queryOverride?: string) => {
    try {
      const activeStatus = statusOverride ?? status;
      const activeQuery = queryOverride ?? query;
      const data = await jsonFetch<{ entries: KnowledgeEntry[]; stats: Stats }>(
        `/api/knowledge?status=${encodeURIComponent(activeStatus)}`
          + `&q=${encodeURIComponent(activeQuery)}`
          + `&kind=${encodeURIComponent(kind)}`,
      );
      setEntries(data.entries);
      setStats(data.stats);
      setError("");
      // 健康状态随列表一起刷新，但它挂了不该拖垮列表——所以单独 catch。
      setVhealth(
        await jsonFetch<VectorHealth>("/api/knowledge-index/health").catch(() => null),
      );
      setSelected((current) => {
        if (current && data.entries.some((e) => e.id === current)) return current;
        return data.entries[0]?.id || "";
      });
    } catch (exc) {
      setError(String(exc));
    }
  }, [status, query, kind]);

  useEffect(() => {
    void load();
  }, [load]);

  const current = useMemo(
    () => entries.find((e) => e.id === selected) || null,
    [entries, selected],
  );

  async function act(id: string, action: "approve" | "reject") {
    setBusy(id);
    try {
      await jsonFetch(`/api/knowledge/${id}/${action}`, { method: "POST" });
      await load();
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy("");
    }
  }

  async function reloadIndex(force: boolean) {
    setBusy("index");
    setError("");
    try {
      const result = await jsonFetch<{ indexed: number; failed: number; skipped: number }>(
        `/api/knowledge-index/reindex?force=${force}`,
        { method: "POST" },
      );
      await load();
      if (result.failed > 0) {
        setError(`索引重建完成，但有 ${result.failed} 条失败；可稍后重试。`);
      }
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy("");
    }
  }

  async function remove(id: string) {
    setBusy(id);
    try {
      await jsonFetch(`/api/knowledge/${id}`, { method: "DELETE" });
      await load();
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy("");
    }
  }

  /**
   * 重新检索一条已经存下的知识。
   *
   * 用条目自己的 query（没有就用标题）再跑一次 `/api/knowledge/research`：
   * 后端会把已有正文带进新一轮整理，抓到更完整的正文时更新**这一条**，
   * 而不是新建。这是"121 字的条目"唯一正确的修法——删了重查会丢掉已经
   * 核对过的来源，而人工改正文又不该发生在这类条目上（没有留痕）。
   */
  async function refresh(entry: KnowledgeEntry) {
    const query = (entry.query || entry.title || "").trim();
    if (!query) {
      setError("这条没有检索词，无法重新检索");
      return;
    }
    setBusy(entry.id);
    setError("");
    try {
      await jsonFetch("/api/knowledge/research", {
        method: "POST",
        body: JSON.stringify({ query, allow_web: true, entry_id: entry.id }),
      });
      await load();
    } catch (exc) {
      setError(`重新检索失败：${String(exc)}`);
    } finally {
      setBusy("");
    }
  }

  // 书籍不走全文渲染，改用章节清单。字数兜底见 FULL_TEXT_LIMIT 的注释。
  const isLongDoc = Boolean(
    current && (current.kind === "book" || (current.body || "").length > FULL_TEXT_LIMIT),
  );

  return (
    <div className="kb">
      <div className="kb-list">
        <div className="kb-tools">
          <input
            className="kb-search"
            value={query}
            placeholder="搜索标题或章节"
            onChange={(e) => setQuery(e.target.value)}
          />
          <button
            className="btn ghost small"
            onClick={() => {
              setShowCreate(false);
              setShowUpload(true);
            }}
            title="上传整本 txt / md 资料"
          >
            传资料
          </button>
          <button
            className="btn primary small"
            onClick={() => {
              setShowUpload(false);
              setShowCreate(true);
            }}
          >
            新建知识
          </button>
        </div>

        <div className="kb-filters">
          {STATUS_FILTERS.map((f) => (
            <button
              key={f.value}
              className={`kb-filter ${status === f.value ? "on" : ""}`}
              onClick={() => setStatus(f.value)}
            >
              {f.label}
              <span className="kb-count">
                {f.value === "" ? stats.total : stats[f.value as keyof Stats] ?? 0}
              </span>
            </button>
          ))}
          {/* 类型筛选只在库里有书时才出现：一开始全是卡片，多一个「全部/卡片/书」
              的三选一只会让第一次使用的人多读三个按钮。 */}
          {(stats.books ?? 0) > 0 && (
            <span className="kb-kind-filter">
              {[
                { value: "", label: "全部类型" },
                { value: "card", label: "卡片" },
                { value: "book", label: "整本资料" },
              ].map((item) => (
                <button
                  key={item.value}
                  className={`kb-filter ${kind === item.value ? "on" : ""}`}
                  onClick={() => setKind(item.value)}
                >
                  {item.label}
                </button>
              ))}
            </span>
          )}
        </div>

        {vhealth && (
          <div className={`kb-index-bar ${vhealth.ready ? "on" : "off"}`}>
            <span className="kb-index-dot" aria-hidden="true" />
            <span className="kb-index-text">
              {vhealth.ready
                ? `语义检索已启用 · ${vhealth.provider === "local" ? "内置模型 " : ""}${vhealth.model} · ${vhealth.indexed_entries} 条 / ${vhealth.chunks} 段`
                : vhealth.reason}
            </span>
            {vhealth.configured && (
              <button
                className="kb-index-btn"
                disabled={Boolean(busy)}
                onClick={() => void reloadIndex(true)}
              >
                {busy === "index" ? "重建中…" : "重建索引"}
              </button>
            )}
          </div>
        )}

        {error && <div className="kb-error">{error}</div>}

        <div className="kb-items">
          {entries.length === 0 && (
            <p className="kb-empty">
              还没有条目。在对话里说「帮我做一个关于×××的视频」时，资料检索会先查本地，
              没有再上网找权威来源，找到的内容会出现在这里等你确认。
            </p>
          )}
          {entries.map((entry) => (
            <button
              key={entry.id}
              className={`kb-item ${entry.id === selected ? "on" : ""}`}
              onClick={() => setSelected(entry.id)}
            >
              <div className="kb-item-head">
                <span className="kb-item-title">
                  {entry.kind === "book" && <span className="kb-kind-tag">整本</span>}
                  {entry.title}
                </span>
                <span className={`kb-badge ${entry.status}`}>{entry.status_label}</span>
              </div>
              <div className="kb-item-sub">
                {entry.subtitle || entry.origin}
                {entry.kind === "book" && ` · ${entry.key_points.length} 章`}
                {entry.uses > 0 && ` · 用过 ${entry.uses} 次`}
              </div>
            </button>
          ))}
        </div>
      </div>

      <div className="kb-detail">
        {showUpload ? (
          <UploadCard
            onClose={() => setShowUpload(false)}
            onImported={async (entry) => {
              setStatus("");
              setQuery("");
              setKind("");
              await load("", "");
              setSelected(entry.id);
            }}
          />
        ) : showCreate ? (
          <ManualEntryCard
            onClose={() => setShowCreate(false)}
            onCreated={async (entry) => {
              setStatus("");
              setQuery("");
              setShowCreate(false);
              await load("", "");
              setSelected(entry.id);
            }}
          />
        ) : current ? (
          <article className="kb-article">
            <header className="kb-article-head">
              <div>
                <h3>{current.title}</h3>
                <p className="kb-article-sub">{current.subtitle || current.origin}</p>
              </div>
              <span className={`kb-badge ${current.status}`}>{current.status_label}</span>
            </header>
            {current.note && <p className="kb-entry-note">{current.note}</p>}

            <section className="kb-section">
              <h4>一、{isLongDoc ? "章节结构" : "知识正文"}</h4>
              {isLongDoc ? (
                /*
                 * 书不能渲染全文。一本六万字，`body.split(/\n+/).map(<p>)` 会一口气
                 * 插进上千个节点，浏览器直接卡住十几秒——而用户点开这本书只是想
                 * 确认「它收了什么」。这里只列章节，正文由检索按片段取。
                 */
                <>
                  <p className="kb-book-meta">
                    整本资料 · {current.key_points.length} 章 · {current.body.length} 字
                  </p>
                  <ul className="kb-points kb-chapters">
                    {(current.chapters?.length
                      ? current.chapters
                      : current.key_points.map((title) => ({
                          title,
                          events: [] as string[],
                          event_total: 0,
                          blocks: 0,
                        }))
                    ).map((chapter, i) => (
                      <li key={i} className="kb-chapter">
                        <span className="kb-chapter-title">{chapter.title}</span>
                        {chapter.events.length > 0 && (
                          <span className="kb-chapter-events">
                            {chapter.events.join(" · ")}
                            {chapter.event_total > chapter.events.length
                              ? ` 等 ${chapter.event_total} 个事件`
                              : ""}
                          </span>
                        )}
                      </li>
                    ))}
                    {current.key_points.length === 0 && <li className="kb-muted">（未识别出章节）</li>}
                  </ul>
                </>
              ) : (
                <>
                  {current.body.length < 240 && (
                    /*
                     * 一条 121 字的"知识正文"在界面上看起来完全正常：有标题、有正文、
                     * 有来源链接。但它撑不起一分钟视频——后面的分镜只能反复拍同一句
                     * 话。后台已经有硬门槛（材料很厚时摘要不足 300 字直接判失败），
                     * 这里再把"薄"标出来，是因为库里可能还留着门槛加上之前存下的条目，
                     * 而人在点「入库」时最需要知道的就是这一条。
                     */
                    <p className="kb-thin-warning">
                      正文只有 {current.body.length} 字，材料本身不应这么短。
                      这样的条目只能支撑十几秒，建议点下方「重新检索」补足正文后再入库。
                    </p>
                  )}
                  {current.body
                    .split(/\n+/)
                    .filter((p) => p.trim())
                    .map((para, i) => (
                      <p key={i}>{para}</p>
                    ))}
                </>
              )}
            </section>

            <section className="kb-section">
              <h4>二、{isLongDoc ? "可检索片段" : "关键知识点"}</h4>
              {isLongDoc ? (
                <p className="kb-book-meta">
                  每条语义单元独立成片，片段不跨章。检索时按语义匹配到具体片段，
                  引用会带上章节标题。
                </p>
              ) : (
                <ul className="kb-points">
                  {current.key_points.map((point, i) => (
                    <li key={i}>{point}</li>
                  ))}
                  {current.key_points.length === 0 && <li className="kb-muted">（未提取）</li>}
                </ul>
              )}
            </section>

            <section className="kb-section">
              <h4>三、来源与参考资料</h4>
              <div className="kb-sources">
                {current.sources.map((source, i) => (
                  <div className="kb-source" key={i}>
                    <span className="kb-source-kind">{source.kind || "参考资料"}</span>
                    <div className="kb-source-body">
                      {indexLikeSource(source.url) && (
                        <div className="kb-source-warn">
                          这是站点首页或栏目列表，不是文章页
                        </div>
                      )}
                      {source.title && <div className="kb-source-title">{source.title}</div>}
                      <div className="kb-source-meta">
                        {[source.org, source.url && "原文链接"].filter(Boolean).join(" · ") || "未记录"}
                      </div>
                      {source.url && (
                        <a className="kb-source-link" href={source.url} target="_blank" rel="noreferrer">
                          {source.url}
                        </a>
                      )}
                    </div>
                  </div>
                ))}
                {current.sources.length === 0 && (
                  <p className="kb-muted">（未记录来源；手工录入的条目建议补上）</p>
                )}
              </div>
            </section>

            <footer className="kb-actions">
              {current.status === "pending" && (
                <>
                  <button
                    className="btn primary small"
                    disabled={busy === current.id}
                    onClick={() => void act(current.id, "approve")}
                  >
                    入库
                  </button>
                  <button
                    className="btn ghost small"
                    disabled={busy === current.id}
                    onClick={() => void act(current.id, "reject")}
                  >
                    放弃
                  </button>
                </>
              )}
              <button
                className="btn ghost small"
                disabled={busy === current.id}
                onClick={() => void refresh(current)}
                title="用这个条目的检索词重新联网，补足正文后再整理一次"
              >
                {busy === current.id ? "检索中…" : "重新检索"}
              </button>
              <button
                className="btn ghost small danger"
                disabled={busy === current.id}
                onClick={() => void remove(current.id)}
              >
                删除
              </button>
              <span className="kb-actions-note">
                {current.status === "pending"
                  ? "待审核：这次运行已经用它了，确认后下次直接命中、不再联网"
                  : current.status === "approved"
                    ? `已入库${current.uses ? `，被用过 ${current.uses} 次` : ""}`
                    : "已放弃：不会再被检索到"}
              </span>
            </footer>
          </article>
        ) : (
          <div className="kb-placeholder">
            <p>选一条知识看详情，或者用「传资料」「新建知识」往库里加内容。</p>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * 上传整本资料。
 *
 * 三处和「新建知识」不同，都是因为书的体量：
 *
 * 1. **传原始字节，不传 JSON 字符串**。中文 txt 有一半是 GBK，浏览器按 UTF-8
 *    读出来是乱码，而乱码入库不报错——只会在之后表现为「这本书什么都搜不到」。
 *    编码只能在后端按字节判断，所以这里用 File 直接做 body。
 * 2. **必须显示进度**。一本六万字的书要调 25 次嵌入接口、跑 30 秒以上。没有
 *    任何反馈的话用户会以为卡死了，然后重复点上传。
 * 3. **入库前先给结构**。章节数、片段数、前几章的目录，用户一眼就能看出
 *    「切得对不对」——切错了（比如整本一章）后面检索再好也没用。
 */
function UploadCard({
  onClose,
  onImported,
}: {
  onClose: () => void;
  onImported: (entry: KnowledgeEntry) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [result, setResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState("");
  const [phase, setPhase] = useState<"idle" | "uploading" | "done">("idle");
  const [dragging, setDragging] = useState(false);

  function pick(next: File | null) {
    setFile(next);
    setError("");
    setResult(null);
    if (next && !title) {
      setTitle(next.name.replace(/\.[^.]+$/, ""));
    }
  }

  async function upload() {
    if (!file) {
      setError("请先选择一个 txt 或 md 文件");
      return;
    }
    setPhase("uploading");
    setError("");
    try {
      const params = new URLSearchParams({ filename: file.name });
      if (title.trim()) params.set("title", title.trim());
      const res = await fetch(`/api/knowledge/documents?${params.toString()}`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: file,
      });
      const text = await res.text();
      if (!res.ok) {
        // FastAPI 的错误是 {"detail": "..."}，但 500 可能是纯文本。
        // 两种都要能显示成一句人话，不能让用户看到一段 JSON。
        let message = text;
        try {
          const parsed = JSON.parse(text);
          message = String(parsed.detail || parsed.message || text);
        } catch {
          /* 不是 JSON，原样显示 */
        }
        throw new Error(message);
      }
      const data = JSON.parse(text) as UploadResult;
      setResult(data);
      setPhase("done");
      onImported(data.entry);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
      setPhase("idle");
    }
  }

  const sizeKb = file ? Math.round(file.size / 1024) : 0;

  return (
    <div className="kb-settings kb-create">
      <h3>上传整本资料</h3>
      <p className="kb-settings-note">
        支持 txt 与 md。入库后会先还原章节结构，再把每一节切成可召回的片段——
        片段不跨章，引用能说到「第几章第几个事件」。上传即入库，不需要审核。
      </p>

      <label
        className={`kb-drop ${dragging ? "on" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          pick(e.dataTransfer.files?.[0] || null);
        }}
      >
        <input
          type="file"
          accept=".txt,.md,.markdown,.text"
          onChange={(e) => pick(e.target.files?.[0] || null)}
        />
        {file ? (
          <span className="kb-drop-file">
            {file.name} · {sizeKb} KB
          </span>
        ) : (
          <span className="kb-drop-hint">点击选择，或把文件拖到这里</span>
        )}
      </label>

      <label className="kb-field">
        <span>书名（留空则用文件名）</span>
        <input value={title} maxLength={120} onChange={(e) => setTitle(e.target.value)} />
      </label>

      {error && <div className="kb-error">{error}</div>}

      {phase === "uploading" && (
        <div className="kb-upload-progress">
          <span className="kb-upload-spinner" aria-hidden="true" />
          正在解析并建立索引，一本书通常需要几十秒。请不要关闭面板。
        </div>
      )}

      {result && (
        <div className="kb-upload-result">
          <div className="kb-upload-title">
            {result.replaced ? "已更新同名资料" : "已入库"}：{result.structure.title}
          </div>
          <div className="kb-upload-stats">
            <span><b>{result.structure.chapters}</b> 章</span>
            <span><b>{result.structure.blocks}</b> 个语义单元</span>
            <span><b>{result.structure.chars}</b> 字</span>
            <span><b>{result.index.chunks ?? 0}</b> 个检索片段</span>
            <span>编码 {result.encoding}</span>
          </div>
          {result.index.ok === false && (
            <div className="kb-error">
              资料已入库，但语义索引没建成：{result.index.reason || "未知原因"}。
              主题检索仍可用，语义检索要等索引重建。
            </div>
          )}
          {result.structure.samples.length > 0 && (
            <div className="kb-upload-samples">
              <div className="kb-upload-subhead">切分预览</div>
              {result.structure.samples.map((sample) => (
                <div className="kb-upload-sample" key={sample.chapter}>
                  <span className="kb-upload-chapter">{sample.chapter}</span>
                  <span className="kb-upload-headings">
                    {sample.headings.length > 0
                      ? sample.headings.join("、")
                      : `${sample.blocks} 段无小标题正文`}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="kb-settings-actions">
        <button
          className="btn primary small"
          disabled={phase === "uploading" || !file}
          onClick={() => void upload()}
        >
          {phase === "uploading" ? "处理中…" : result ? "再传一本" : "开始导入"}
        </button>
        <button className="btn ghost small" disabled={phase === "uploading"} onClick={onClose}>
          关闭
        </button>
      </div>
    </div>
  );
}


function ManualEntryCard({
  onClose,
  onCreated,
}: {
  onClose: () => void;
  onCreated: (entry: KnowledgeEntry) => Promise<void>;
}) {
  const [title, setTitle] = useState("");
  const [category, setCategory] = useState("历史知识");
  const [period, setPeriod] = useState("");
  const [body, setBody] = useState("");
  const [points, setPoints] = useState("");
  const [sourceOrg, setSourceOrg] = useState("");
  const [sourceTitle, setSourceTitle] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);

  async function save() {
    if (!title.trim()) {
      setError("请填写标题");
      return;
    }
    if (body.trim().length < 20) {
      setError("知识正文至少需要 20 个字");
      return;
    }
    setSaving(true);
    setError("");
    try {
      const sources = sourceOrg.trim() || sourceTitle.trim() || sourceUrl.trim()
        ? [{ kind: "参考资料", org: sourceOrg.trim(), title: sourceTitle.trim(), url: sourceUrl.trim() }]
        : [];
      const data = await jsonFetch<{ entry: KnowledgeEntry }>("/api/knowledge", {
        method: "POST",
        body: JSON.stringify({
          title: title.trim(),
          category: category.trim(),
          period: period.trim(),
          body: body.trim(),
          key_points: points.split(/\r?\n/).map((p) => p.trim()).filter(Boolean),
          sources,
        }),
      });
      await onCreated(data.entry);
    } catch (exc) {
      setError(String(exc));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="kb-settings kb-create">
      <h3>新建知识</h3>
      <p className="kb-settings-note">
        人工录入的内容会直接标记为“已入库”，但自动生成只会使用带白名单权威网址的条目；
        没有权威链接的内容只作为私人笔记保留。
      </p>

      <label className="kb-field">
        <span>标题 *</span>
        <input value={title} maxLength={80} onChange={(e) => setTitle(e.target.value)} />
      </label>
      <div className="kb-form-row">
        <label className="kb-field">
          <span>分类</span>
          <input value={category} maxLength={40} onChange={(e) => setCategory(e.target.value)} />
        </label>
        <label className="kb-field">
          <span>时间范围</span>
          <input value={period} maxLength={40} placeholder="如：1904—1997" onChange={(e) => setPeriod(e.target.value)} />
        </label>
      </div>
      <label className="kb-field">
        <span>知识正文 *（至少 20 字）</span>
        <textarea value={body} rows={10} maxLength={30000} onChange={(e) => setBody(e.target.value)} />
      </label>
      <label className="kb-field">
        <span>关键知识点（每行一条）</span>
        <textarea value={points} rows={5} onChange={(e) => setPoints(e.target.value)} />
      </label>

      <h4 className="kb-create-subhead">来源（自动生成时必须填写权威链接）</h4>
      <div className="kb-form-row">
        <label className="kb-field">
          <span>来源机构</span>
          <input value={sourceOrg} maxLength={80} onChange={(e) => setSourceOrg(e.target.value)} />
        </label>
        <label className="kb-field">
          <span>资料标题</span>
          <input value={sourceTitle} maxLength={160} onChange={(e) => setSourceTitle(e.target.value)} />
        </label>
      </div>
      <label className="kb-field">
        <span>原文链接</span>
        <input value={sourceUrl} maxLength={500} onChange={(e) => setSourceUrl(e.target.value)} />
      </label>

      {error && <div className="kb-error">{error}</div>}
      <div className="kb-settings-actions">
        <button className="btn primary small" disabled={saving} onClick={() => void save()}>
          {saving ? "保存中…" : "保存并入库"}
        </button>
        <button className="btn ghost small" disabled={saving} onClick={onClose}>取消</button>
      </div>
    </div>
  );
}
