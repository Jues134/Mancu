/**
 * 设置 → 日志。
 *
 * 这个面板的目的不是「展示日志」，是让人在出问题的那一刻不必开命令行。
 * 因此顺序是刻意排的：
 *
 *   1. 先看**错误**——90% 的问题在这一屏就解释完了；
 *   2. 再看**事件**——一次模型调用花了多久、重试了几次、状态码是几；
 *   3. 最后才是整个运行记录，那是要顺着时间线读的。
 *
 * 诊断包按钮放在右上角而不是藏进菜单：会点它的人一般已经不想再解释了，
 * 想的是「把这个发过去」。
 */

import { useCallback, useEffect, useState } from "react";
import { jsonFetch } from "../api/client";

interface LogEvent {
  ts: string;
  level: "info" | "warn" | "error" | string;
  event: string;
  req: string;
  run: string;
  agent: string;
  /** 各事件自己的字段：status / ms / model / error / reason … */
  [key: string]: unknown;
}

interface LogSummary {
  dir: string;
  level: string;
  total_kb: number;
  keep_days: number;
  files: Array<{ name: string; kb: number; modified: string }>;
}

interface LogsResponse {
  events: LogEvent[];
  summary: LogSummary;
  errors: string;
  app: string;
}

/** 这些键在每一行里都会显示，不再重复塞进「详情」。 */
const FIXED_KEYS = new Set(["ts", "level", "event", "req", "run", "agent"]);

const LEVEL_LABEL: Record<string, string> = {
  info: "信息",
  warn: "警告",
  error: "错误",
};

function timeOf(iso: string): string {
  // 当天的只显示时分秒：一屏里二十行都带着同一个日期，那是噪声。
  const text = String(iso || "");
  const today = new Date().toISOString().slice(0, 10);
  if (text.startsWith(today)) return text.slice(11, 19);
  return text.replace("T", " ").slice(0, 19);
}

function detail(event: LogEvent): string {
  return Object.entries(event)
    .filter(([key, value]) => !FIXED_KEYS.has(key) && value !== null && value !== "" && value !== undefined)
    .map(([key, value]) => {
      const text = typeof value === "object" ? JSON.stringify(value) : String(value);
      return `${key}=${text.length > 120 ? `${text.slice(0, 120)}…` : text}`;
    })
    .join("  ");
}

type Tab = "events" | "errors" | "app";

const TABS: Array<{ id: Tab; label: string }> = [
  { id: "events", label: "事件" },
  { id: "errors", label: "错误日志" },
  { id: "app", label: "运行记录" },
];

export default function LogPanel() {
  const [data, setData] = useState<LogsResponse | null>(null);
  const [tab, setTab] = useState<Tab>("events");
  const [problemsOnly, setProblemsOnly] = useState(false);
  const [limit, setLimit] = useState(200);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams({ limit: String(limit) });
      // 服务端按级别过滤，而不是在前端筛：库里可能有几千条，只取需要的那些。
      if (problemsOnly) params.set("level", "error");
      setData(await jsonFetch<LogsResponse>(`/api/diagnostics/logs?${params}`, { timeoutMs: 15000 }));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [limit, problemsOnly]);

  useEffect(() => {
    void load();
  }, [load]);

  const copyPath = async () => {
    try {
      await navigator.clipboard.writeText(data?.summary.dir || "");
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setError("复制失败，请手动选中路径");
    }
  };

  const events = (data?.events || []).filter((e) => !problemsOnly || e.level === "error" || e.level === "warn");

  return (
    <section className="panel" id="sec-logs" style={{ margin: 0 }}>
      <h2>日志</h2>
      <p className="hint">
        这里记录每次运行、每次模型调用和每次报错。平时不用管；出问题时先看「错误日志」，
        它带着完整的堆栈，能指出崩在哪一行。日志只保存在本机，不联网、不上传。
      </p>

      {error && <div className="banner fail">{error}</div>}

      <div className="log-meta">
        <div>
          <span className="muted">位置</span>
          <code className="log-path">{data?.summary.dir || "…"}</code>
          <button type="button" className="btn ghost log-copy" onClick={() => void copyPath()}>
            {copied ? "已复制" : "复制路径"}
          </button>
        </div>
        <div className="muted">
          {data ? `${data.summary.total_kb} KB · 保留 ${data.summary.keep_days} 天 · 级别 ${data.summary.level}` : ""}
        </div>
      </div>

      <div className="log-toolbar">
        <div className="log-tabs" role="tablist" aria-label="日志分类">
          {TABS.map((item) => (
            <button
              key={item.id}
              type="button"
              role="tab"
              aria-selected={tab === item.id}
              className={`log-tab ${tab === item.id ? "active" : ""}`}
              onClick={() => setTab(item.id)}
            >
              {item.label}
            </button>
          ))}
        </div>
        <div className="log-actions">
          <label className="log-check">
            <input
              type="checkbox"
              checked={problemsOnly}
              onChange={(e) => setProblemsOnly(e.target.checked)}
            />
            只看问题
          </label>
          <select
            className="input log-limit"
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
            aria-label="显示条数"
          >
            <option value={100}>最近 100 条</option>
            <option value={200}>最近 200 条</option>
            <option value={500}>最近 500 条</option>
          </select>
          <button type="button" className="btn ghost" onClick={() => void load()} disabled={loading}>
            {loading ? "读取中…" : "刷新"}
          </button>
          <a
            className="btn primary"
            href="/api/diagnostics/bundle"
            download
            title="打包日志、环境信息和供应商配置（Key 已截断），便于发给别人排查"
          >
            导出诊断包
          </a>
        </div>
      </div>

      {tab === "events" && (
        <div className="log-list">
          {events.length === 0 ? (
            <p className="muted log-empty">
              {problemsOnly ? "没有警告或错误 —— 这是好消息。" : "还没有记录。跑一次任务后再回来看。"}
            </p>
          ) : (
            events.map((event, index) => (
              <div key={`${event.ts}-${index}`} className={`log-row level-${event.level}`}>
                <span className="log-time">{timeOf(event.ts)}</span>
                <span className={`log-level ${event.level}`}>{LEVEL_LABEL[event.level] || event.level}</span>
                <span className="log-event">{event.event}</span>
                {event.run && event.run !== "-" && <span className="log-run">{event.run.slice(0, 8)}</span>}
                <span className="log-detail">{detail(event)}</span>
              </div>
            ))
          )}
        </div>
      )}

      {tab === "errors" && (
        <pre className="log-text">{data?.errors?.trim() || "还没有错误记录。"}</pre>
      )}

      {tab === "app" && (
        <pre className="log-text">{data?.app?.trim() || "还没有运行记录。"}</pre>
      )}
    </section>
  );
}
