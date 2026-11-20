/**
 * 「停在等你选」的卡：本地没查到材料时，把「接下来从哪儿找」交回用户。
 *
 * 这不是错误提示。任务没有失败，它挂在这里等一个回答；回答完从断点继续，前面
 * 已经跑过的判定和检索不会重来。
 *
 * 三条规则：
 *
 * 1. **只列这次真能走的路。** 选项由后端按能力生成——没连飞书就不出现「去飞书查」。
 *    显示了但点下去报错，比不显示更糟：用户会以为是 bug。
 * 2. **选中「我给你正文」才出输入框。** 输入框一直摆着，会让人以为非得贴点什么
 *    才能继续。
 * 3. **回答过就不再问第二遍。** 卡片带着 `answer` 回来时，只显示当时选了什么，
 *    按钮全部消失——历史里一张还在闪烁的按钮会让人以为任务还卡在这里。
 */

import { useState } from "react";

interface Option {
  id: string;
  label: string;
  hint?: string;
  needs_text?: boolean;
}

interface Tried {
  source: string;
  ok?: boolean;
}

export interface AskQuestion {
  id?: string;
  title?: string;
  reason?: string;
  topic?: string;
  tried?: Tried[];
  options?: Option[];
  answer?: { choice?: string; text?: string } | null;
}

interface Props {
  card: AskQuestion;
  runId?: string;
  /** 任务还在等回答吗。不是的话按钮不出——点了只会拿到 409。 */
  canAnswer?: boolean;
  onAnswered?: () => void;
}

export function AskCard({ card, runId, canAnswer = true, onAnswered }: Props) {
  const options = card.options || [];
  const answered = card.answer || null;
  const [choice, setChoice] = useState<string>("");
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const picked = options.find((o) => o.id === choice);
  const answeredLabel = answered
    ? options.find((o) => o.id === answered.choice)?.label || answered.choice
    : "";

  async function submit() {
    if (!runId || !choice) return;
    setBusy(true);
    setError("");
    try {
      const res = await fetch(`/api/doc/runs/${runId}/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question_id: card.id || "",
          choice,
          text: picked?.needs_text ? text : "",
        }),
      });
      if (!res.ok) {
        let detail = "";
        try {
          detail = (await res.text()).slice(0, 200);
        } catch {
          /* body is optional */
        }
        throw new Error(`HTTP ${res.status}${detail ? ` — ${detail}` : ""}`);
      }
      onAnswered?.();
    } catch (err) {
      setError(String(err).slice(0, 200));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="turn agent ask-turn">
      <div className="turn-head">
        <span className="turn-avatar">导</span>
        <span className="turn-name">导演</span>
        <span className="turn-tag">需要你确认</span>
        {answered ? (
          <span className="chip ok">已选：{answeredLabel}</span>
        ) : canAnswer ? (
          <span className="chip warn">等你回复</span>
        ) : (
          <span className="chip">未回答</span>
        )}
      </div>

      <div className="turn-body">
        <div className="ask-card">
          <div className="ask-title">{card.title}</div>
          {card.reason && <div className="ask-reason">{card.reason}</div>}

          {(card.tried || []).length > 0 && (
            <div className="ask-tried">
              {card.tried!.map((item, i) => (
                <span key={i} className={`ask-tried-item ${item.ok ? "ok" : ""}`}>
                  {item.source}
                  {item.ok ? " 命中" : " 没找到"}
                </span>
              ))}
            </div>
          )}

          {answered ? (
            <div className="ask-answered">
              {answered.text ? (
                <>
                  <div className="ask-key">你给的正文（{answered.text.length} 字）</div>
                  <blockquote className="dc-source">{answered.text}</blockquote>
                </>
              ) : (
                <div className="ask-key">已按你的选择继续。</div>
              )}
            </div>
          ) : (
            <>
              <ul className="ask-options">
                {options.map((option) => (
                  <li key={option.id}>
                    <button
                      className={`ask-option ${choice === option.id ? "picked" : ""}`}
                      onClick={() => setChoice(option.id)}
                      disabled={!canAnswer || busy}
                    >
                      <span className="ask-radio" />
                      <span className="ask-option-body">
                        <span className="ask-option-label">{option.label}</span>
                        {option.hint && <span className="ask-option-hint">{option.hint}</span>}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>

              {picked?.needs_text && (
                <textarea
                  className="ask-input"
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="把正文贴在这里。只用于这次任务，不写入知识库。"
                  rows={6}
                  disabled={busy}
                />
              )}

              {error && <div className="ask-error">{error}</div>}

              <div className="ask-actions">
                <button
                  className="btn primary small"
                  onClick={submit}
                  disabled={!canAnswer || !choice || busy || (picked?.needs_text && !text.trim())}
                >
                  {busy ? "提交中…" : "确定"}
                </button>
                {!canAnswer && (
                  <span className="ask-hint">这条任务已经结束，不能再回答</span>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
