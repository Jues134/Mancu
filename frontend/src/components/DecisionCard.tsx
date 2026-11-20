/**
 * 导演决策卡：输入判定的可见过程。
 *
 * 以前这一步只有一行「输入识别：主题要求，先查本地知识库」——读者要自己从五个
 * 固定标签里猜它读了什么、凭什么这么判。卡把它拆成五段：
 *
 *   收到   → 收到的是什么形态，我要先做什么（只分诊，不下结论）
 *   原文   → 用户到底说了什么（判错的唯一参照）
 *   理解   → 读出了什么，每项都带原文引用
 *   判定   → 判成什么，依据分三行：程序测量 / 规则判断 / 导演自述
 *   下一步 → 接下来干什么
 *
 * 引用由后端保证是原文子串；这里只负责显示，不补造任何内容。没有置信度、没有
 * 百分比——没有数据支撑的数字会被读成事实。
 */

interface Row {
  label: string;
  value: string;
  quote?: string;
}

interface Basis {
  text: string;
  from: string;
}

export interface DecisionCardData {
  received?: { line?: string; triage?: string; chars?: number };
  source?: { text?: string; chars?: number };
  understanding?: Row[];
  verdict?: { title?: string; need_kb?: boolean; basis?: Basis[] };
  next?: { line?: string; hint?: string };
}

/** 「程序测量 / 规则判断 / 导演自述」三种依据的可信度不同，分开标出来。 */
const FROM_HINT: Record<string, string> = {
  程序测量: "不依赖模型，可以对账",
  规则判断: "可以反驳",
  导演自述: "模型自己的说法",
};

export function DecisionCard({ card }: { card: DecisionCardData }) {
  const received = card.received || {};
  const understanding = card.understanding || [];
  const verdict = card.verdict || {};
  const basis = verdict.basis || [];
  const next = card.next || {};
  const text = card.source?.text || "";

  return (
    <div className="turn agent decision-turn">
      <div className="turn-head">
        <span className="turn-avatar">{`导`}</span>
        <span className="turn-name">导演</span>
        <span className="turn-tag">输入判定</span>
        <span className="chip ok">已判定</span>
      </div>

      <div className="turn-body">
        <div className="decision-card">
          <section className="dc-received">
            <div className="dc-label">收到</div>
            <div>{received.line}</div>
            {received.triage && <div className="dc-dim">{received.triage}</div>}
          </section>

          <section>
            <div className="dc-label">原文</div>
            <blockquote className="dc-source">{text}</blockquote>
          </section>

          <section>
            <div className="dc-label">导演的理解</div>
            <div className="dc-grid">
              {understanding.map((row) => (
                <div className="dc-pair" key={row.label}>
                  <span className="dc-key">{row.label}</span>
                  <span>
                    {row.value}
                    {row.quote ? <span className="dc-quote">「{row.quote}」</span> : null}
                  </span>
                </div>
              ))}
            </div>
          </section>

          <section>
            <div className="dc-label">判定</div>
            <div className="dc-verdict">{verdict.title}</div>
            <ul className="dc-basis">
              {basis.map((item, index) => (
                <li key={index}>
                  <span>{item.text}</span>
                  <span className="dc-from" title={FROM_HINT[item.from] || ""}>
                    {item.from}
                  </span>
                </li>
              ))}
            </ul>
          </section>

          <section>
            <div className="dc-label">下一步</div>
            <div>{next.line}</div>
            {next.hint && <div className="dc-hint">{next.hint}</div>}
          </section>
        </div>
      </div>
    </div>
  );
}
