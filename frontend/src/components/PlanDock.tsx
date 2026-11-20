/**
 * 右上角常驻的任务板。
 *
 * 为什么不能只放在对话里：清单是用户唯一能实时看出「跑到第几步、卡在哪一段」的
 * 东西。它在消息流里会随着新消息一路往上滚，用户想去核对一眼就得往回翻——而
 * 翻的时候恰好看不到最新的那一条。所以它固定在右上角，点开就看见，再点收回。
 *
 * 三条刻意的选择：
 *
 * 1. **正文和对话里那一张是同一份实现**（`PlanBody`）。两处各写一遍的结果是
 *    「板上显示 5 段、对话里 6 段」——而它们本来就该是同一件事的两种看法。
 * 2. **默认展开，但只对「这一轮」展开。** 一有清单就自己出现（用户要的是实时
 *    看到，不是自己去点），用户手动收起之后就不再自己弹回来；换一轮任务重新
 *    按默认来，否则上一轮的收起动作会静默地关掉下一轮的板。
 * 3. **收起状态也留一个把手**（顶栏那个按钮），带进度。完全隐藏掉的常驻面板，
 *    用户下次想找它时会先怀疑它是不是坏了。
 */

import { PlanBody, type PlanCardData } from "./PlanCard";

export function PlanDock({
  card,
  onClose,
}: {
  card: PlanCardData;
  onClose: () => void;
}) {
  const state = card.state || "running";
  const done = card.done || 0;
  const total = card.total || (card.steps || []).length;

  return (
    <aside className="plan-dock" aria-label="任务板">
      <div className="plan-dock-head">
        <span className="plan-dock-title">任务板</span>
        <span
          className={`chip ${state === "done" ? "ok" : state === "failed" ? "fail" : "warn"}`}
        >
          {state === "done" ? "已完成" : state === "failed" ? "未完成" : "执行中"} {done}/{total}
        </span>
        <button
          className="plan-dock-close"
          onClick={onClose}
          title="收起任务板"
          aria-label="收起任务板"
        >
          ✕
        </button>
      </div>
      <div className="plan-dock-body">
        <PlanBody card={card} />
      </div>
    </aside>
  );
}

/** 顶栏上的那个把手。收起时它是唯一的入口，所以必须带进度。 */
export function PlanDockButton({
  card,
  open,
  onToggle,
}: {
  card: PlanCardData;
  open: boolean;
  onToggle: () => void;
}) {
  const state = card.state || "running";
  const failed = state === "failed" || (card.notes || []).length > 0;
  return (
    <button
      className={`btn ghost small plan-dock-btn ${open ? "active" : ""}`}
      onClick={onToggle}
      title={open ? "收起任务板" : "展开任务板"}
      aria-expanded={open}
      aria-controls="plan-dock"
    >
      <span className={`plan-dock-dot ${state === "done" ? "ok" : failed ? "fail" : "live"}`} />
      任务板
      <span className="plan-dock-count">
        {card.done || 0}/{card.total || (card.steps || []).length}
      </span>
    </button>
  );
}
