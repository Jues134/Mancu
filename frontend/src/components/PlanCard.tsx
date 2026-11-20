/**
 * 导演任务清单：开工前贴出来，跑到哪一步改到哪一步。
 *
 * 和决策卡的分工——决策卡说「这段话被读成了什么」（判定），清单说「接下来分几步、
 * 每步谁做、产出什么」（规划）。清单是同一张卡原地推进的：后端用同一个 msg_key
 * 覆写它，所以这里不需要任何合并逻辑，每次渲染都是最新状态。
 *
 * 三个刻意的选择：
 *
 * 1. **每段折叠成一行。** 三段 × 四步全铺开就是 12 行，清单会长到没人读完。
 *    默认只显示一行（段名 + 四个圆点 + 当前卡在哪），点开才看每步的名字和状态。
 * 2. **每步的产出直接摊在行下，不藏在折叠里。** 折叠起来的东西等于没写；这一步
 *    做成了什么，是读者唯一能核对「它说的和做的一样吗」的地方。
 * 3. **没有「合成」这一步。** 合成确实在后台发生，所以它写在脚下的边界说明里，
 *    而不是装作不存在，也不是假装它是计划的最后一步。
 */

import { useState } from "react";

type StepStatus = "pending" | "running" | "done" | "failed" | "not_needed";

interface StageDef {
  id: string;
  label: string;
}

interface SegmentRow {
  id: string;
  title: string;
  status: string;
  stages?: Record<string, string>;
  note?: string;
}

interface PlanStep {
  id: string;
  index: number;
  title: string;
  owner: string;
  output: string;
  depends?: string;
  status: StepStatus;
  summary?: string;
  result?: string;
  stages?: StageDef[];
  segments?: SegmentRow[];
  seg_done?: number;
  seg_total?: number;
}

export interface PlanCardData {
  state?: string;
  done?: number;
  total?: number;
  rule?: string;
  beyond?: string;
  notes?: string[];
  steps?: PlanStep[];
}

/** 状态 → 圆点样式。未知值按「待开始」画，不另造颜色。 */
const DOT: Record<string, string> = {
  pending: "dot q",
  running: "dot live",
  done: "dot done",
  failed: "dot fail",
  not_needed: "dot skip",
  rework: "dot rework",
};

const STEP_LABEL: Record<string, string> = {
  pending: "待开始",
  running: "执行中",
  done: "完成",
  failed: "未完成",
  not_needed: "不需要",
};

const SEG_LABEL: Record<string, string> = {
  pending: "待开始",
  running: "进行中",
  done: "完成",
  failed: "未完成",
  rework: "返工中",
};

const STEP_GLYPH: Record<number, string> = { 1: "①", 2: "②", 3: "③", 4: "④" };

/**
 * 对话里的一行占位。
 *
 * 清单本体已经从消息流搬到右上角那块常驻的板上（`PlanDock`）。如果这里也画一整
 * 张卡，屏幕上会同时出现两份同样的东西，一份还被另一份盖住一半——那不像「固定在
 * 右上角」，像界面出了重复渲染。
 *
 * 但整条消息也不能删掉：清单确实在对话里发生过，读者往回翻时应该看到「这里有过
 * 一次计划」，而不是什么都没有。所以留一行：谁说的、几步、现在到哪儿。
 */
export function PlanStub({ card }: { card: PlanCardData }) {
  const steps = card.steps || [];
  const state = card.state || "running";
  const total = card.total || steps.length;

  return (
    <div className="turn agent plan-turn plan-stub">
      <div className="turn-head">
        <span className="turn-avatar">导</span>
        <span className="turn-name">导演</span>
        <span className="turn-tag">任务清单</span>
        <span className="plan-stub-note">已固定到右上角，随时能看</span>
        <span
          className={`chip ${state === "done" ? "ok" : state === "failed" ? "fail" : "warn"}`}
        >
          {state === "done" ? "已完成" : state === "failed" ? "未完成" : "执行中"}{" "}
          {card.done || 0}/{total}
        </span>
      </div>
    </div>
  );
}

/**
 * 清单本体，不带「谁说的」那一行。
 *
 * 它只有右上角一个位置。头部自带标题，再套一层 turn-head 会出现两个
 * 「导演 · 任务清单」。
 */
export function PlanBody({ card }: { card: PlanCardData }) {
  const steps = card.steps || [];
  // 只有分段需要展开：步骤的产出是常显的，再做一次折叠等于多一次点击。
  const [open, setOpen] = useState<Record<string, boolean>>({});

  return (
    <div className="plan-card">
      <ol className="plan-steps">
        {steps.map((step) => (
          <li key={step.id} className={`plan-step ${step.status}`}>
            <div className="plan-step-head">
              <span className="plan-index">{STEP_GLYPH[step.index] || step.index}</span>
              <span className="plan-title">{step.title}</span>
              <span className="plan-meta">
                {step.owner}
                {step.depends ? ` · 等 ${step.depends}` : ""} · 产出：{step.output}
              </span>
              <span className={`plan-state ${step.status}`}>
                {STEP_LABEL[step.status] || step.status}
              </span>
            </div>

            {step.summary && <div className="plan-summary">{step.summary}</div>}
            {step.result && <div className="plan-result-line">{step.result}</div>}

            {/* ③ 的段。折叠态正好一行：段名 + 四个圆点 + 卡在哪一步。 */}
            {step.segments?.length ? (
              <ul className="plan-segs">
                {step.segments.map((seg) => {
                  const segKey = `${step.id}:${seg.id}`;
                  const segOpen = !!open[segKey];
                  const stages = step.stages || [];
                  return (
                    <li key={seg.id} className={`plan-seg ${seg.status}`}>
                      <button
                        className="plan-seg-head"
                        onClick={() => setOpen((v) => ({ ...v, [segKey]: !v[segKey] }))}
                      >
                        <span className="plan-seg-id">{seg.id}</span>
                        <span className="plan-seg-title">{seg.title}</span>
                        <span className="plan-dots">
                          {stages.map((stage) => {
                            const value = seg.stages?.[stage.id] || "pending";
                            return (
                              <span
                                className={DOT[value] || "dot q"}
                                key={stage.id}
                                title={`${stage.label}：${STEP_LABEL[value] || value}`}
                              />
                            );
                          })}
                        </span>
                        <span className={`plan-state ${seg.status}`}>
                          {SEG_LABEL[seg.status] || seg.status}
                        </span>
                      </button>

                      {seg.note && <div className="plan-seg-note">{seg.note}</div>}

                      {segOpen && (
                        <ul className="plan-seg-stages">
                          {stages.map((stage) => {
                            const value = seg.stages?.[stage.id] || "pending";
                            return (
                              <li key={stage.id}>
                                <span className={DOT[value] || "dot q"} />
                                <span>{stage.label}</span>
                                <span className="plan-dim">
                                  {STEP_LABEL[value] || value}
                                </span>
                              </li>
                            );
                          })}
                        </ul>
                      )}
                    </li>
                  );
                })}
              </ul>
            ) : null}
          </li>
        ))}
      </ol>

      {/* 返工规则不改行为，只说明「计划会不会变」。不写，用户看到某一步被打回
          会以为流程失控了。 */}
      {card.rule && <div className="plan-foot">{card.rule}</div>}
      {card.beyond && <div className="plan-foot dim">{card.beyond}</div>}
      {(card.notes || []).map((line, i) => (
        <div className="plan-foot warn" key={i}>
          {line}
        </div>
      ))}
    </div>
  );
}
