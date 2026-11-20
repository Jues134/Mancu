/**
 * 大屏「光流动效」偏好：设置页「常规」写入，AgentBigScreen 读取。
 *
 * 用 localStorage 持久化 + 自定义事件广播，而不是把它们挂到 React context：
 * 大屏和设置页分属两棵渲染树，中间还隔着路由，context 要一路穿到最外层；
 * 一个模块级单例 + 事件更直接，且设置改动能在已打开的大屏上立刻生效。
 *
 * 默认 "on"（强制开）。系统级动画被关掉时 prefers-reduced-motion 会命中，
 * 若默认跟随系统，用户第一次打开大屏看到的就是一片静止的线，还以为坏了。
 */
export type MotionMode = "on" | "system" | "off";

const STORAGE_KEY = "mancu.motionMode";
const CHANGE_EVENT = "mancu:motion-mode-change";

export function readMotionMode(): MotionMode {
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved === "on" || saved === "system" || saved === "off") return saved;
  } catch {
    /* storage 可能被禁；用默认值 */
  }
  return "on";
}

export function writeMotionMode(mode: MotionMode): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, mode);
  } catch {
    /* 存不进去也要让当前页面生效 */
  }
  applyMotionMode(mode);
  window.dispatchEvent(new CustomEvent<MotionMode>(CHANGE_EVENT, { detail: mode }));
}

/**
 * 把模式写到 <html data-motion=…>。styles.css 里那条全局
 * `prefers-reduced-motion` 压制规则带 `* { animation-duration:.01ms !important }`，
 * 系统关了动画时会把光流摁成静止；只有标记了「强制开」才豁免它。
 * 必须在 React 挂载前调用一次，否则默认 on 在首次渲染时还没生效。
 */
export function applyMotionMode(mode: MotionMode): void {
  if (typeof document !== "undefined") {
    document.documentElement.dataset.motion = mode;
  }
}

/** 订阅变化；返回取消订阅函数。同时监听跨标签页的 storage 事件。 */
export function subscribeMotionMode(listener: (mode: MotionMode) => void): () => void {
  const onLocal = (e: Event) => listener((e as CustomEvent<MotionMode>).detail);
  const onOtherTab = (e: StorageEvent) => {
    if (e.key === STORAGE_KEY) listener(readMotionMode());
  };
  window.addEventListener(CHANGE_EVENT, onLocal);
  window.addEventListener("storage", onOtherTab);
  return () => {
    window.removeEventListener(CHANGE_EVENT, onLocal);
    window.removeEventListener("storage", onOtherTab);
  };
}

/** 是否应关闭动效：手动关，或「跟随系统」且系统要求减少动效。 */
export function isMotionOff(mode: MotionMode, systemReduced: boolean): boolean {
  return mode === "off" || (mode === "system" && systemReduced);
}
