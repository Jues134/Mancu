import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import Studio from "./pages/Studio";
import "./styles.css";
import "./history.css";
import { readMotionMode, applyMotionMode } from "./motionMode";

// 挂载前定好 <html data-motion>：全局 reduce 规则要靠它决定是否压制动画。
// 默认「强制开」，系统关了动画的机器也能看到大屏光流。
applyMotionMode(readMotionMode());

/**
 * One entry point: the workbench.
 *
 * Settings used to be a route (`/settings`), which meant leaving the conversation
 * to change a model. It is now a dialog opened from the workbench — see the gear
 * button in Studio's topbar — so there is no second page to route to.
 *
 * `/settings`, `/relay` and `/form` are kept as redirects for old bookmarks and for
 * documentation that still points at them. Serving a second page would keep the
 * duplicate route alive, which is the thing being removed; a 404 would break links
 * for no benefit.
 */
const path = window.location.pathname.replace(/\/+$/, "") || "/";

if (path === "/settings" || path === "/relay" || path === "/form") {
  // Keep the hash: `/settings#free-models` used to deep-link a section.
  window.location.replace(`/${window.location.hash || ""}`);
}

const root = document.getElementById("root");
if (!root) throw new Error("找不到 #root 挂载点");

createRoot(root).render(
  <StrictMode>
    <Studio />
  </StrictMode>,
);
