import { useEffect, useState } from "react";

/** `mm:ss`, so a two-minute wait reads as 02:10 and not as 130.
 *
 * Seconds alone stop feeling like progress past a minute, and a clock that only
 * changes every 60 s is indistinguishable from a frozen one — which is the
 * impression this helper exists to remove.
 */
export function formatElapsed(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const m = Math.floor(total / 60);
  const sec = total % 60;
  return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

type WorkingHeartbeat = { agent?: string; label?: string; started_at?: number; seconds?: number };

/** Elapsed seconds for the working agent, ticking once a second.
 *
 * The server is the clock: `started_at` comes from it, so a client whose own
 * clock is off still shows a sensible duration. The interval only re-renders
 * the component that reads it, not the whole conversation.
 */
export function useElapsedSeconds(working?: WorkingHeartbeat): number {
  const [now, setNow] = useState(() => Date.now() / 1000);
  const startedAt = Number(working?.started_at) || 0;

  useEffect(() => {
    if (!working?.agent) return;
    // Resync immediately: the poll that delivered this heartbeat may be a second
    // or two behind, and a timer that starts from a stale `now` drifts.
    setNow(Date.now() / 1000);
    const timer = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(timer);
  }, [working?.agent, startedAt]);

  if (!working?.agent) return 0;
  return startedAt > 0 ? Math.max(0, Math.floor(now - startedAt)) : Number(working.seconds) || 0;
}

/** A shared clock, ticking once a second while `active`.
 *
 * Separate from `useElapsedSeconds` because the big screen lists one row per
 * concurrent agent: one tick drives many durations, and a hook cannot be called
 * in a loop.
 */
export function useTick(active: boolean): number {
  const [now, setNow] = useState(() => Date.now() / 1000);

  useEffect(() => {
    if (!active) return;
    setNow(Date.now() / 1000);
    const timer = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(timer);
  }, [active]);

  return now;
}
