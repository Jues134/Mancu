/**
 * Settings API client.
 *
 * The document pipeline is called directly from `Studio.tsx`, so this module is
 * limited to settings and provider operations.
 */

/**
 * fetch JSON, with the HTTP status folded into the error message.
 *
 * `timeoutMs` exists because a request can hang without failing: probing a
 * provider that accepts the connection but never answers leaves the button
 * spinning forever, and "no answer" is indistinguishable from "still working".
 * With a deadline the UI can say which one it is.
 */
async function jsonFetch<T>(
  path: string,
  init?: RequestInit & { timeoutMs?: number },
): Promise<T> {  const { timeoutMs, ...rest } = init || {};
  const ctrl = timeoutMs ? new AbortController() : undefined;
  const timer =
    timeoutMs && ctrl
      ? window.setTimeout(() => ctrl.abort(), timeoutMs)
      : undefined;
  try {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(rest.headers || {}) },
      ...rest,
      ...(ctrl ? { signal: ctrl.signal } : {}),
    });
    if (!res.ok) {
      let body = "";
      try {
        body = (await res.text()).slice(0, 300);
      } catch {
        /* the body is optional; the status alone is enough to report */
      }
      throw new Error(`HTTP ${res.status} ${path}${body ? ` — ${body}` : ""}`);
    }
    return (await res.json()) as T;
  } catch (err) {
    if (ctrl?.signal.aborted) {
      throw new Error(
        `请求超时：${Math.round((timeoutMs || 0) / 1000)} 秒内没有响应（地址可能不通或网关不回包）`,
      );
    }
    throw err;
  } finally {
    if (timer) window.clearTimeout(timer);
  }
}

export interface SharedModelInfo {
  enabled: boolean;
  label: string;
  model: string;
  base_url: string;
  calls_per_run: number;
  hint: string;
}

export interface FreeModelEntry {
  id: string;
  label: string;
  provider_preset: string;
  base_url: string;
  model: string;
  vision_model: string;
  keyless: boolean;
  signup_url: string;
  note: string;
  reachable: boolean | null;
}

export interface RelayAccountInfo {
  id: string;
  label: string;
  enabled: boolean;
  has_key: boolean;
  key_hint: string;
  model: string;
  quota_used: number;
  quota_total: number;
  quota_unit: string;
  usable: boolean;
  cooldown_seconds: number;
  last_error: string;
}

export interface RelayChannelInfo {
  id: string;
  preset: string;
  label: string;
  base_url: string;
  default_model: string;
  enabled: boolean;
  daily_checkin: boolean;
  last_checkin_at: number;
  last_checkin_result: string;
  account_count: number;
  usable_account_count: number;
  accounts: RelayAccountInfo[];
  quota_note: string;
  signup_url: string;
}

export interface RelayPreset {
  id: string;
  label: string;
  base_url: string;
  default_model: string;
  quota_kind: string;
  quota_note: string;
  signup_url: string;
  daily_checkin: boolean;
  docs_url: string;
}

export interface RelayInfo {
  enabled: boolean;
  rotation: string;
  max_attempts: number;
  summary: {
    enabled: boolean;
    channels: number;
    enabled_channels: number;
    accounts: number;
    usable_accounts: number;
    rotation: string;
  };
  channels: RelayChannelInfo[];
  endpoint: string;
}

export const settingsApi = {
  get: () =>
    jsonFetch<{
      providers: Array<Record<string, unknown>>;
      assignments: Record<string, string>;
      default_provider: string;
      fallbacks: Record<string, string[]>;
      shared_model: SharedModelInfo;
      agents: Array<{ id: string; label: string; resolved_provider: string }>;
      /** 是否流式输出；设置页的开关读它，缺省为开。 */
      stream?: boolean;
    }>("/api/settings"),

  presets: () =>
    jsonFetch<{
      presets: Array<Record<string, string>>;
      agents: Array<{ id: string; label: string }>;
      vision_agents: string[];
    }>("/api/settings/presets"),

  freeModels: () => jsonFetch<{ models: FreeModelEntry[] }>("/api/settings/free-models"),

  importFreeModel: (modelId: string) =>
    jsonFetch<{
      provider: Record<string, unknown>;
      needs_key: boolean;
      signup_url: string;
      note: string;
    }>(`/api/settings/free-models/${modelId}/import`, { method: "POST" }),

  save: (payload: {
    providers: Array<Record<string, unknown>>;
    assignments: Record<string, string>;
    default_provider: string;
    fallbacks: Record<string, string[]>;
    /** 省略表示“这次没改”，后端保留原值。 */
    stream?: boolean;
  }) =>
    jsonFetch<Record<string, unknown>>("/api/settings", {
      method: "PUT",
      body: JSON.stringify(payload),
    }),

  test: (payload: {
    provider_id?: string;
    base_url?: string;
    api_key?: string;
    model?: string;
    timeout?: number;
  }) =>
    jsonFetch<{
      ok: boolean;
      model?: string;
      reply?: string;
      error?: string;
      source?: "upstream" | "upstream_response" | "network" | "local";
      upstream_status?: number | null;
      upstream_body?: string;
      upstream_url?: string;
    }>(
      "/api/settings/test",
      // Do not impose a browser-side deadline. The provider's configured timeout
      // is authoritative, and the button remains spinning until that request
      // really finishes instead of inventing an earlier frontend failure.
      { method: "POST", body: JSON.stringify(payload) },
    ),

  /**
   * List every model this key can use, grouped by capability.
   *
   * `provider_id` rescans a saved provider using the key stored server-side —
   * the browser never receives the key back, so an unsaved provider must send
   * `base_url` + `api_key` explicitly.
   */
  scanModels: (payload: { provider_id?: string; base_url?: string; api_key?: string; timeout?: number }) =>
    jsonFetch<{
      ok: boolean;
      error?: string;
      source?: "upstream" | "upstream_response" | "network" | "local";
      upstream_status?: number | null;
      upstream_body?: string;
      upstream_url?: string;
      total?: number;
      counts?: Record<string, number>;
      groups?: Record<string, Array<{ id: string; label: string; reason: string }>>;
      suggested?: Record<string, string[]>;
    }>("/api/settings/models", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};

export { jsonFetch };
