import { useEffect, useState } from "react";

import SearchSettings, { type SearchSettingsValue } from "./SearchSettings";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const raw = await response.text();
    let detail = raw;
    try {
      const parsed = JSON.parse(raw) as { detail?: unknown };
      if (typeof parsed.detail === "string") detail = parsed.detail;
    } catch { /* keep the plain response */ }
    throw new Error(detail.slice(0, 500) || `请求失败（HTTP ${response.status}）`);
  }
  return response.json() as Promise<T>;
}

interface MCPTool {
  id: string;
  name: string;
  description: string;
  connected: boolean;
  enabled: boolean;
}

interface FeishuPermission {
  scope: string;
  label: string;
}

interface FeishuAccountDetail {
  name?: string;
  en_name?: string;
  email?: string;
  avatar_url?: string;
  tenant?: string;
}

interface FeishuConnection {
  id: "feishu";
  name: string;
  description: string;
  configured: boolean;
  state: "not_configured" | "not_connected" | "authorizing" | "connected" | "error";
  connected: boolean;
  enabled: boolean;
  message: string;
  account?: string;
  account_detail?: FeishuAccountDetail;
  permissions?: FeishuPermission[];
  scopes?: string[];
  authorization_url?: string;
  app_id?: string;
  app_secret_configured?: boolean;
  redirect_uri?: string;
  required_scopes?: string[];
}

interface FeishuDocument {
  title: string;
  token: string;
  type: string;
  url: string;
  owner: string;
  edited_at: string;
}

interface FeishuDocumentList {
  ok: boolean;
  documents: FeishuDocument[];
  count: number;
  reason?: string;
  detail?: string;
}

interface MCPDirectory { tools: MCPTool[]; feishu: FeishuConnection }

interface AuthorizationDialog {
  open: boolean;
  phase: "configure" | "starting" | "waiting" | "done" | "error";
  url: string;
  browserOpened: boolean | null;
  message: string;
}

interface FeishuSettingsDraft {
  app_id: string;
  app_secret: string;
  redirect_uri: string;
}

const CLOSED_AUTH_DIALOG: AuthorizationDialog = {
  open: false,
  phase: "starting",
  url: "",
  browserOpened: null,
  message: "",
};

function suggestedFeishuCallback(): string {
  return `${window.location.origin}/api/mcp/feishu/callback`;
}

/**
 * Open the authorization page in the user's real, default browser.
 *
 * `window.open` with a target name opens a small popup, and in a wrapped or
 * kiosk shell it can be swallowed into an in-app window — the user then faces
 * a login form they cannot scan with their phone. `_blank` with `noopener`
 * is the one form every browser routes to the system default browser, which is
 * what "扫码登录" actually needs.
 *
 * Returns false when the browser blocked it, so the caller can show the raw
 * link instead of silently doing nothing.
 */
function openInDefaultBrowser(url: string): boolean {
  try {
    const opened = window.open(url, "_blank", "noopener,noreferrer");
    if (opened) return true;
    // Some managers return null yet still navigate. A hidden anchor click is
    // the fallback and inherits the current window's permission state.
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.target = "_blank";
    anchor.rel = "noopener noreferrer";
    anchor.style.display = "none";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    return true;
  } catch {
    return false;
  }
}

export function MCPSettings({ active }: { active: boolean }) {
  const [directory, setDirectory] = useState<MCPDirectory | null>(null);
  const [loading, setLoading] = useState(true);
  const [operating, setOperating] = useState("");
  const [error, setError] = useState("");
  const [authorization, setAuthorization] = useState<AuthorizationDialog>(CLOSED_AUTH_DIALOG);
  const [feishuSettings, setFeishuSettings] = useState<FeishuSettingsDraft>({
    app_id: "",
    app_secret: "",
    redirect_uri: "",
  });
  const [feishuSettingsError, setFeishuSettingsError] = useState("");
  const [documents, setDocuments] = useState<FeishuDocumentList | null>(null);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  // 资料检索 的联网检索配置。放在这里而不是知识库页：它管的是**工具**的行为，
  // 和工具的启停是同一件事的两面。
  const [searchSettings, setSearchSettings] = useState<SearchSettingsValue | null>(null);
  const [searchOpen, setSearchOpen] = useState(false);

  async function loadSearchSettings() {
    try { setSearchSettings(await request<SearchSettingsValue>("/api/knowledge/settings")); }
    catch { /* 读不到就保持 null，展开时会显示未取到，不影响工具启停 */ }
  }

  async function load(silent = false) {
    if (!silent) setLoading(true);
    setError("");
    try {
      const [next] = await Promise.all([
        request<MCPDirectory>("/api/mcp/directory"),
        loadSearchSettings(),
      ]);
      setDirectory(next);
    }
    catch (exc) { setError(exc instanceof Error ? exc.message : String(exc)); }
    finally { if (!silent) setLoading(false); }
  }

  /**
   * Read the documents this account can actually see.
   *
   * This is the proof the login worked. A "已连接" badge is a claim; a list of
   * the user's own documents is evidence, and it also surfaces a silent scope
   * gap (connected, but cannot list) before the user tries to use the agent.
   */
  async function loadDocuments() {
    setDocumentsLoading(true);
    try { setDocuments(await request<FeishuDocumentList>("/api/mcp/feishu/docs?limit=20")); }
    catch (exc) {
      setDocuments({
        ok: false, documents: [], count: 0,
        reason: exc instanceof Error ? exc.message : String(exc),
      });
    } finally { setDocumentsLoading(false); }
  }

  useEffect(() => { if (active) void load(); }, [active]);

  // 已登录就拉一次文档清单；账号变了（重新登录）也要重新拉。
  useEffect(() => {
    if (!active || !directory?.feishu.connected) { setDocuments(null); return; }
    void loadDocuments();
  }, [active, directory?.feishu.connected, directory?.feishu.account]);

  // 取消登录：让后端清掉待授权状态，
  // 这样界面会回到「未登录」，而不是一直停在「授权中」。
  // 浏览器那一页不归我们管，用户自己关掉即可。
  async function cancelAuthorization() {
    setAuthorization(CLOSED_AUTH_DIALOG);
    try {
      await request("/api/mcp/feishu/cancel", { method: "POST" });
    } catch (exc) {
      // 失败也要说清楚，否则状态还在，用户不知道发生了什么。
      setError(exc instanceof Error ? exc.message : String(exc));
    }
    await load(true);
  }

  useEffect(() => {
    if (!active || directory?.feishu.state !== "authorizing") return;
    let cancelled = false;
    let timer = 0;
    const poll = async () => {
      try {
        const state = await request<Partial<FeishuConnection>>("/api/mcp/feishu/status");
        if (!cancelled) setDirectory(current => current ? {
          ...current,
          feishu: { ...current.feishu, ...state },
        } : current);
        if (!cancelled && state.connected) setAuthorization(current => ({
          ...current,
          open: true,
          phase: "done",
          url: "",
          message: "飞书账号授权成功，现在可以读取已授权的云文档。",
        }));
      } catch { /* 状态还没落库时继续轮询 */ }
      if (!cancelled) timer = window.setTimeout(poll, 1800);
    };
    timer = window.setTimeout(poll, 1200);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [active, directory?.feishu.state]);

  // 登录成功后停留 3 秒，然后自动关闭弹窗。
  useEffect(() => {
    if (authorization.phase !== "done" || !authorization.open) return;
    const timer = window.setTimeout(() => setAuthorization(CLOSED_AUTH_DIALOG), 3000);
    return () => window.clearTimeout(timer);
  }, [authorization.phase, authorization.open]);

  useEffect(() => {
    if (!active) return;
    const receive = (event: MessageEvent) => {
      // 回调页在后端（:8100），设置页开发时在 Vite（:5173），
      // 所以不能按同源校验，否则消息会被丢掉。消息里只有成功标记和账号名，
      // 不含凭证，真正的状态始终以后端接口为准。
      if (event.data?.type !== "feishu-oauth") return;
      if (event.data?.ok) setAuthorization(current => ({
        ...current,
        open: true,
        phase: "done",
        url: "",
        message: "飞书账号授权成功，现在可以读取已授权的云文档。",
      }));
      else setAuthorization(current => ({
        ...current,
        open: true,
        phase: "error",
        message: "飞书没有完成授权，请重新打开授权网页后再试。",
      }));
      void load(true);
    };
    window.addEventListener("message", receive);
    return () => window.removeEventListener("message", receive);
  }, [active]);

  async function toggle(service: "knowledge" | "feishu", enabled: boolean) {
    setOperating(`${service}-switch`);
    setError("");
    try {
      await request(`/api/mcp/tools/${service}/enabled`, {
        method: "PUT",
        body: JSON.stringify({ enabled }),
      });
      await load(true);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setOperating("");
    }
  }

  function openFeishuSettings(
    connection = directory?.feishu,
    message = "请填写飞书开放平台应用信息，保存后将继续登录。",
  ) {
    setError("");
    setFeishuSettingsError("");
    setFeishuSettings({
      app_id: connection?.app_id || "",
      // App Secret 永不从后端回显。已配置时留空表示沿用原值。
      app_secret: "",
      redirect_uri: connection?.redirect_uri || suggestedFeishuCallback(),
    });
    setAuthorization({
      open: true,
      phase: "configure",
      url: "",
      browserOpened: null,
      message,
    });
  }

  async function startFeishuAuthorization() {
    setOperating("feishu-login");
    setError("");
    setAuthorization({
      open: true,
      phase: "starting",
      url: "",
      browserOpened: null,
      message: "正在创建飞书网页授权连接…",
    });
    try {
      const state = await request<FeishuConnection>("/api/mcp/feishu/connect", { method: "POST" });
      if (!state.authorization_url) throw new Error("飞书没有返回网页授权地址");
      // 打开默认浏览器整页授权。这里不能判断「有没有真的弹出来」——
      // 整页跳转由操作系统接管，成功与否我们看不到，所以提示语写成
      // 「已发送到浏览器」而不是「已打开」，并始终把链接亮在下面。
      const browserOpened = openInDefaultBrowser(state.authorization_url);
      setAuthorization({
        open: true,
        phase: "waiting",
        url: state.authorization_url,
        browserOpened,
        message: browserOpened
          ? "已在默认浏览器打开飞书授权页，请扫码或登录。完成后本程序会自动识别。"
          : "浏览器拦截了跳转，请点击下面的授权链接手动打开。",
      });
      await load(true);
    } catch (exc) {
      const message = exc instanceof Error ? exc.message : String(exc);
      if (/尚未.*配置|App ID|App Secret|回调地址/.test(message)) {
        openFeishuSettings(directory?.feishu, message);
      } else {
        setError(message);
        setAuthorization({
          open: true,
          phase: "error",
          url: "",
          browserOpened: false,
          message,
        });
      }
    } finally {
      setOperating("");
    }
  }

  async function loginFeishu() {
    if (!directory?.feishu.configured) {
      openFeishuSettings();
      return;
    }
    await startFeishuAuthorization();
  }

  async function saveFeishuSettingsAndLogin() {
    const appId = feishuSettings.app_id.trim();
    const secret = feishuSettings.app_secret.trim();
    const callback = feishuSettings.redirect_uri.trim();
    if (!appId.startsWith("cli_")) {
      setFeishuSettingsError("App ID 格式不正确，应以 cli_ 开头。");
      return;
    }
    if (!secret && !directory?.feishu.app_secret_configured) {
      setFeishuSettingsError("请填写 App Secret。");
      return;
    }
    try {
      const parsed = new URL(callback);
      if (!/^https?:$/.test(parsed.protocol)) throw new Error();
    } catch {
      setFeishuSettingsError("回调地址必须是完整的 http:// 或 https:// 地址。");
      return;
    }

    setOperating("feishu-settings");
    setFeishuSettingsError("");
    try {
      await request<FeishuConnection>("/api/mcp/feishu/settings", {
        method: "PUT",
        body: JSON.stringify({
          app_id: appId,
          app_secret: secret,
          redirect_uri: callback,
        }),
      });
      await load(true);
      await startFeishuAuthorization();
    } catch (exc) {
      setFeishuSettingsError(exc instanceof Error ? exc.message : String(exc));
      setAuthorization(current => ({
        ...current,
        open: true,
        phase: "configure",
        message: "飞书应用信息尚未保存，请检查后重试。",
      }));
    } finally {
      setOperating("");
    }
  }

  async function logoutFeishu() {
    setOperating("feishu-logout");
    setError("");
    try {
      await request("/api/mcp/feishu", { method: "DELETE" });
      await load(true);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setOperating("");
    }
  }

  function reopenAuthorization() {
    if (!authorization.url) return;
    const opened = openInDefaultBrowser(authorization.url);
    setAuthorization(current => ({
      ...current,
      browserOpened: opened,
      message: opened
        ? "已在默认浏览器打开飞书授权页，请扫码或登录。"
        : "浏览器拦截了跳转，请复制下面的链接到浏览器打开。",
    }));
  }

  const knowledge = directory?.tools.find(tool => tool.id === "search_knowledge");
  const feishu = directory?.feishu;
  const permissionList = feishu?.permissions ?? [];
  const searchProvider = searchSettings?.providers.find(
    item => item.value === searchSettings.provider,
  )?.label || searchSettings?.provider || "";
  return <div className="settings-main mcp-settings-simple">
    <section className="panel">
      <h2>MCP 工具</h2>
      <p className="hint">管理 Agent 可以使用的资料服务。停用只关闭调用权限，不删除账号授权。</p>
      {error && <div className="banner fail" role="alert">{error}</div>}
      {loading && !directory && <p className="muted">正在读取工具状态…</p>}
      {directory && <div className="mcp-service-list">
        {knowledge && <article className="mcp-service-row">
          <div className="mcp-service-info">
            <div className="mcp-service-title">资料检索</div>
            <div className="muted">
              检索本地资料库和已配置的权威网站
              {searchSettings && (
                <> · 联网检索：{searchSettings.has_key ? searchProvider : "未配置搜索服务"}</>
              )}
            </div>
          </div>
          <div className="mcp-service-actions">
            <button
              className="btn ghost"
              onClick={() => {
                setSearchOpen(current => !current);
                if (!searchSettings) void loadSearchSettings();
              }}
            >{searchOpen ? "收起设置" : "检索设置"}</button>
            <span className={`mcp-status ${knowledge.enabled ? "connected" : ""}`}>{knowledge.enabled ? "已启用" : "已停用"}</span>
            <button
              className="btn ghost"
              disabled={Boolean(operating)}
              onClick={() => void toggle("knowledge", !knowledge.enabled)}
            >{operating === "knowledge-switch" ? "处理中…" : knowledge.enabled ? "停用" : "启用"}</button>
          </div>
        </article>}

        {knowledge && searchOpen && <SearchSettings
          value={searchSettings}
          onClose={() => setSearchOpen(false)}
          onSaved={(next) => setSearchSettings(next)}
        />}

        {feishu && <article className="mcp-service-row">
          <div className="mcp-service-info">
            <div className="mcp-service-title">飞书</div>
            <div className="muted">
              {feishu.connected
                ? `${feishu.account || "当前账号"} · ${feishu.enabled ? "允许 Agent 读取已授权云文档" : "账号仍保持登录"}`
                : feishu.configured
                  ? "登录后可在默认浏览器扫码，读取当前账号授权的云文档"
                  : "登录服务尚未由项目管理员配置"}
            </div>
          </div>
          <div className="mcp-service-actions">
            <span className={`mcp-status ${feishu.connected && feishu.enabled ? "connected" : ""}`}>
              {!feishu.connected ? "未登录" : feishu.enabled ? "已启用" : "已停用"}
            </span>
            {!feishu.connected && <button
              className="btn ghost"
              disabled={Boolean(operating)}
              onClick={() => openFeishuSettings(
                feishu,
                feishu.configured
                  ? "可以修改飞书应用信息；App Secret 留空将沿用已保存的值。"
                  : "请填写飞书开放平台应用信息，保存后将继续登录。",
              )}
            >登录配置</button>}
            {!feishu.connected ? <button
              className="btn primary"
              disabled={Boolean(operating)}
              onClick={() => {
                if (authorization.url && !authorization.open) {
                  setAuthorization(current => ({ ...current, open: true }));
                } else {
                  void loginFeishu();
                }
              }}
            >{operating === "feishu-login" ? "打开中…" : "登录"}</button> : <>
              <button
                className="btn ghost"
                disabled={Boolean(operating)}
                onClick={() => void toggle("feishu", !feishu.enabled)}
              >{operating === "feishu-switch" ? "处理中…" : feishu.enabled ? "停用" : "启用"}</button>
              <button
                className="btn ghost"
                disabled={Boolean(operating)}
                onClick={() => void logoutFeishu()}
              >{operating === "feishu-logout" ? "登出中…" : "登出"}</button>
            </>}
          </div>
        </article>}
      </div>}

      {/* 登录后的账号卡片：把「我是谁、拿到什么权限」摊开给用户看。
          权限清单来自这次授权实际返回的 scope，不是我们申请的清单。 */}
      {feishu?.connected && <div className="feishu-account">
        <div className="feishu-account-head">
          {feishu.account_detail?.avatar_url
            ? <img src={feishu.account_detail.avatar_url} alt="" className="feishu-avatar" />
            : <div className="feishu-avatar placeholder">{(feishu.account || "飞").slice(0, 1)}</div>}
          <div className="feishu-account-id">
            <div className="feishu-account-name">{feishu.account || "飞书账号"}</div>
            <div className="muted">
              {[feishu.account_detail?.tenant, feishu.account_detail?.email]
                .filter(Boolean).join(" · ") || "已授权飞书账号"}
            </div>
          </div>
        </div>
        <div className="feishu-scopes">
          {permissionList.length
            ? permissionList.map(item => (
                <span className="feishu-scope" key={item.scope} title={item.scope}>{item.label}</span>
              ))
            : <span className="muted">本次授权没有返回权限清单</span>}
        </div>
      </div>}

      {/* 云文档清单：登录是否真的生效，靠这个列表证明。 */}
      {feishu?.connected && <div className="feishu-docs">
        <div className="feishu-docs-head">
          <h3>可访问的云文档</h3>
          <button
            className="btn ghost"
            disabled={documentsLoading}
            onClick={() => void loadDocuments()}
          >{documentsLoading ? "读取中…" : "刷新"}</button>
        </div>
        {documentsLoading && !documents && <p className="muted">正在读取当前账号的云文档…</p>}
        {documents && documents.ok && documents.documents.length === 0 && (
          <p className="muted">当前账号下暂时没有可见文档。</p>
        )}
        {documents && !documents.ok && (
          <p className="muted">{documents.reason || "读取文档列表失败"}；Agent 仍可通过搜索查找文档。</p>
        )}
        {documents && documents.documents.length > 0 && <ul className="feishu-doc-list">
          {documents.documents.map(doc => (
            <li key={doc.token}>
              <a href={doc.url} target="_blank" rel="noreferrer" className="feishu-doc-title">
                {doc.title}
              </a>
              <span className="feishu-doc-meta">
                {doc.type}{doc.owner ? ` · ${doc.owner}` : ""}
              </span>
            </li>
          ))}
        </ul>}
        <p className="muted feishu-docs-note">
          这里只列出当前账号有权限访问的文档，点标题在浏览器中打开。
        </p>
      </div>}
    </section>
    {authorization.open && <div className="mcp-auth-backdrop" role="presentation">
      <section className="mcp-auth-dialog" role="dialog" aria-modal="true" aria-labelledby="mcp-auth-title">
        <div className={`mcp-auth-mark ${authorization.phase}`} aria-hidden="true">
          {authorization.phase === "done" ? "✓" : authorization.phase === "error" ? "!" : authorization.phase === "configure" ? "✎" : "↗"}
        </div>
        <h3 id="mcp-auth-title">
          {authorization.phase === "done" ? "飞书登录完成" : authorization.phase === "error" ? "飞书登录未完成" : authorization.phase === "configure" ? "飞书登录配置" : "登录飞书"}
        </h3>
        <p className="mcp-auth-message">{authorization.message}</p>
        {authorization.phase === "configure" && <form
          className="feishu-settings-form"
          onSubmit={(event) => { event.preventDefault(); void saveFeishuSettingsAndLogin(); }}
        >
          <label>
            <span>App ID</span>
            <input
              autoFocus
              autoComplete="off"
              placeholder="cli_xxxxxxxxxxxxxxxx"
              value={feishuSettings.app_id}
              onChange={(event) => setFeishuSettings(current => ({ ...current, app_id: event.target.value }))}
            />
          </label>
          <label>
            <span>App Secret</span>
            <input
              type="password"
              autoComplete="new-password"
              placeholder={feishu?.app_secret_configured ? "已保存；留空表示不修改" : "填写飞书应用的 App Secret"}
              value={feishuSettings.app_secret}
              onChange={(event) => setFeishuSettings(current => ({ ...current, app_secret: event.target.value }))}
            />
          </label>
          <label>
            <span>回调地址（Redirect URL）</span>
            <input
              autoComplete="off"
              placeholder={suggestedFeishuCallback()}
              value={feishuSettings.redirect_uri}
              onChange={(event) => setFeishuSettings(current => ({ ...current, redirect_uri: event.target.value }))}
            />
          </label>
          <p className="muted">
            请把完全相同的回调地址添加到飞书开放平台应用的安全设置中。配置只保存在本机服务端，App Secret 不会回显到浏览器。
          </p>
          {feishuSettingsError && <div className="feishu-settings-error" role="alert">{feishuSettingsError}</div>}
          <div className="mcp-auth-buttons">
            <button type="button" className="btn ghost" onClick={() => setAuthorization(CLOSED_AUTH_DIALOG)}>取消</button>
            <button type="submit" className="btn primary" disabled={operating === "feishu-settings"}>
              {operating === "feishu-settings" ? "保存中…" : "保存并继续登录"}
            </button>
          </div>
        </form>}
        {authorization.phase === "waiting" && authorization.url && <>
          <div className={`mcp-browser-state ${authorization.browserOpened ? "opened" : "blocked"}`}>
            {authorization.browserOpened ? "已发送到默认浏览器" : "浏览器拦截了跳转"}
          </div>
          <a className="mcp-auth-url" href={authorization.url} target="_blank" rel="noreferrer">
            {authorization.url}
          </a>
          <p className="muted">
            默认浏览器里的授权页会显示账号、权限和可访问的云文档；扫码完成后本程序会自动识别，无需回来点任何按钮。
          </p>
        </>}
        {authorization.phase === "starting" && <div className="mcp-auth-progress">正在连接飞书开放平台…</div>}
        {authorization.phase !== "configure" && <div className="mcp-auth-buttons">
          {authorization.phase === "waiting" && authorization.url && <button className="btn ghost" onClick={reopenAuthorization}>在浏览器中打开</button>}
          {authorization.phase === "error" && <button className="btn primary" onClick={() => void loginFeishu()}>重新登录</button>}
          {authorization.phase === "done"
            ? <button className="btn primary" onClick={() => setAuthorization(CLOSED_AUTH_DIALOG)}>完成</button>
            : <button className="btn ghost" onClick={() => void cancelAuthorization()}>
                {authorization.phase === "waiting" ? "取消登录" : "关闭"}
              </button>}
        </div>}
        {authorization.phase === "waiting" && <p className="muted">关闭这个提示不会中断浏览器里已经打开的授权页。</p>}
      </section>
    </div>}
  </div>;
}
