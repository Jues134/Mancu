import { useEffect, useState } from "react";

/**
 * 资料检索 的联网检索设置。
 *
 * 这张卡原来放在知识库面板里，但它管的东西不是「库里有什么」，而是**内置检索
 * 工具在本地索引没命中时怎么去网上找**——搜索服务、凭证、允许来源白名单。
 * 所以它归在「MCP 工具 → 资料检索」下面，跟工具的启停放在一起：一个工具的能力
 * 和它的开关、凭证该在同一页。
 */

export interface SearchSettingsValue {
  provider: string;
  has_key: boolean;
  key_hint: string;
  endpoint: string;
  providers: { value: string; label: string }[];
  allowed_sites: { domain: string; label: string; tier: number }[];
}

type AllowedSite = SearchSettingsValue["allowed_sites"][number];

/** 各搜索服务的 Key 获取步骤。问号按钮展开的内容，跟所选服务走。 */
const KEY_HELP: Record<string, { name: string; steps: string[]; note?: string }> = {
  bocha: {
    name: "博查 Bocha",
    steps: [
      "浏览器打开 open.bochaai.com，点右上角「控制台」，微信扫码或手机号登录，不需要信用卡。",
      "左侧「API KEY 管理」→「创建 API KEY」。密钥只在创建时显示这一次，立即复制保存。",
      "粘贴到这里的是 sk- 开头的那串 key，不要带上 Bearer 前缀。",
      "免费额度：左侧「资源包管理」可领 0 元 1000 次试用；「总览 → 账户可用额度 → 兑换」输入口令「博查搜索」再领 1000 次。",
    ],
    note: "国内直连可用，中文结果最好，是默认推荐。",
  },
  tavily: {
    name: "Tavily",
    steps: [
      "浏览器打开 app.tavily.com，用邮箱 / Google / GitHub 注册，免信用卡。",
      "登录后 key 直接显示在首页（tvly- 开头），复制粘贴到这里即可。",
      "每月免费 1000 次，月初自动重置。",
    ],
    note: "英文源更强，但需要能访问国外网站；中文源覆盖不如博查。",
  },
  serper: {
    name: "Serper",
    steps: [
      "浏览器打开 serper.dev，注册后 key 显示在 dashboard 首页。",
      "新用户免费 2500 次。",
    ],
    note: "走 Google 检索，必须有能访问 Google 的网络环境，国内不推荐。",
  },
};

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
    } catch {
      /* 状态码本身就够说明问题 */
    }
    throw new Error(detail.slice(0, 300) || `请求失败（HTTP ${response.status}）`);
  }
  return response.json() as Promise<T>;
}

export default function SearchSettings({
  value,
  onSaved,
  onClose,
}: {
  /** 由父组件预先取回的配置；为 null 时表示还没读到。 */
  value: SearchSettingsValue | null;
  onSaved: (next: SearchSettingsValue) => void;
  onClose: () => void;
}) {
  const [provider, setProvider] = useState(value?.provider || "bocha");
  const [apiKey, setApiKey] = useState("");
  const [sites, setSites] = useState<AllowedSite[]>(value?.allowed_sites || []);
  const [siteLabel, setSiteLabel] = useState("");
  const [siteDomain, setSiteDomain] = useState("");
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [keyHelpOpen, setKeyHelpOpen] = useState(false);

  useEffect(() => {
    if (!value) return;
    setProvider(value.provider);
    setSites(value.allowed_sites);
  }, [value]);

  function addSite() {
    const domain = siteDomain.trim();
    if (!domain) {
      setError("请填写来源域名或网址");
      return;
    }
    if (sites.some((site) => site.domain.toLowerCase() === domain.toLowerCase())) {
      setError("这个来源已经在允许名单中");
      return;
    }
    setSites((current) => [
      ...current,
      { domain, label: siteLabel.trim() || domain, tier: 2 },
    ]);
    setSiteLabel("");
    setSiteDomain("");
    setError("");
  }

  async function save() {
    setSaving(true);
    setError("");
    try {
      const next = await request<SearchSettingsValue>("/api/knowledge/settings", {
        method: "POST",
        body: JSON.stringify({ provider, api_key: apiKey, allowed_sites: sites }),
      });
      onSaved(next);
      setApiKey("");
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setSaving(false);
    }
  }

  if (!value) {
    return (
      <div className="mcp-search-settings">
        <p className="muted">正在读取检索设置…</p>
      </div>
    );
  }

  return (
    <div className="mcp-search-settings">
      <div className="mcp-search-settings-head">
        <div>
          <h3>联网检索设置</h3>
          <p className="kb-settings-note">
            本地索引没有命中时，这个工具才会用它去网上找候选链接。只接受白名单内的机构，
            其余结果直接丢弃——宁可这一条查不到，也不拿百科充数。
          </p>
        </div>
        <span className={`mcp-status ${value.has_key ? "connected" : ""}`}>
          {value.has_key ? `已配置 ${value.key_hint}` : "未配置"}
        </span>
      </div>

      <label className="kb-field">
        <span>搜索服务</span>
        <select value={provider} onChange={(e) => setProvider(e.target.value)}>
          {(value.providers || []).map((p) => (
            <option key={p.value} value={p.value}>
              {p.label}
            </option>
          ))}
        </select>
        <small>只负责搜索候选网页链接，不生成正文或视频。</small>
      </label>

      <div className="kb-field">
        <div className="kb-field-head">
          <span>API Key</span>
          <button
            type="button"
            className={`kb-key-help-btn ${keyHelpOpen ? "open" : ""}`}
            aria-label="怎么获取这个 Key"
            aria-expanded={keyHelpOpen}
            title="怎么获取这个 Key"
            onClick={() => setKeyHelpOpen((current) => !current)}
          >?</button>
        </div>
        <input
          type="password"
          value={apiKey}
          placeholder={value.has_key ? `已配置 ${value.key_hint}，留空不修改` : "粘贴搜索服务的 Key"}
          onChange={(e) => setApiKey(e.target.value)}
        />
        <small>
          所选搜索服务的访问凭证，不是模型 API Key。搜到链接后仍会按下方名单过滤并抓取正文。
        </small>
        {keyHelpOpen && (() => {
          const help = KEY_HELP[provider] || {
            name: provider,
            steps: ["请在该服务的官网注册账号并创建 API Key，然后粘贴到这里。"],
            note: "",
          };
          return <div className="kb-key-help" role="note">
            <h4>怎么获取 {help.name} 的 Key</h4>
            <ol>
              {help.steps.map((step) => <li key={step}>{step}</li>)}
            </ol>
            {help.note && <p className="muted">{help.note}</p>}
          </div>;
        })()}
      </div>

      {error && <div className="kb-error">{error}</div>}

      <div className="kb-settings-actions">
        <button className="btn primary small" disabled={saving} onClick={() => void save()}>
          {saving ? "保存中…" : "保存"}
        </button>
        <button className="btn ghost small" onClick={onClose}>
          收起
        </button>
      </div>

      <div className="kb-allowed">
        <h4>
          允许的来源 <span>{sites.length} 个</span>
        </h4>
        <p>可填写域名或粘贴完整网址。删除后，该网站的新检索结果和跳转页面都会被拒绝。</p>
        <div className="kb-site-add">
          <input
            value={siteLabel}
            placeholder="机构名称（可选）"
            onChange={(event) => setSiteLabel(event.target.value)}
          />
          <input
            value={siteDomain}
            placeholder="例如 gov.cn 或 https://example.gov.cn/"
            onChange={(event) => setSiteDomain(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                addSite();
              }
            }}
          />
          <button className="btn ghost small" type="button" onClick={addSite}>
            新增来源
          </button>
        </div>
        <ul>
          {sites.map((site) => (
            <li key={site.domain}>
              <div>
                <span>{site.label}</span>
                <code>{site.domain}</code>
              </div>
              <button
                className="btn ghost small danger"
                type="button"
                aria-label={`删除 ${site.label}`}
                onClick={() =>
                  setSites((current) => current.filter((item) => item.domain !== site.domain))
                }
              >
                删除
              </button>
            </li>
          ))}
          {!sites.length && (
            <li className="kb-sites-empty">当前没有允许来源，联网结果将全部被拒绝。</li>
          )}
        </ul>
      </div>
    </div>
  );
}
