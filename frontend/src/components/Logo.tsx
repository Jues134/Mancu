/**
 * 品牌标志：两个人形在中间交汇成 M 形（对应 frontend/public/logo.svg）。
 *
 * 内联而不是 <img src="/logo.svg">：顶栏只有几十像素高，内联省一次请求，
 * 而且和文字在同一行渲染，不会出现图片异步加载完把标题顶一下的跳动。
 * 颜色取自原图：珊瑚（左）、绛红（右）、奶油底。
 */
export default function Logo({ size = 26 }: { size?: number }) {
  return (
    <svg
      className="brand-logo"
      width={size}
      height={size}
      viewBox="0 0 200 200"
      role="img"
      aria-label="漫簇"
    >
      <defs>
        <linearGradient id="logo-coral" x1="0.8" y1="0.08" x2="0.12" y2="0.95">
          <stop offset="0" stopColor="#d08160" />
          <stop offset="1" stopColor="#a8442e" />
        </linearGradient>
        <linearGradient id="logo-fold" x1="0.2" y1="0.9" x2="0.85" y2="0.1">
          <stop offset="0" stopColor="#dd9e7f" />
          <stop offset="1" stopColor="#cb7454" />
        </linearGradient>
        <linearGradient id="logo-crimson" x1="0.15" y1="0.1" x2="0.9" y2="0.95">
          <stop offset="0" stopColor="#8d1423" />
          <stop offset="1" stopColor="#c52532" />
        </linearGradient>
        <radialGradient id="logo-headL" cx="0.35" cy="0.3" r="1">
          <stop offset="0" stopColor="#d38b68" />
          <stop offset="1" stopColor="#b0563e" />
        </radialGradient>
        <radialGradient id="logo-headR" cx="0.35" cy="0.3" r="1">
          <stop offset="0" stopColor="#a32330" />
          <stop offset="1" stopColor="#7b1120" />
        </radialGradient>
      </defs>
      <rect x="6" y="6" width="188" height="188" rx="48" fill="#f8e9db" />
      <path
        d="M61 156 L86 106 L104 140"
        fill="none"
        stroke="url(#logo-coral)"
        strokeWidth="27"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M104 140 L120 110"
        fill="none"
        stroke="url(#logo-fold)"
        strokeWidth="27"
        strokeLinecap="round"
      />
      <path
        d="M120 110 L143 156"
        fill="none"
        stroke="url(#logo-crimson)"
        strokeWidth="27"
        strokeLinecap="round"
      />
      <circle cx="76" cy="64" r="16.5" fill="url(#logo-headL)" />
      <circle cx="127" cy="64" r="16.5" fill="url(#logo-headR)" />
    </svg>
  );
}
