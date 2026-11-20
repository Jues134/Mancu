import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

/**
 * Full-screen preview for a generated image.
 *
 * Clips already have this affordance — the <video> element's own controls offer
 * fullscreen — but a reference sheet only ever appeared at thumbnail size, which
 * is exactly the size at which you cannot judge the face. Clicking the picture
 * now opens it here.
 *
 * Two things the overlay must get right that a plain <img> does not:
 *
 *   · It must escape the conversation's scroll container, so it is portalled to
 *     <body> and the page behind it stops scrolling while it is open.
 *   · "Fit to window" is not enough for a 1024px sheet, so the image toggles
 *     between fit and natural size, scrollable at natural size.
 */
export function Lightbox({
  src,
  title,
  caption,
  onClose,
}: {
  src: string;
  title?: string;
  caption?: string;
  onClose: () => void;
}) {
  const [natural, setNatural] = useState(false);
  const [copied, setCopied] = useState(false);
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    closeRef.current?.focus();
    // The page behind the overlay must not scroll with the wheel: reading a
    // long prompt at natural size scrolls the overlay, not the conversation.
    const before = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = before;
    };
  }, [onClose]);

  return createPortal(
    <div
      className="lightbox"
      role="dialog"
      aria-modal="true"
      aria-label={title || "图片预览"}
      onClick={(e) => {
        // Only the backdrop closes: a click on the picture toggles its zoom, and
        // a click on the prompt is a selection the user is making.
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="lightbox-head">
        <span className="lightbox-title">{title || "图片预览"}</span>
        <div className="lightbox-actions">
          <button className="btn ghost small" onClick={() => setNatural((v) => !v)}>
            {natural ? "适应窗口" : "原始尺寸"}
          </button>
          <a className="btn ghost small" href={src} target="_blank" rel="noreferrer">
            新窗口打开
          </a>
          <button
            className="btn ghost small"
            onClick={onClose}
            ref={closeRef}
            title="Esc 关闭"
          >
            关闭 ✕
          </button>
        </div>
      </div>

      <div className={`lightbox-stage ${natural ? "natural" : "fit"}`}>
        <img
          src={src}
          alt={title || "生成的图片"}
          onClick={() => setNatural((v) => !v)}
          title={natural ? "点击适应窗口" : "点击看原始尺寸"}
        />
      </div>

      {caption && (
        <div className="lightbox-caption">
          <button
            className="btn ghost small"
            onClick={async () => {
              try {
                await navigator.clipboard.writeText(caption);
                setCopied(true);
                window.setTimeout(() => setCopied(false), 1500);
              } catch {
                setCopied(false);
              }
            }}
          >
            {copied ? "已复制" : "复制提示词"}
          </button>
          <div className="lightbox-caption-text mono">{caption}</div>
        </div>
      )}
    </div>,
    document.body,
  );
}

/** A thumbnail that opens the overlay; the picture alone used to be inert. */
export function AssetImage({
  src,
  alt,
  prompt,
  className = "asset-image",
}: {
  src: string;
  alt?: string;
  prompt?: string;
  /** Kept so an older gallery's sizing class is not lost when it gains zoom. */
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const [failed, setFailed] = useState(false);

  if (failed) {
    return (
      <a className="asset-image-fallback" href={src} target="_blank" rel="noreferrer">
        图片未能在这里显示，点击打开原图
      </a>
    );
  }

  return (
    <>
      <button
        type="button"
        className="asset-image-btn"
        onClick={() => setOpen(true)}
        title="点击放大预览"
        aria-label={`放大预览 ${alt || "图片"}`}
      >
        <img
          className={className}
          src={src}
          alt={alt || "生成的图片"}
          loading="lazy"
          onError={() => setFailed(true)}
        />
        <span className="asset-zoom">放大 ⤢</span>
      </button>
      {open && (
        <Lightbox src={src} title={alt} caption={prompt} onClose={() => setOpen(false)} />
      )}
    </>
  );
}
