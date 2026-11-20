"""Configuration for 漫簇.

All values can be overridden by environment variables so the same code runs
locally and in packaged deployments without edits.

No secrets are hard-coded: the LLM key is read from the environment, and every
endpoint reports a clear "not configured" state instead of failing silently.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

#: The user's own content: the run database, uploaded images, generated assets and
#: films. Separated from configuration on purpose — this directory is the one a
#: user backs up or deletes, and it must never be confused with anything that
#: identifies the machine or holds a credential.
DATA_DIR = Path(os.environ.get("STUDIO_DATA_DIR", BASE_DIR / "userdata"))

#: Machine configuration, including `providers.json` (which holds API keys).
#: Kept out of DATA_DIR so a backup of the user's work cannot accidentally carry
#: their credentials with it.
CONFIG_DIR = Path(os.environ.get("STUDIO_CONFIG_DIR", BASE_DIR / "config"))

UPLOAD_DIR = DATA_DIR / "uploads"

#: Generated media, one directory per run: `<run_id>/<kind>/<file>` where kind is
#: `portraits` (reference sheets) or `shots` (rendered clips). Kept under DATA_DIR
#: with the rest of the user's work, and served to the browser by
#: `/api/asset/{run_id}/{kind}/{name}` — the renderer's own download link carries
#: an API key and answers 401 to a plain <img>, which is why the UI showed empty
#: frames for images that had in fact been produced.
ASSET_DIR = DATA_DIR / "assets"
DB_PATH = Path(os.environ.get("STUDIO_DB_PATH", DATA_DIR / "studio.db"))
PROVIDERS_PATH = Path(
    os.environ.get("STUDIO_PROVIDERS_PATH", CONFIG_DIR / "providers.json")
)


def _env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class LLMConfig:
    """OpenAI-compatible endpoint. Works with DeepSeek, Qwen, Atlas, Ollama, OpenAI."""

    base_url: str = field(default_factory=lambda: _env_str("STUDIO_LLM_BASE_URL", "https://api.deepseek.com/v1"))
    api_key: str = field(default_factory=lambda: _env_str("STUDIO_LLM_API_KEY"))
    model: str = field(default_factory=lambda: _env_str("STUDIO_LLM_MODEL", "deepseek-chat"))
    # Vision model for image understanding; falls back to `model` when unset.
    vision_model: str = field(default_factory=lambda: _env_str("STUDIO_LLM_VISION_MODEL"))
    timeout_seconds: float = field(default_factory=lambda: _env_float("STUDIO_LLM_TIMEOUT", 60.0))
    temperature: float = field(default_factory=lambda: _env_float("STUDIO_LLM_TEMPERATURE", 0.4))

    @property
    def configured(self) -> bool:
        """True when a key is present. Local keyless endpoints (Ollama) may leave it blank.

        Presence only — it does not mean the key works. A placeholder such as
        `t-key` passes this check, which is why the run failure must not say
        "not configured": the key IS configured, it is just not real.
        """
        return bool(self.api_key) or "127.0.0.1" in self.base_url or "localhost" in self.base_url

    def missing_reason(self) -> str:
        """Why this config cannot be used, in terms that point at the actual fix.

        Two distinct situations need two distinct messages:

          · no key at all          → the user never entered one
          · a placeholder-shaped key → a value is stored but it is scratch data

        Saying "尚未配置" to the second case is what sent the user looking for a
        settings page they had already filled in, while the chip beside it said
        "已配置". Both messages name the place to fix, because "set STUDIO_LLM_API_KEY"
        pointed at an environment variable that a UI-configured user has never used.
        """
        if not self.configured:
            return (
                "尚未配置大模型：请在「设置 → 供应商与 Key」里填入 API Key 并保存"
                "（也可以设置环境变量 STUDIO_LLM_API_KEY）"
            )

        from .models_config import KEY_PLACEHOLDER, KEY_SUSPICIOUS, key_verdict

        verdict = key_verdict(self.api_key)
        if verdict == KEY_PLACEHOLDER:
            return (
                f"大模型 Key 是占位值（{self.api_key!r}），不是真实密钥。"
                "请在「设置 → 供应商与 Key」里粘贴真实 Key 后保存。"
            )
        if verdict == KEY_SUSPICIOUS:
            return (
                f"大模型 Key 过短（{self.api_key!r}），明显不是有效密钥。"
                "请在「设置 → 供应商与 Key」里核对后重新填入。"
            )
        return ""

    @property
    def effective_vision_model(self) -> str:
        return self.vision_model or self.model


@dataclass
class MancuMediaConfig:
    """Connection settings for the optional media production backend."""

    base_url: str = field(default_factory=lambda: _env_str("MANCU_MEDIA_BASE_URL", "http://127.0.0.1:3300"))
    timeout_seconds: float = field(default_factory=lambda: _env_float("MANCU_MEDIA_TIMEOUT", 180.0))
    # Provider used for image generation; must match the configured provider name.
    image_provider: str = field(default_factory=lambda: _env_str("STUDIO_IMAGE_PROVIDER", "atlas-cloud"))
    image_api_key: str = field(default_factory=lambda: _env_str("STUDIO_IMAGE_API_KEY"))
    image_base_url: str = field(default_factory=lambda: _env_str("STUDIO_IMAGE_BASE_URL"))
    image_model: str = field(default_factory=lambda: _env_str("STUDIO_IMAGE_MODEL"))


@dataclass
class BudgetConfig:
    """Per-run spend guard checked before paid calls."""

    max_usd_per_run: float = field(default_factory=lambda: _env_float("STUDIO_MAX_USD_PER_RUN", 2.0))
    max_steps_per_run: int = 40


@dataclass
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    mancu_media: MancuMediaConfig = field(default_factory=MancuMediaConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    # Agent revision loop: how many times the reviewer may send work back.
    max_revision_rounds: int = 2


settings = Settings()


def ensure_dirs() -> None:
    for path in (DATA_DIR, UPLOAD_DIR, ASSET_DIR, DB_PATH.parent, CONFIG_DIR):
        path.mkdir(parents=True, exist_ok=True)
