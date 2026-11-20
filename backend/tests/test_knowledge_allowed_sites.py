from __future__ import annotations

import json

import pytest

from app.kb import authority, websearch


def _use_temp_config(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(websearch, "config_path", lambda: tmp_path / "knowledge.json")


def test_defaults_are_used_before_user_saves_allowlist(monkeypatch, tmp_path) -> None:
    _use_temp_config(monkeypatch, tmp_path)

    assert authority.is_authoritative("https://cpc.people.com.cn/a.html")
    assert not authority.is_authoritative("https://example.com/a.html")


def test_saved_allowlist_replaces_defaults_without_restart(monkeypatch, tmp_path) -> None:
    _use_temp_config(monkeypatch, tmp_path)
    sites = authority.normalize_sites(
        [{"domain": "https://docs.example.org/path", "label": "示例机构", "tier": 2}]
    )

    websearch.save_config("bocha", "secret", allowed_sites=sites)

    assert authority.is_authoritative("https://sub.docs.example.org/article")
    assert authority.site_label("https://docs.example.org/article") == "示例机构"
    assert not authority.is_authoritative("https://people.com.cn/article")
    saved = json.loads((tmp_path / "knowledge.json").read_text(encoding="utf-8"))
    assert saved["allowed_sites"] == [
        {"domain": "docs.example.org", "label": "示例机构", "tier": 2}
    ]


def test_empty_saved_allowlist_rejects_every_site(monkeypatch, tmp_path) -> None:
    _use_temp_config(monkeypatch, tmp_path)

    websearch.save_config("bocha", "", allowed_sites=[])

    assert authority.allowed_sites() == ()
    assert not authority.is_authoritative("https://www.gov.cn/")


@pytest.mark.parametrize(
    "value",
    ["", "localhost", "https://not a domain.example/path", "https://example"],
)
def test_invalid_source_domain_is_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        authority.normalize_domain(value)
