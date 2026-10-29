"""Config loading. All tunables come from config/ev.toml; nothing is hard-coded elsewhere."""
from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


class Config:
    """Thin dotted-access wrapper over the parsed TOML dict."""

    def __init__(self, data: dict[str, Any], root: Path = ROOT):
        self._data = data
        self.root = root

    def __getitem__(self, section: str) -> dict[str, Any]:
        return self._data[section]

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self._data.get(section, {}).get(key, default)

    def path(self, section: str, key: str) -> Path:
        p = Path(self._data[section][key])
        return p if p.is_absolute() else self.root / p

    def data_path(self, name: str) -> Path:
        d = self.path("general", "data_dir")
        d.mkdir(parents=True, exist_ok=True)
        return d / name

    @property
    def raw(self) -> dict[str, Any]:
        return self._data


def load_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    path = path or Path(os.environ.get("EV_CONFIG", ROOT / "config" / "ev.toml"))
    data = load_toml(path)
    local = path.with_name("ev.local.toml")  # optional untracked per-machine overrides
    if local.exists():
        data = _deep_merge(data, load_toml(local))
    if overrides:
        data = _deep_merge(data, overrides)
    return Config(data)


@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()
