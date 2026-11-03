"""Reading and writing config/ev.local.toml, the only file the dashboard ever writes settings to (never ev.toml, never policy.toml).

The file is plain TOML that the user may also edit. Writing it again keeps every key it already had and rewrites it in a
normal layout, so *comments in that file are not kept*; the first time, a copy goes to ev.local.toml.bak.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import tomllib
from pathlib import Path
from typing import Any

_BARE = re.compile(r"[A-Za-z0-9_-]+")


def read(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}


def _key(k: str) -> str:
    return k if _BARE.fullmatch(k) else json.dumps(k)


def _value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError("TOML can't hold NaN or infinity here")
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)                                     # a JSON string is a valid TOML basic string
    if isinstance(v, list):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    raise TypeError(f"can't write {type(v).__name__} to TOML")


def dumps(data: dict[str, Any]) -> str:
    out: list[str] = []

    def table(prefix: str, d: dict[str, Any]) -> None:
        scalars = {k: v for k, v in d.items() if not isinstance(v, dict)}
        if scalars or not any(isinstance(v, dict) for v in d.values()):
            if prefix:
                out.append(f"[{prefix}]")
            out.extend(f"{_key(k)} = {_value(v)}" for k, v in scalars.items())
            out.append("")
        for k, v in d.items():
            if isinstance(v, dict):
                table(f"{prefix}.{_key(k)}" if prefix else _key(k), v)

    table("", data)
    return "\n".join(out).strip() + "\n"


def update(path: Path, changes: dict[str, Any]) -> None:
    """Merge {"section.name": value} into the file and write it back atomically. Other keys are untouched."""
    data = read(path)
    for key, value in changes.items():
        section, name = key.split(".", 1)
        data.setdefault(section, {})[name] = value
    if path.exists() and not path.with_name(path.name + ".bak").exists():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    tmp = path.with_name(path.name + ".evtmp")
    tmp.write_text("# Per-machine overrides. EV's dashboard writes here; it never touches ev.toml or policy.toml.\n" + dumps(data),
                   encoding="utf-8", newline="\n")
    os.replace(tmp, path)
