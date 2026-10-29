"""Windows UI Automation reader (pywinauto, UIA backend): the cheapest, most reliable layer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ev.core.log import get_logger

log = get_logger("uia")

MAX_ELEMENTS = 200
CLICKABLE = {"Button", "MenuItem", "TabItem", "Hyperlink", "ListItem", "CheckBox", "RadioButton",
             "TreeItem", "SplitButton", "ComboBox", "Edit", "Document", "Text"}


@dataclass
class UIElement:
    name: str
    control_type: str
    rect: tuple[int, int, int, int]      # left, top, right, bottom (screen px)
    is_password: bool = False
    wrapper: Any = None                  # pywinauto wrapper (not serialised)

    @property
    def center(self) -> tuple[int, int]:
        return (self.rect[0] + self.rect[2]) // 2, (self.rect[1] + self.rect[3]) // 2


def _desktop():  # noqa: ANN202
    from pywinauto import Desktop

    return Desktop(backend="uia")


def foreground_window():  # noqa: ANN202
    import win32gui  # type: ignore[import-untyped]
    from pywinauto import Desktop

    hwnd = win32gui.GetForegroundWindow()
    return Desktop(backend="uia").window(handle=hwnd) if hwnd else None


def window_title(win: Any) -> str:
    try:
        return str(win.window_text())
    except Exception:
        return ""


def list_windows() -> list[str]:
    try:
        return [w.window_text() for w in _desktop().windows() if w.window_text().strip()]
    except Exception:
        return []


def find_window(title: str) -> Any | None:
    t = title.lower()
    try:
        cands = [w for w in _desktop().windows() if t in w.window_text().lower()]
    except Exception:
        return None
    return cands[0] if cands else None


def elements(win: Any, limit: int = MAX_ELEMENTS) -> list[UIElement]:
    out: list[UIElement] = []
    try:
        descendants = win.descendants()
    except Exception as e:
        log.warning("UIA descendants failed: %s", e)
        return out
    for d in descendants:
        if len(out) >= limit:
            break
        try:
            info = d.element_info
            ct = str(info.control_type)
            name = str(info.name or "").strip()
            if ct not in CLICKABLE or (not name and ct not in ("Edit", "Document")):
                continue
            r = info.rectangle
            if r.width() <= 0 or r.height() <= 0:
                continue
            try:
                pw = bool(info.element.CurrentIsPassword)
            except Exception:
                pw = False
            out.append(UIElement(name, ct, (r.left, r.top, r.right, r.bottom), pw, d))
        except Exception:  # noqa: S112
            continue
    return out


def find_element(items: list[UIElement], name: str) -> UIElement | None:
    n = name.lower().strip()
    exact = [e for e in items if e.name.lower() == n]
    if exact:
        return exact[0]
    part = [e for e in items if n in e.name.lower()]
    return min(part, key=lambda e: len(e.name)) if part else None


def summarize(win: Any, items: list[UIElement], max_items: int = 60) -> str:
    lines = [f"Window: {window_title(win)}"]
    for e in items[:max_items]:
        if e.control_type in ("Document", "Edit") and e.wrapper is not None:
            try:
                val = e.wrapper.get_value() if e.control_type == "Edit" else e.wrapper.window_text()
            except Exception:
                val = ""
            if e.is_password:
                val = "<hidden>"
            lines.append(f"{e.control_type} '{e.name}': {str(val)[:300]}")
        else:
            lines.append(f"{e.control_type}: {e.name}")
    return "\n".join(lines)
