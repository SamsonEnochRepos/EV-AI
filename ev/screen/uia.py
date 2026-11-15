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


def window_for(hwnd: int):  # noqa: ANN201
    from pywinauto import Desktop

    return Desktop(backend="uia").window(handle=hwnd)


def terminal_text(hwnd: int, limit: int = 20000) -> str:
    """The text showing in a Windows Terminal window (its TermControl exposes it through UIA's TextPattern), or "" when it can't
    be read. Read-only; the caller decides what the words mean."""
    try:
        for ctl in window_for(hwnd).descendants(class_name="TermControl"):
            return str(ctl.iface_text.DocumentRange.GetText(limit))
    except Exception as e:  # noqa: BLE001  (no UIA, a closed window or a terminal that hides its text: nothing to read)
        log.info("could not read terminal text: %s", e)
    return ""


def browser_tabs(win: Any) -> list[UIElement]:
    """The tabs along the top of a browser window, left to right. A browser also reports tab lists that belong to the page it
    is showing (a web app's own tabs), so only the items in the same row as the first one count."""
    try:
        items = win.descendants(control_type="TabItem")
    except Exception as e:
        log.warning("UIA tab list failed: %s", e)
        return []
    out: list[UIElement] = []
    row: tuple[int, int] | None = None
    for d in items:
        try:
            info = d.element_info
            r = info.rectangle
            if r.width() <= 0 or r.height() <= 0:
                continue
            if row is None:
                row = (r.top, r.bottom)
            elif (r.top, r.bottom) != row:
                continue
            out.append(UIElement(str(info.name or "").strip(), "TabItem", (r.left, r.top, r.right, r.bottom), False, d))
        except Exception:  # noqa: S112
            continue
    return out


def find_tab(tabs: list[UIElement], name: str) -> UIElement | None:
    """The tab the user means by a few words of its title: all the words must appear, the shortest title wins ("youtube" picks
    the tab called YouTube over one that merely mentions it in a long page title)."""
    words = [w for w in name.lower().split() if w]
    if not words:
        return None
    hits = [t for t in tabs if all(w in t.name.lower() for w in words)]
    return min(hits, key=lambda t: len(t.name)) if hits else None


def select_tab(tab: UIElement) -> bool:
    """Switch to a tab through its own "select" action; a click at its position is only the fallback."""
    try:
        tab.wrapper.iface_selection_item.Select()
        return True
    except Exception:
        return False


def find_element(items: list[UIElement], name: str) -> UIElement | None:
    n = name.lower().strip()
    exact = [e for e in items if e.name.lower() == n]
    if exact:
        return exact[0]
    part = [e for e in items if n in e.name.lower() and len(e.name) <= max(48, 4 * len(n))]    # a control, not a paragraph that mentions it
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
