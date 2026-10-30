"""The pill: frameless, always-on-top, click-through, never takes focus. PySide6 implementation."""
from __future__ import annotations

import ctypes
import socket
import sys
import threading
import time
from typing import Any

from PySide6.QtCore import QObject, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from ev.core.config import Config

from . import protocol
from .anim import VISIBLE_STATES, BarAnimator, demo_level, demo_schedule
from .display import Exchange, tail_that_fits

GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x20
WS_EX_TOOLWINDOW = 0x80
WS_EX_NOACTIVATE = 0x08000000
WS_EX_LAYERED = 0x80000
CARD_GAP = 8
CARD_PAD = 14
CARD_RADIUS = 16
PARA_GAP = 6
USER_MAX_LINES = 3
USER_COLOR = QColor("#9aa0a6")
ACCENT = QColor("#ff9f1c")
AUTO_HIDE_STATES = {"paused": 3.0}


def anchor_position(anchor: str, screen_w: int, screen_h: int, w: int, h: int,
                    ox: int, oy: int) -> tuple[int, int]:
    v, _, hz = anchor.partition("-")
    x = {"left": ox, "center": (screen_w - w) // 2 + ox, "right": screen_w - w - ox}.get(hz, (screen_w - w) // 2 + ox)
    y = oy if v == "top" else screen_h - h - oy if v == "bottom" else (screen_h - h) // 2 + oy
    return x, y


class Bridge(QObject):
    state = Signal(str)
    level = Signal(float)
    text = Signal(str, str, bool)
    display = Signal(bool)
    quit = Signal()


class Pill(QWidget):
    def __init__(self, cfg: Config):
        super().__init__()
        o = cfg["overlay"]
        self.o = o
        self.w_, self.h_ = o["width"], o["height"]
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput | Qt.WindowType.WindowDoesNotAcceptFocus | Qt.WindowType.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.resize(self.w_, self.h_)
        # display mode: a text card under the pill showing what the user said and what EV answers
        self.display = False
        self.convo = Exchange()
        self.card_w = max(self.w_, int(o.get("display_width", 460)))
        self.card_max_h = int(o.get("display_max_height", 300))
        self.linger = float(o.get("display_linger_s", 8))
        self.font_ = QFont("Segoe UI", int(o.get("display_font_pt", 12)))
        self.horizontal = o["anchor"].partition("-")[2] or "center"
        self._user_txt = self._ev_txt = ""
        self._card_h = 0
        self.anim = BarAnimator(o["bar_count"])
        self.fill = QColor(o["fill"])
        self.fill.setAlpha(o["fill_alpha"])
        self.border = QColor(o["border"])
        self.state = "idle"
        self.level = 0.0
        self.opacity = 0.0
        self.target = 0.0
        self.idle_since: float | None = time.monotonic()
        self.state_since = time.monotonic()
        self.t0 = time.monotonic()
        self.last = self.t0
        self.fade_s = max(0.01, o["fade_ms"] / 1000)
        self.hide_delay = o["hide_delay_ms"] / 1000
        self.timer = QTimer(self)
        self.timer.setInterval(int(1000 / o["fps"]))
        self.timer.timeout.connect(self.tick)
        self.setWindowOpacity(0.0)
        self._cfg = cfg
        self._styled = False
        self._place(cfg)

    def _place(self, cfg: Config) -> None:
        o = self.o
        screens = QGuiApplication.screens()
        scr = screens[o["monitor"]] if 0 <= o["monitor"] < len(screens) else QGuiApplication.primaryScreen()
        g = scr.geometry()
        x, y = anchor_position(o["anchor"], g.width(), g.height(), self.width(), self.height(),
                               o["offset_x"], o["offset_y"])
        self.move(g.x() + x, g.y() + y)

    def _apply_native_styles(self) -> None:
        if self._styled or sys.platform != "win32":
            return
        hwnd = int(self.winId())
        u = ctypes.windll.user32
        get = getattr(u, "GetWindowLongPtrW", u.GetWindowLongW)
        setl = getattr(u, "SetWindowLongPtrW", u.SetWindowLongW)
        style = get(hwnd, GWL_EXSTYLE)
        setl(hwnd, GWL_EXSTYLE, style | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | WS_EX_LAYERED)
        self._styled = True

    # ---- display mode
    def set_display(self, on: bool) -> None:
        if on == self.display:
            return
        self.display = on
        if not on:
            self.convo.clear()
        self._relayout()

    def set_text(self, who: str, text: str, append: bool) -> None:
        if not self.display:
            return
        self.convo.apply(who, text, append)
        self._relayout()
        self.idle_since = None if self.state in VISIBLE_STATES else time.monotonic()
        self.target = 1.0
        self._ensure_visible()

    def _measure(self, text: str, width: int) -> int:
        flags = int(Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        return QFontMetrics(self.font_).boundingRect(QRect(0, 0, width, 100000), flags, text).height()

    def _relayout(self) -> None:
        """Fit the text to the card, then size the window to pill + card. The pill never moves."""
        W = self.card_w if self.display else self.w_
        user, ev, card_h = "", "", 0
        if self.display and not self.convo.empty:
            tw = W - 2 * CARD_PAD
            line = QFontMetrics(self.font_).lineSpacing()
            inner = self.card_max_h - self.h_ - CARD_GAP - 2 * CARD_PAD
            user = tail_that_fits(self.convo.user, lambda s: self._measure(s, tw) <= USER_MAX_LINES * line) if self.convo.user else ""
            user_h = self._measure(user, tw) if user else 0
            gap = PARA_GAP if user and self.convo.ev else 0
            ev_room = max(line, inner - user_h - gap)
            ev = tail_that_fits(self.convo.ev, lambda s: self._measure(s, tw) <= ev_room) if self.convo.ev else ""
            card_h = 2 * CARD_PAD + user_h + gap + (self._measure(ev, tw) if ev else 0)
        self._user_txt, self._ev_txt, self._card_h = user, ev, card_h
        H = self.h_ + (CARD_GAP + card_h if card_h else 0)
        if (W, H) != (self.width(), self.height()):
            self.resize(W, H)
            self._place(self._cfg)
        self.update()

    # ---- state/level inputs (called on the GUI thread via signals)
    def set_state(self, s: str) -> None:
        s = "paused" if s == "muted" else s
        if s == self.state:
            return
        self.state = s
        self.anim.state = s
        self.state_since = time.monotonic()
        if s in VISIBLE_STATES:
            self.idle_since = None
            self.target = 1.0
            self._ensure_visible()
        else:
            self.idle_since = time.monotonic()

    def set_level(self, v: float) -> None:
        self.level = max(0.0, min(1.0, v))

    def _ensure_visible(self) -> None:
        if not self.isVisible():
            self.show()
            self._apply_native_styles()
        if not self.timer.isActive():
            self.last = time.monotonic()
            self.timer.start()

    def tick(self) -> None:
        now = time.monotonic()
        dt = now - self.last
        self.last = now
        delay = self.linger if self.display and self._card_h else self.hide_delay   # let the card be read
        if self.idle_since is not None and now - self.idle_since >= delay:
            self.target = 0.0
        limit = AUTO_HIDE_STATES.get(self.state)
        if limit is not None and now - self.state_since >= limit:
            self.target = 0.0
        step = dt / self.fade_s
        if self.opacity < self.target:
            self.opacity = min(self.target, self.opacity + step)
        elif self.opacity > self.target:
            self.opacity = max(self.target, self.opacity - step)
        self.setWindowOpacity(self.opacity)
        self.anim.step(now - self.t0, self.level)
        self.update()
        if self.opacity <= 0.0 and self.target <= 0.0:
            self.timer.stop()   # 0% CPU while hidden
            self.hide()
            if self.display and not self.convo.empty:
                self.convo.clear()
                self._relayout()

    def paintEvent(self, _e: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        W = self.width()
        px = {"left": 0, "right": W - self.w_}.get(self.horizontal, (W - self.w_) // 2)
        p.save()
        p.translate(px, 0)
        rect = QRectF(0.5, 0.5, self.w_ - 1, self.h_ - 1)
        radius = rect.height() / 2
        dim = self.state in ("paused", "muted")
        fill = QColor(self.fill)
        p.setBrush(fill)
        pen = QPen(ACCENT if self.state == "acting" else self.border)
        pen.setWidthF(1.0)
        p.setPen(pen)
        p.drawRoundedRect(rect, radius, radius)

        n = self.anim.n
        bar_w, gap = 3.0, 3.0
        total = n * bar_w + (n - 1) * gap
        x0 = (self.w_ - total) / 2
        max_h = self.h_ - 12
        col = QColor(255, 255, 255, 110 if dim else 255)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(col)
        for i, h in enumerate(self.anim.h):
            bh = max(bar_w, h * max_h)
            x = x0 + i * (bar_w + gap)
            p.drawRoundedRect(QRectF(x, (self.h_ - bh) / 2, bar_w, bh), bar_w / 2, bar_w / 2)
        if self.state == "acting":
            p.setBrush(ACCENT)
            p.drawEllipse(QRectF(10, self.h_ / 2 - 2.5, 5, 5))
        p.restore()
        if self._card_h:
            self._paint_card(p, W)
        p.end()

    def _paint_card(self, p: QPainter, W: int) -> None:
        top = self.h_ + CARD_GAP
        card = QRectF(0.5, top + 0.5, W - 1, self._card_h - 1)
        p.setBrush(self.fill)
        pen = QPen(self.border)
        pen.setWidthF(1.0)
        p.setPen(pen)
        p.drawRoundedRect(card, CARD_RADIUS, CARD_RADIUS)
        p.setFont(self.font_)
        flags = int(Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        tw = W - 2 * CARD_PAD
        y = top + CARD_PAD
        if self._user_txt:
            h = self._measure(self._user_txt, tw)
            p.setPen(USER_COLOR)
            p.drawText(QRect(CARD_PAD, y, tw, h), flags, self._user_txt)
            y += h + PARA_GAP
        if self._ev_txt:
            p.setPen(QColor("#ffffff"))
            p.drawText(QRect(CARD_PAD, y, tw, self._measure(self._ev_txt, tw)), flags, self._ev_txt)


def _drop(conn: socket.socket, f: Any) -> None:
    for c in (f, conn):
        try:
            c.close()
        except OSError:
            pass


def _serve(sock: socket.socket, token: str, bridge: Bridge, stop: threading.Event) -> None:
    """Accept one authenticated core connection. When it drops, the overlay exits."""
    sock.settimeout(1.0)
    started = time.monotonic()
    authed = False
    while not stop.is_set():
        try:
            conn, addr = sock.accept()
        except TimeoutError:
            if not authed and time.monotonic() - started > 60:
                bridge.quit.emit()
                return
            continue
        except OSError:
            return
        if not protocol.is_loopback(addr[0]):
            conn.close()
            continue
        conn.settimeout(protocol.AUTH_TIMEOUT_S)
        f = conn.makefile("rb")
        try:
            first = protocol.decode(f.readline(protocol.MAX_LINE + 1))
        except (OSError, TimeoutError):
            first = None
        if not first or not protocol.token_ok(token, first.get("token")) or authed:
            _drop(conn, f)      # bad/missing token, or a second core: close socket AND its file wrapper
            continue
        authed = True
        conn.settimeout(None)
        try:
            for line in f:
                m = protocol.decode(line)
                if m is None or not protocol.valid_message(m):
                    continue
                if m["t"] == "state":
                    bridge.state.emit(m["v"])
                elif m["t"] == "level":
                    bridge.level.emit(float(m["v"]))
                elif m["t"] == "text":
                    bridge.text.emit(m["who"], m["v"], bool(m.get("append", False)))
                elif m["t"] == "display":
                    bridge.display.emit(bool(m["v"]))
                elif m["t"] == "quit":
                    break
        except OSError:
            pass
        bridge.quit.emit()      # core went away -> overlay exits
        return


class _DemoState:
    def __init__(self, t: float) -> None:
        self.i = 0
        self.t = t


DEMO_LINES = [
    ("user", "What's the weather like in London today?", False),
    ("ev", "It's about twelve degrees and cloudy in London right now.", False),
    ("ev", "Maybe grab a light jacket if you're heading out.", True),
]


def run_demo(cfg: Config, display: bool = False) -> int:
    app = QApplication(sys.argv)
    pill = Pill(cfg)
    if display:
        pill.set_display(True)
    sched = demo_schedule()
    t0 = time.monotonic()
    st = _DemoState(t0)

    def advance() -> None:
        now = time.monotonic()
        name, dur = sched[st.i % len(sched)]
        if now - st.t >= dur:
            st.i += 1
            st.t = now
            name, dur = sched[st.i % len(sched)]
            pill.set_state(name)
            if display and name == "thinking":
                pill.set_text(*DEMO_LINES[0])
            elif display and name == "speaking":
                pill.set_text(*DEMO_LINES[1])
                QTimer.singleShot(2500, lambda: pill.set_text(*DEMO_LINES[2]))
        pill.set_level(demo_level(name, now - t0))

    pill.set_state(sched[0][0])
    t = QTimer()
    t.setInterval(1000 // 60)
    t.timeout.connect(advance)
    t.start()
    return app.exec()


def run_server(cfg: Config) -> int:
    token = sys.stdin.readline().strip()
    if not token:
        return 2
    sock = protocol.bind_loopback()
    print(sock.getsockname()[1], flush=True)
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    pill = Pill(cfg)
    bridge = Bridge()
    bridge.state.connect(pill.set_state)
    bridge.level.connect(pill.set_level)
    bridge.text.connect(pill.set_text)
    bridge.display.connect(pill.set_display)
    bridge.quit.connect(app.quit)
    stop = threading.Event()
    threading.Thread(target=_serve, args=(sock, token, bridge, stop), daemon=True).start()
    code = app.exec()
    stop.set()
    sock.close()
    return code
