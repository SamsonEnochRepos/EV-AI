"""The island: a frameless, always-on-top, click-through "dynamic island" at the top of the screen that never takes focus.

A small black pill with a row of level bars: it springs open while EV listens or talks and the bars follow the voice, and it
half-opens with a slow wave while EV thinks. It fades away when EV is idle (overlay.idle_visible turns the resting pill on). In
display mode the same body grows downward to hold the conversation text. PySide6 implementation; the animation maths lives in anim.py.
"""
from __future__ import annotations

import ctypes
import math
import socket
import sys
import threading
import time
from typing import Any

from PySide6.QtCore import QObject, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QGuiApplication, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QApplication, QWidget

from ev.core.config import Config

from . import protocol
from .anim import (
    VISIBLE_STATES,
    BarAnimator,
    IslandSizes,
    Spring,
    alert_target,
    approach,
    body_origin,
    clamp,
    corner_radius,
    demo_level,
    demo_schedule,
    expansion,
    island_shape,
    lerp_color,
    place_window,
)
from .display import Exchange, tail_that_fits

GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x20
WS_EX_TOOLWINDOW = 0x80
WS_EX_NOACTIVATE = 0x08000000
WS_EX_LAYERED = 0x80000
MARGIN = 12                 # transparent room around the body, for the glow and the spring's overshoot
TEXT_TOP = 4                # gap between the bar strip and the text
TEXT_BOTTOM = 14
TEXT_SIDE = 16
PARA_GAP = 6
USER_MAX_LINES = 3
USER_COLOR = QColor("#9aa0a6")
AUTO_HIDE_STATES = {"paused": 3.0}      # only when the island is not set to stay visible
ERROR_S = 2.5
FAST_MS = 16
IDLE_MS = 33


def _rgb(c: QColor) -> tuple[int, int, int]:
    return c.red(), c.green(), c.blue()


class Bridge(QObject):
    state = Signal(str)
    level = Signal(float)
    text = Signal(str, str, bool)
    display = Signal(bool)
    notice = Signal(str)
    quit = Signal()


class Pill(QWidget):
    """The island window. (Still called Pill: it is what EV's client calls the overlay.)"""

    def __init__(self, cfg: Config):
        super().__init__()
        o = cfg["overlay"]
        self.o = o
        self.w_, self.h_ = o["width"], o["height"]                 # the resting pill
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput | Qt.WindowType.WindowDoesNotAcceptFocus | Qt.WindowType.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # display mode: the island grows to show what the user said and what EV answers
        self.display = False
        self._notice = False                 # a one-off card (a proactive line) shown even when display mode is off
        self._notice_until = 0.0
        self.convo = Exchange()
        wide_w = float(o.get("wide_width", 236))
        wide_h = float(o.get("wide_height", 42))
        self.card_w = max(int(o.get("display_width", 460)), int(wide_w))
        self.card_max_h = max(int(o.get("display_max_height", 300)), int(wide_h))
        self.sizes = IslandSizes(self.w_, self.h_, wide_w, wide_h, self.card_w, wide_h)
        self.linger = float(o.get("display_linger_s", 8))
        self.font_ = QFont("Segoe UI", int(o.get("display_font_pt", 12)))
        self.horizontal = o["anchor"].partition("-")[2] or "center"
        self.vertical = o["anchor"].partition("-")[0] or "top"
        self.keep_visible = bool(o.get("idle_visible", True))
        self._user_txt = self._ev_txt = ""
        self._card_h = 0
        self.anim = BarAnimator(o["bar_count"])
        self.alert = 0.0                     # 0 = bar colour, 1 = accent colour (working on the screen, or an error)
        self.fill = QColor(o["fill"])
        self.fill.setAlpha(o["fill_alpha"])
        self.border = QColor(o["border"])
        self.bar_rgb = _rgb(QColor(o.get("bar_color", "#ffffff")))
        self.accent = QColor(o.get("accent", "#ff2d3f"))
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
        self.win_w = self.card_w + 2 * MARGIN
        self.win_h = self.card_max_h + 2 * MARGIN
        self.resize(self.win_w, self.win_h)
        self.spring_w = Spring(self.w_)
        self.spring_h = Spring(self.h_)
        self._sig: tuple[Any, ...] = ()
        self.timer = QTimer(self)
        self.timer.setInterval(IDLE_MS)
        self.timer.timeout.connect(self.tick)
        self.setWindowOpacity(0.0)
        self._cfg = cfg
        self._styled = False
        self._place(cfg)

    # ---- geometry
    @property
    def cur_w(self) -> float:
        return max(self.w_ * 0.8, self.spring_w.x)

    @property
    def cur_h(self) -> float:
        return max(self.h_ * 0.8, self.spring_h.x)

    def target_shape(self) -> tuple[float, float]:
        return island_shape(self.state, self.sizes, self._card_h)

    def body_rect(self) -> QRectF:
        w, h = self.cur_w, self.cur_h
        x, y = body_origin(self.horizontal, self.vertical, self.win_w, self.win_h, w, h, MARGIN)
        return QRectF(x, y, w, h)

    def _place(self, cfg: Config) -> None:
        o = self.o
        screens = QGuiApplication.screens()
        scr = screens[o["monitor"]] if 0 <= o["monitor"] < len(screens) else QGuiApplication.primaryScreen()
        g = scr.geometry()
        x, y = place_window(o["anchor"], g.width(), g.height(), self.win_w, self.win_h, o["offset_x"], o["offset_y"], MARGIN)
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

    def start(self) -> None:
        """Show the resting island straight away (unless it is set to appear only when EV is busy)."""
        if self.keep_visible:
            self.target = 1.0
            self.idle_since = None
            self._ensure_visible()

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
        self._notice = False                 # a real conversation line takes the card over from any notice
        self.convo.apply(who, text, append)
        self._relayout()
        self.idle_since = None if self.state in VISIBLE_STATES else time.monotonic()
        self.target = 1.0
        self._ensure_visible()

    def set_notice(self, text: str) -> None:
        """Show a short line for a few seconds, whether or not display mode is on (EV's own proactive notices)."""
        self._notice = True
        self._notice_until = time.monotonic() + self.linger + 4
        self.convo.apply("ev", text, False)
        self._relayout()
        self.idle_since = None if self.state in VISIBLE_STATES else time.monotonic()
        self.target = 1.0
        self._ensure_visible()

    def _end_notice(self) -> None:
        self._notice = False
        self.convo.clear()
        self._relayout()

    def _measure(self, text: str, width: int) -> int:
        flags = int(Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        return QFontMetrics(self.font_).boundingRect(QRect(0, 0, width, 100000), flags, text).height()

    def _relayout(self) -> None:
        """Fit the text under the bar strip. The window never resizes; the body springs to the new height."""
        cards = self.display or self._notice
        user, ev, card_h = "", "", 0
        if cards and not self.convo.empty:
            tw = self.card_w - 2 * TEXT_SIDE
            line = QFontMetrics(self.font_).lineSpacing()
            inner = self.card_max_h - self.sizes.strip_h - TEXT_TOP - TEXT_BOTTOM
            user = tail_that_fits(self.convo.user, lambda s: self._measure(s, tw) <= USER_MAX_LINES * line) if self.convo.user else ""
            user_h = self._measure(user, tw) if user else 0
            gap = PARA_GAP if user and self.convo.ev else 0
            ev_room = max(line, inner - user_h - gap)
            ev = tail_that_fits(self.convo.ev, lambda s: self._measure(s, tw) <= ev_room) if self.convo.ev else ""
            card_h = TEXT_TOP + user_h + gap + (self._measure(ev, tw) if ev else 0) + TEXT_BOTTOM
        self._user_txt, self._ev_txt, self._card_h = user, ev, card_h
        self._kick()

    # ---- state/level inputs (called on the GUI thread via signals)
    def set_state(self, s: str) -> None:
        s = "paused" if s == "muted" else s
        if s == self.state:
            return
        now = time.monotonic()
        self.state = s
        self.anim.state = s
        self.state_since = now
        if s in VISIBLE_STATES:
            self.idle_since = None
            self.target = 1.0
            self._ensure_visible()
        else:
            self.idle_since = now
        self._kick()

    def set_level(self, v: float) -> None:
        self.level = clamp(v)

    def _kick(self) -> None:
        """Something changed: animate at full rate until it settles."""
        if self.timer.isActive():
            self.timer.setInterval(FAST_MS)

    def _ensure_visible(self) -> None:
        if not self.isVisible():
            self.show()
            self._apply_native_styles()
        if not self.timer.isActive():
            self.last = time.monotonic()
            self.timer.start()
        self._kick()

    # ---- frame
    def tick(self) -> None:
        now = time.monotonic()
        dt = min(0.1, now - self.last)
        self.last = now
        if self._notice and now >= self._notice_until:
            self._end_notice()
        if self.state == "error" and now - self.state_since >= ERROR_S:
            self.set_state("idle")
        # finished text lingers so it can be read, then folds back into the island
        if (self.display and not self._notice and not self.convo.empty and self.idle_since is not None
                and now - self.idle_since >= self.linger):
            self.convo.clear()
            self._relayout()
        if self.keep_visible:
            self.target = 1.0
        else:
            delay = self.linger if (self.display or self._notice) and self._card_h else self.hide_delay
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
        tw, th = self.target_shape()
        self.spring_w.step(tw, dt)
        self.spring_h.step(th, dt)
        self.alert = approach(self.alert, alert_target(self.state), dt, 8.0)
        self.anim.step(now - self.t0, self.level)
        sig = (round(self.cur_w, 1), round(self.cur_h, 1), round(self.alert, 2), round(self.opacity, 2),
               tuple(round(b, 2) for b in self.anim.h), self._user_txt, self._ev_txt, self.state)
        if sig != self._sig:
            self._sig = sig
            self.setWindowOpacity(self.opacity)
            self.update()
        busy = not (self.spring_w.settled(tw) and self.spring_h.settled(th)) or self.state != "idle" or self.opacity != self.target
        self.timer.setInterval(FAST_MS if busy else IDLE_MS)
        if self.opacity <= 0.0 and self.target <= 0.0:
            self.timer.stop()   # 0% CPU while hidden
            self.hide()
            if self._notice:
                self._end_notice()
            if self.display and not self.convo.empty:
                self.convo.clear()
                self._relayout()

    def settle(self, state: str, level: float = 0.0, steps: int = 90) -> None:
        """Jump straight to a state's resting look (tests and the render eval use this; no timer, no waiting)."""
        self.state = state
        self.anim.state = state
        self.level = level
        self.opacity = 1.0
        self.setWindowOpacity(1.0)
        tw, th = self.target_shape()
        self.spring_w.x, self.spring_h.x = tw, th
        self.spring_w.v = self.spring_h.v = 0.0
        self.alert = alert_target(state)
        for i in range(steps):
            self.anim.step(i / 30, level)

    # ---- painting
    def paintEvent(self, _e: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.body_rect()
        s = self.sizes
        radius = corner_radius(rect.width(), rect.height(), s)
        grow = expansion(rect.width(), s)
        if self.alert > 0.02:                                    # a soft ring of the accent colour: EV is working on the screen
            pulse = 0.65 + 0.35 * (0.5 + 0.5 * math.sin((time.monotonic() - self.t0) * 5.0))
            for i, a in enumerate((70, 38, 16)):
                glow = QColor(self.accent)
                glow.setAlphaF(clamp(a / 255 * self.alert * pulse))
                pen = QPen(glow)
                pen.setWidthF(2.0)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                g = rect.adjusted(-(i + 1) * 2.2, -(i + 1) * 2.2, (i + 1) * 2.2, (i + 1) * 2.2)
                p.drawRoundedRect(g, radius + (i + 1) * 2.2, radius + (i + 1) * 2.2)
        body = QPainterPath()
        body.addRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)
        edge = QColor(*lerp_color(_rgb(self.border), _rgb(self.accent), self.alert * 0.8))
        pen = QPen(edge)
        pen.setWidthF(1.0)
        p.setPen(pen)
        p.setBrush(self.fill)
        p.drawPath(body)
        p.setClipPath(body)
        strip = min(rect.height(), s.strip_h) if self._card_h else rect.height()
        cy = rect.top() + strip / 2
        dim = 0.45 if self.state in ("paused", "muted") else 1.0
        self._paint_bars(p, rect, cy, strip, grow, lerp_color(self.bar_rgb, _rgb(self.accent), self.alert), dim)
        if self._card_h:
            self._paint_text(p, rect)
        p.end()

    def _paint_bars(self, p: QPainter, rect: QRectF, cy: float, strip: float, grow: float, rgb: tuple[int, int, int], dim: float) -> None:
        """One row of level bars, centred in the strip; they fade in as the island opens and are tallest in the middle."""
        if grow < 0.05:
            return
        hs = self.anim.h
        n = len(hs)
        bw, gap = 3.5, 3.0
        max_h = max(6.0, strip - 14)
        col = QColor(*rgb)
        col.setAlphaF(clamp(grow * dim))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(col)
        x0 = rect.center().x() - (n * bw + (n - 1) * gap) / 2
        for i, h in enumerate(hs):
            bh = max(bw, h * max_h)
            p.drawRoundedRect(QRectF(x0 + i * (bw + gap), cy - bh / 2, bw, bh), bw / 2, bw / 2)

    def _paint_text(self, p: QPainter, rect: QRectF) -> None:
        shown = clamp((rect.height() - self.sizes.strip_h - 6) / 30)       # text fades in as the body opens under the bars
        if shown <= 0.01:
            return
        p.setFont(self.font_)
        flags = int(Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        tw = int(self.card_w - 2 * TEXT_SIDE)
        x = int(rect.left() + (rect.width() - self.card_w) / 2 + TEXT_SIDE)
        y = int(rect.top() + self.sizes.strip_h + TEXT_TOP)
        if self._user_txt:
            h = self._measure(self._user_txt, tw)
            c = QColor(USER_COLOR)
            c.setAlphaF(shown)
            p.setPen(c)
            p.drawText(QRect(x, y, tw, h), flags, self._user_txt)
            y += h + PARA_GAP
        if self._ev_txt:
            c = QColor("#ffffff")
            c.setAlphaF(shown)
            p.setPen(c)
            p.drawText(QRect(x, y, tw, self._measure(self._ev_txt, tw)), flags, self._ev_txt)


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
                elif m["t"] == "notice":
                    bridge.notice.emit(m["v"])
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

    pill.start()
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
    bridge.notice.connect(pill.set_notice)
    bridge.quit.connect(app.quit)
    pill.start()
    stop = threading.Event()
    threading.Thread(target=_serve, args=(sock, token, bridge, stop), daemon=True).start()
    code = app.exec()
    stop.set()
    sock.close()
    return code
