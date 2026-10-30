"""Display mode: the text card under the pill. Model, wire protocol, client behaviour and card layout."""
from __future__ import annotations

import os

import pytest

from ev.core.config import load_config
from ev.overlay import protocol
from ev.overlay.client import OverlayClient
from ev.overlay.display import ELLIPSIS, Exchange, tail_that_fits


# ---- model
def test_user_line_starts_a_new_exchange_and_ev_lines_build_up():
    x = Exchange()
    x.apply("user", "what's the weather")
    x.apply("ev", "It's cloudy.", append=True)
    x.apply("ev", "Bring a jacket.", append=True)
    assert (x.user, x.ev) == ("what's the weather", "It's cloudy. Bring a jacket.")
    x.apply("user", "thanks")
    assert (x.user, x.ev) == ("thanks", "")


def test_ev_line_without_append_replaces_and_whitespace_is_tidied():
    x = Exchange(ev="old")
    x.apply("ev", "  new\n  line ")
    assert x.ev == "new line"
    x.clear()
    assert x.empty


def test_overflowing_text_keeps_the_newest_words():
    fits = lambda s: len(s) <= 30  # noqa: E731
    out = tail_that_fits("one two three four five six seven eight nine ten", fits)
    assert out.startswith(ELLIPSIS) and out.endswith("nine ten") and fits(out)
    assert tail_that_fits("short text", fits) == "short text"
    assert tail_that_fits("x" * 80, fits).endswith("x" * 80)        # one huge word is still shown, never dropped


# ---- wire protocol
@pytest.mark.parametrize("msg,ok", [
    ({"t": "display", "v": True}, True),
    ({"t": "display", "v": "yes"}, False),
    ({"t": "text", "who": "user", "v": "hello"}, True),
    ({"t": "text", "who": "ev", "v": "hi", "append": True}, True),
    ({"t": "text", "who": "robot", "v": "hi"}, False),
    ({"t": "text", "who": "ev", "v": ""}, False),
    ({"t": "text", "who": "ev", "v": "x" * (protocol.MAX_TEXT + 1)}, False),
    ({"t": "text", "who": "ev", "v": "hi", "append": "yes"}, False),
    ({"t": "text", "who": "ev"}, False),
])
def test_protocol_validates_display_messages(msg, ok):
    assert protocol.valid_message(msg) is ok


def test_curly_quotes_and_non_english_text_stay_under_the_line_limit():
    msg = {"t": "text", "who": "ev", "v": "I’m happy — नमस्ते " * 20, "append": True}
    msg["v"] = msg["v"][: protocol.MAX_TEXT]
    line = protocol.encode(msg)
    assert len(line) <= protocol.MAX_LINE and protocol.decode(line) == msg


# ---- client
class FakeSock:
    def __init__(self):
        self.sent = []

    def sendall(self, b):
        self.sent.append(protocol.decode(b))

    def close(self):
        pass


@pytest.fixture
def client():
    c = OverlayClient(load_config())
    c.sock = FakeSock()                     # type: ignore[assignment]
    return c


def test_text_is_only_sent_in_display_mode(client):
    client.send_text("user", "hello")
    assert client.sock.sent == []
    client.send_display(True)
    client.send_text("user", "hello")
    client.send_text("ev", "  hi \n there ", append=True)
    assert client.sock.sent == [
        {"t": "display", "v": True}, {"t": "text", "who": "user", "v": "hello", "append": False},
        {"t": "text", "who": "ev", "v": "hi there", "append": True}]
    client.send_display(False)
    client.send_text("user", "ignored")
    assert client.sock.sent[-1] == {"t": "display", "v": False} and len(client.sock.sent) == 4


def test_long_sentences_are_trimmed_to_the_protocol_limit(client):
    client.send_display(True)
    client.send_text("ev", "word " * 400)
    sent = client.sock.sent[-1]["v"]
    assert len(sent) <= protocol.MAX_TEXT and sent.endswith("…")


def test_display_mode_survives_an_overlay_restart(client):
    client.send_display(True)
    assert client._display is True          # _start_locked replays this to a freshly spawned overlay


# ---- card layout (Qt, offscreen: nothing is shown on screen)
@pytest.fixture
def pill():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from ev.overlay.qt_app import Pill

    QApplication.instance() or QApplication([])
    p = Pill(load_config())
    p._ensure_visible = lambda: None        # type: ignore[method-assign]
    return p


def test_card_appears_grows_with_text_and_goes_away(pill):
    pill.set_text("user", "ignored while display mode is off", False)
    assert (pill.width(), pill.height(), pill._card_h) == (pill.w_, pill.h_, 0)

    pill.set_display(True)
    assert pill.width() == pill.card_w and pill._card_h == 0         # wide window, but no card until there is text
    pill.set_text("user", "What's the weather in London?", False)
    pill.set_text("ev", "Cloudy.", False)
    short = pill.height()
    pill.set_text("ev", "It's cloudy and about twelve degrees in London right now. " * 3, False)
    assert pill.height() > short

    pill.set_display(False)
    assert (pill.width(), pill.height(), pill._card_h) == (pill.w_, pill.h_, 0) and pill.convo.empty


def test_card_never_outgrows_its_maximum_and_shows_the_latest_words(pill):
    pill.set_display(True)
    pill.set_text("user", "Explain it in detail please", False)
    pill.set_text("ev", " ".join(f"sentence{i} goes on and on." for i in range(200)), False)
    assert pill.height() <= pill.card_max_h
    assert pill._ev_txt.startswith(ELLIPSIS) and "sentence199" in pill._ev_txt


def test_pill_stays_put_when_the_card_opens(pill):
    x0 = pill.x() + (pill.width() - pill.w_) // 2                   # screen x of the pill's left edge
    pill.set_display(True)
    pill.set_text("user", "hi", False)
    assert abs((pill.x() + (pill.width() - pill.w_) // 2) - x0) <= 1
