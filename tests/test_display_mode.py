"""Display mode: the text card under the pill. Model, wire protocol, client behaviour and card layout."""
from __future__ import annotations

import os
import time

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
    assert (pill.target_shape(), pill._card_h) == ((pill.w_, pill.h_), 0)

    pill.set_display(True)
    assert pill._card_h == 0 and pill.target_shape() == (pill.w_, pill.h_)      # no text yet: the island stays a pill
    pill.set_text("user", "What's the weather in London?", False)
    pill.set_text("ev", "Cloudy.", False)
    short = pill.target_shape()[1]
    assert pill.target_shape()[0] == pill.card_w
    pill.set_text("ev", "It's cloudy and about twelve degrees in London right now. " * 3, False)
    assert pill.target_shape()[1] > short

    pill.set_display(False)
    assert (pill.target_shape(), pill._card_h) == ((pill.w_, pill.h_), 0) and pill.convo.empty


@pytest.mark.parametrize("msg,ok", [
    ({"t": "notice", "v": "Battery is at 15 percent."}, True), ({"t": "notice", "v": ""}, False), ({"t": "notice"}, False),
    ({"t": "notice", "v": 5}, False), ({"t": "notice", "v": "x" * (protocol.MAX_TEXT + 1)}, False),
])
def test_protocol_validates_notices(msg, ok):
    assert protocol.valid_message(msg) is ok


def test_a_notice_is_sent_even_when_display_mode_is_off_and_is_trimmed(client):
    client.send_notice("  Battery is at\n15 percent. ")
    client.send_notice("")
    client.send_notice("word " * 400)
    assert client.sock.sent[0] == {"t": "notice", "v": "Battery is at 15 percent."} and len(client.sock.sent) == 2
    assert len(client.sock.sent[1]["v"]) <= protocol.MAX_TEXT and client.sock.sent[1]["v"].endswith("…")


def test_a_notice_shows_a_card_with_display_mode_off_then_goes_away(pill):
    assert not pill.display
    pill.set_notice("Your build just finished.")
    assert pill._card_h > 0 and pill.target_shape()[0] == pill.card_w and pill._ev_txt == "Your build just finished."
    pill._notice_until = 0.0
    pill.tick()                                                     # the card's time is up
    assert (pill.target_shape(), pill._card_h) == ((pill.w_, pill.h_), 0) and pill.convo.empty and not pill.display


def test_a_notice_does_not_turn_display_mode_on_or_off(pill):
    pill.set_display(True)
    pill.set_notice("Heads up.")
    pill._notice_until = 0.0
    pill.tick()
    assert pill.display and pill._card_h == 0                       # display mode is still on; the notice left no text behind
    pill.set_text("user", "hello", False)
    assert pill._card_h > 0


def test_a_real_conversation_line_takes_the_card_from_a_notice(pill):
    pill.set_display(True)
    pill.set_notice("Heads up.")
    pill.set_text("user", "What was that?", False)
    pill._notice_until = 0.0
    pill.tick()
    assert pill._user_txt == "What was that?"                       # the expiring notice did not wipe the conversation


def test_card_never_outgrows_its_maximum_and_shows_the_latest_words(pill):
    pill.set_display(True)
    pill.set_text("user", "Explain it in detail please", False)
    pill.set_text("ev", " ".join(f"sentence{i} goes on and on." for i in range(200)), False)
    assert pill.target_shape()[1] <= pill.card_max_h
    assert pill._ev_txt.startswith(ELLIPSIS) and "sentence199" in pill._ev_txt


def test_island_stays_put_when_the_card_opens(pill):
    """The window never resizes or moves, and the body stays centred under the same spot while it grows."""
    win = (pill.x(), pill.y(), pill.width(), pill.height())
    cx = pill.body_rect().center().x()
    pill.set_display(True)
    pill.set_text("user", "hi", False)
    pill.settle("speaking", 0.5)
    assert (pill.x(), pill.y(), pill.width(), pill.height()) == win
    assert abs(pill.body_rect().center().x() - cx) <= 1
    assert pill.body_rect().top() == pytest.approx(12)              # pinned to the top edge, growing downward


def test_island_opens_for_speech_and_closes_when_idle(pill):
    pill.settle("idle")
    resting = pill.body_rect().width()
    pill.settle("speaking", 0.6)
    assert pill.body_rect().width() > resting
    pill.settle("thinking")
    assert resting < pill.body_rect().width() < pill.sizes.wide_w
    pill.settle("idle")
    assert pill.body_rect().width() == pytest.approx(resting)


def test_a_finished_card_lingers_then_folds_back_into_the_island(pill):
    pill.set_display(True)
    pill.set_text("user", "hello", False)
    pill.set_text("ev", "Hi.", False)
    assert pill._card_h > 0
    pill.set_state("idle")                                          # EV is done talking
    pill.idle_since = time.monotonic() - pill.linger - 1
    pill.tick()
    assert pill._card_h == 0 and pill.convo.empty and pill.display  # the text went; display mode is still on
