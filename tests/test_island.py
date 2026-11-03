"""The island's animation maths (ev/overlay/anim.py): eyes, springs, shape and placement. No Qt needed."""
from __future__ import annotations

import random

import pytest

from ev.overlay.anim import (
    Eyes,
    IslandSizes,
    Spring,
    approach,
    body_origin,
    corner_radius,
    expansion,
    island_shape,
    place_window,
)

S = IslandSizes()


def run(eyes: Eyes, seconds: float, state: str = "idle", **kw) -> None:
    t = kw.pop("start", 0.0)
    for i in range(int(seconds * 60)):
        eyes.step(1 / 60, t + i / 60, state, **kw)


def test_eyes_look_toward_the_pointer_and_saturate():
    e = Eyes(random.Random(1))
    run(e, 1.0, cursor=(400.0, 300.0))
    assert e.gx > 0.7 and e.gy > 0.6                       # pointer is right of and below the island
    run(e, 1.0, cursor=(-400.0, -300.0))
    assert e.gx < -0.7 and e.gy < -0.6
    run(e, 1.0, cursor=(100000.0, 0.0))
    assert -1.0 <= e.gx <= 1.0                              # a far away pointer never pushes the eyes out of the pill


def test_a_pointer_that_has_stopped_moving_lets_the_eyes_wander_but_a_busy_state_does_not_follow_it():
    e = Eyes(random.Random(2))
    run(e, 0.5, cursor=(300.0, 0.0), still_s=60.0)         # idle + a long-still pointer: free to look around
    seen = set()
    for i in range(600):
        e.step(1 / 60, 40.0 + i / 60, "idle", cursor=(300.0, 0.0), still_s=60.0)
        seen.add(round(e.gx, 1))
    assert len(seen) > 3
    t = Eyes(random.Random(2))
    run(t, 1.0, "thinking", cursor=(300.0, 0.0))
    assert t.gy < -0.4                                      # thinking: looking up and away


def test_eyes_blink_regularly_and_close_when_paused():
    e = Eyes(random.Random(3))
    lids = []
    for i in range(60 * 20):
        e.step(1 / 60, i / 60, "idle")
        lids.append(e.lid)
    assert min(lids) < 0.2 and max(lids) == pytest.approx(1.0, abs=0.01)
    s = Eyes(random.Random(3))
    run(s, 1.0, "paused")
    assert s.lid < 0.15 and s.squint == 0.0


def test_hovering_or_a_cheer_makes_the_eyes_smile_and_it_wears_off():
    e = Eyes(random.Random(4))
    run(e, 1.0, hover=True)
    assert e.squint > 0.9
    run(e, 1.0, hover=False, start=1.0)
    assert e.squint < 0.1
    e.cheer(5.0, 1.0)
    for i in range(30):
        e.step(1 / 60, 5.0 + i / 60, "idle")
    assert e.squint > 0.3
    run(e, 2.0, start=7.0)
    assert e.squint < 0.1


def test_eyes_swell_with_the_voice_and_turn_accent_when_acting():
    e = Eyes(random.Random(5))
    run(e, 1.0, "listening", level=1.0)
    assert e.scale > 1.2
    run(e, 1.0, "acting")
    assert e.alert > 0.9 and e.open < 0.7
    run(e, 1.0, "idle", start=1.0)
    assert e.alert < 0.1


def test_spring_arrives_with_a_little_overshoot_and_survives_a_long_frame():
    sp = Spring(116.0)
    peak = 0.0
    for _ in range(240):
        peak = max(peak, sp.step(236.0, 1 / 60))
    assert sp.settled(236.0) and 236.0 < peak < 236.0 * 1.08
    big = Spring(0.0)
    assert 0.0 < big.step(100.0, 5.0) < 200.0                # a 5 s stall is clamped, not exploded


def test_approach_is_frame_rate_independent():
    a = approach(0.0, 1.0, 0.5, 4.0)
    b = 0.0
    for _ in range(50):
        b = approach(b, 1.0, 0.01, 4.0)
    assert a == pytest.approx(b, abs=1e-6)


def test_shape_follows_state_and_text_takes_over():
    assert island_shape("idle", S) == (S.idle_w, S.idle_h)
    assert island_shape("paused", S) == (S.idle_w, S.idle_h)
    assert island_shape("speaking", S) == (S.wide_w, S.wide_h) == island_shape("listening", S)
    mid = island_shape("thinking", S)
    assert S.idle_w < mid[0] < S.wide_w and S.idle_h < mid[1] < S.wide_h
    assert island_shape("idle", S, card_h=80) == (S.card_w, S.strip_h + 80)
    assert expansion(S.idle_w, S) == 0 and expansion(S.wide_w, S) == 1 and expansion(S.card_w, S) == 1


def test_corners_are_a_full_pill_at_rest_and_never_rounder_than_a_card():
    assert corner_radius(S.idle_w, S.idle_h, S) == pytest.approx(S.idle_h / 2)
    assert corner_radius(S.card_w, 300, S) == S.card_radius
    assert corner_radius(S.wide_w, S.wide_h, S) < S.wide_h / 2


@pytest.mark.parametrize("anchor,ox,expect", [
    ("top-center", 0, (200, -6)),            # (800 - 400) / 2 centred; body 6 px from the top, window starts one margin above that
    ("top-left", 10, (-2, -6)),              # body 10 px from the left edge
    ("top-right", 10, (402, -6)),            # 800 - 400 + 12 - 10: the body then ends 10 px from the right edge
    ("bottom-center", 0, (200, 506)),        # 600 - 100 + 12 - 6
])
def test_window_placement_puts_the_body_at_the_anchored_edge(anchor, ox, expect):
    assert place_window(anchor, 800, 600, 400, 100, ox, 6, 12) == expect


def test_body_grows_away_from_the_anchored_edge():
    assert body_origin("center", "top", 460, 300, 116, 34, 12) == (172.0, 12.0)
    assert body_origin("left", "top", 460, 300, 116, 34, 12)[0] == 12.0
    assert body_origin("right", "top", 460, 300, 116, 34, 12)[0] == 460 - 12 - 116
    assert body_origin("center", "bottom", 460, 300, 116, 34, 12)[1] == 300 - 12 - 34
