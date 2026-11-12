"""The island's animation maths (ev/overlay/anim.py): springs, shape and placement. No Qt needed."""
from __future__ import annotations

import pytest

from ev.overlay.anim import (
    IslandSizes,
    Spring,
    alert_target,
    approach,
    body_origin,
    corner_radius,
    expansion,
    island_shape,
    place_window,
)

S = IslandSizes()


def test_the_accent_colour_is_for_working_on_the_screen_and_errors_only():
    assert alert_target("acting") == alert_target("error") == 1.0
    assert all(alert_target(st) == 0.0 for st in ("idle", "listening", "thinking", "speaking", "paused"))


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
