"""Latency and conversation-feel features: direct tool replies, parallel tools, filler, smalltalk, warm-up."""
from __future__ import annotations

import datetime as dt
import json
import threading
import time

import numpy as np

from ev.llm.brain import Brain
from ev.llm.client import OllamaClient
from ev.llm.fastpath import match_fast, match_smalltalk
from ev.llm.sentences import SentenceSplitter
from ev.safety.gate import SafetyGate
from ev.tools.base import Tool, ToolRegistry, ToolResult

from .fakes import FakeOllama, text_reply, tool_reply

NOW = lambda: dt.datetime(2026, 1, 2, 15, 4)  # noqa: E731


def make(cfg, memory, policy, registry, fake):
    client = OllamaClient(cfg, transport=fake.transport())
    brain = Brain(cfg, client, memory, registry, SafetyGate(policy, ask=lambda _r: False), persona="You are EV.", now=NOW)
    brain.direct_replies = True                          # off by default (see config/ev.toml); these tests cover the feature
    return brain


def multi_reply(*calls):
    """One model turn that asks for several tools at once."""
    tcs = [{"function": {"name": n, "arguments": a}} for n, a in calls]
    line = {"message": {"role": "assistant", "content": "", "tool_calls": tcs}, "done": False}
    done = {"message": {"role": "assistant", "content": ""}, "done": True}
    return (json.dumps(line) + "\n" + json.dumps(done) + "\n").encode()


# ---- direct tool replies
def test_direct_tool_reply_skips_the_second_model_call(cfg, memory, policy, registry):
    fake = FakeOllama([tool_reply("remember_fact", {"key": "dog", "value": "Biscuit"}), text_reply("never used")])
    brain = make(cfg, memory, policy, registry, fake)
    spoken = []
    res = brain.respond("my dog is Biscuit, remember that", on_sentence=spoken.append, allow_fast=False)
    assert len(fake.requests) == 1                      # one model call, not two
    assert spoken == ["Got it, I'll remember that."] and res.text == spoken[0]
    assert memory.recall("dog") == "Biscuit"


def test_failed_tool_still_goes_back_to_the_model(cfg, memory, policy, registry):
    fake = FakeOllama([tool_reply("set_timer", {"minutes": 0, "seconds": 0}), text_reply("How long should it be?")])
    brain = make(cfg, memory, policy, registry, fake)
    res = brain.respond("set a timer", allow_fast=False)
    assert len(fake.requests) == 2 and res.text == "How long should it be?"


def test_untrusted_results_are_never_spoken_directly(cfg, memory, policy, registry):
    registry.register(Tool("read_screen", "r", {}, lambda: ToolResult(True, "some page text", untrusted=True,
                                                                      source="screen"), direct=True))
    fake = FakeOllama([tool_reply("read_screen", {}), text_reply("It's a web page.")])
    brain = make(cfg, memory, policy, registry, fake)
    res = brain.respond("read my screen", allow_fast=False)
    assert len(fake.requests) == 2 and res.text == "It's a web page."


def test_direct_replies_are_off_by_default_and_the_model_rephrases(cfg, memory, policy, registry):
    fake = FakeOllama([tool_reply("take_note", {"text": "milk"}), text_reply("Noted, milk.")])
    brain = make(cfg, memory, policy, registry, fake)
    brain.direct_replies = bool(cfg["llm"].get("direct_tool_replies", False))
    assert brain.direct_replies is False
    assert brain.respond("note milk", allow_fast=False).text == "Noted, milk."


# ---- parallel tools
def test_safe_tools_in_one_turn_run_concurrently(cfg, memory, policy):
    reg = ToolRegistry()
    seen = []
    barrier = threading.Barrier(2, timeout=3)

    def slow(city: str) -> ToolResult:
        barrier.wait()                                   # only passes if both calls are running at the same time
        seen.append(city)
        return ToolResult(True, f"{city} is sunny.")

    reg.register(Tool("get_weather", "w", {"city": {"type": "string"}}, slow, ("city",), direct=True))
    policy.levels["get_weather"] = "safe"
    fake = FakeOllama([multi_reply(("get_weather", {"city": "Pune"}), ("get_weather", {"city": "Delhi"}))])
    brain = make(cfg, memory, policy, reg, fake)
    res = brain.respond("weather in pune and delhi", allow_fast=False)
    assert sorted(seen) == ["Delhi", "Pune"]
    assert res.text == "Pune is sunny. Delhi is sunny." and len(fake.requests) == 1


def test_ui_tools_stay_in_order(cfg, memory, policy):
    reg = ToolRegistry()
    order = []
    for name in ("open_app", "close_app"):
        reg.register(Tool(name, "x", {"name": {"type": "string"}},
                          lambda name, _n=name: order.append(_n) or ToolResult(True, "ok."), ("name",), direct=True))
    policy.levels.update(open_app="safe", close_app="safe")
    brain = make(cfg, memory, policy, reg, FakeOllama([multi_reply(("open_app", {"name": "notepad"}),
                                                                    ("close_app", {"name": "paint"}))]))
    brain.respond("do both", allow_fast=False)
    assert order == ["open_app", "close_app"]


def test_parallel_batch_still_goes_through_the_gate(cfg, memory, policy):
    reg = ToolRegistry()
    ran = []
    reg.register(Tool("take_note", "n", {"text": {"type": "string"}},
                      lambda text: ran.append(text) or ToolResult(True, "Noted."), ("text",), direct=True))
    reg.register(Tool("set_reminder", "n", {"text": {"type": "string"}},
                      lambda text: ran.append(text) or ToolResult(True, "ok."), ("text",), direct=True))
    policy.levels.update(take_note="safe", set_reminder="blocked")
    fake = FakeOllama([multi_reply(("take_note", {"text": "a"}), ("set_reminder", {"text": "b"})),
                       text_reply("Only one went through.")])
    brain = make(cfg, memory, policy, reg, fake)
    res = brain.respond("both", allow_fast=False)
    assert ran == ["a"] and res.tool_log[1].get("denied")
    assert res.text == "Only one went through."          # one result was denied, so the model gets to explain


# ---- filler for slow tools
def test_slow_tool_gets_a_spoken_filler(cfg, memory, policy):
    reg = ToolRegistry()
    reg.register(Tool("web_search", "s", {"query": {"type": "string"}},
                      lambda query: (time.sleep(0.4), ToolResult(True, "- A: b", untrusted=True, source="web_search"))[1],
                      ("query",)))
    policy.levels["web_search"] = "safe"
    fake = FakeOllama([tool_reply("web_search", {"query": "x"}), text_reply("It's fine.")])
    brain = make(cfg, memory, policy, reg, fake)
    brain.filler_after_s = 0.1
    spoken = []
    res = brain.respond("look into this for me", on_sentence=spoken.append, allow_fast=False)
    assert len(spoken) == 2 and spoken[1] == "It's fine." and spoken[0].endswith(".")
    assert res.text == "It's fine."                      # the filler is not part of the answer or the history


def test_quick_tool_gets_no_filler(cfg, memory, policy, registry):
    brain = make(cfg, memory, policy, registry, FakeOllama([tool_reply("get_time_date", {}), text_reply("x")]))
    brain.filler_after_s = 0.5
    spoken = []
    brain.respond("what's the time please sir", on_sentence=spoken.append, allow_fast=False)
    assert len(spoken) == 1


# ---- smalltalk
def test_smalltalk_bypasses_the_model(cfg, memory, policy, registry):
    fake = FakeOllama([text_reply("unused")])
    brain = make(cfg, memory, policy, registry, fake)
    spoken = []
    res = brain.respond("thanks!", on_sentence=spoken.append)
    assert res.fast_path and spoken == [res.text] and fake.requests == []
    assert brain.history[-1]["content"] == res.text


def test_smalltalk_matches_only_whole_pleasantries():
    def pick(options):
        return options[0]

    for said in ("thanks", "Thank you so much", "hey Evie", "hello there", "good morning", "goodbye", "ev, thanks a lot"):
        assert match_smalltalk(said, pick), said
    for said in ("thanks for the weather, now open youtube", "hello world program in python", "yes", "okay", "later",
                 "no thanks", "hey can you open spotify"):
        assert match_smalltalk(said, pick) is None, said


def test_smalltalk_replies_vary():
    assert len({match_smalltalk("thanks") for _ in range(60)}) > 1


def test_smalltalk_never_shadows_a_command():
    assert match_fast("thanks") is None and match_fast("hey") is None


# ---- early clause chunk
def test_first_chunk_can_end_at_a_comma():
    s = SentenceSplitter(min_chars=5, clause_chars=25)
    assert s.feed_ex("Honestly, I think that's a really good idea ") == []         # no comma long enough yet
    out = s.feed_ex("for today, and I'd go for it")
    assert out == [("Honestly, I think that's a really good idea for today,", False)]
    out = s.feed_ex(". Then rest. ")
    assert out == [("and I'd go for it.", True), ("Then rest.", True)]


def test_clause_split_happens_once_and_never_inside_numbers():
    assert SentenceSplitter(min_chars=5, clause_chars=10).feed_ex("It costs 1,000 rupees and more ") == []
    s = SentenceSplitter(min_chars=5, clause_chars=10)
    out = s.feed_ex("Well, that is one thing, and that is another, ok")
    out += s.feed_ex(". ")
    assert [c[1] for c in out] == [False, True]          # one clause chunk, then the sentence


def test_clause_splitting_is_off_by_default():
    assert SentenceSplitter(min_chars=5).feed_ex("A long opening part here, and then the rest of it follows ") == []


def test_clause_chunk_does_not_use_up_the_sentence_cap(cfg, memory, policy, registry):
    cfg.raw["tts"]["first_clause_chars"] = 20
    reply = "Well, that is a pretty good question, to ask me right now. And here is the second one. Not a third one."
    brain = make(cfg, memory, policy, registry, FakeOllama([text_reply(reply, parts=24)]))
    spoken = []
    brain.respond("what do you think about all this", on_sentence=spoken.append, allow_fast=False)
    assert spoken[0].endswith(",") and len(spoken) == 3        # one clause + two whole sentences = the 2-sentence cap
    assert spoken[1] == "to ask me right now." and spoken[-1] == "And here is the second one."


# ---- prompt cache friendliness and warm-up
def test_system_prompt_ignores_the_minute(cfg, memory, policy, registry):
    times = iter([dt.datetime(2026, 1, 2, 15, 4), dt.datetime(2026, 1, 2, 15, 59)])
    brain = Brain(cfg, OllamaClient(cfg, transport=FakeOllama([]).transport()), memory, registry,
                  SafetyGate(policy), persona="You are EV.", now=lambda: next(times))
    first, second = brain.system_prompt(), brain.system_prompt()
    assert first == second and "afternoon" in first and "Friday, January 2, 2026" in first


def test_warm_sends_the_real_prompt_and_tools(cfg, memory, policy, registry):
    fake = FakeOllama([text_reply("")])
    brain = make(cfg, memory, policy, registry, fake)
    brain.warm()
    req = fake.requests[0]
    assert req["messages"][0] == {"role": "system", "content": brain.system_prompt()}
    assert req["tools"] == registry.schemas() and req["options"]["num_predict"] == 1


def test_rewarm_only_when_cold(cfg, memory, policy, registry):
    brain = make(cfg, memory, policy, registry, FakeOllama([text_reply("")]))

    def settle():
        for _ in range(100):
            if not brain._warm_lock.locked():
                return
            time.sleep(0.02)

    assert brain.rewarm_if_cold()                          # never used: cold
    settle()
    assert not brain.rewarm_if_cold()                      # the warm-up itself made it warm
    brain.client.unload(brain.client.model)                # screen reading swaps the model out
    assert brain.rewarm_if_cold()
    settle()


def test_summary_runs_in_the_background(cfg, memory, policy, registry):
    release = threading.Event()

    def reply(body, i):
        if "Update the running summary" in body["messages"][0]["content"]:
            release.wait(2)
            return text_reply("Likes jazz.")
        return text_reply("Sure thing.")

    brain = make(cfg, memory, policy, registry, FakeOllama(reply))
    cfg.raw["memory"]["summary_every_turns"] = 1
    t0 = time.perf_counter()
    brain.respond("tell me something nice today", allow_fast=False)
    assert time.perf_counter() - t0 < 1.0                  # the turn did not wait for the summary
    release.set()
    for _ in range(100):
        if memory.get_kv("summary"):
            break
        time.sleep(0.05)
    assert memory.get_kv("summary") == "Likes jazz."


# ---- registry
def test_run_many_keeps_order_and_reports_timeouts():
    reg = ToolRegistry()
    fast = Tool("fast", "", {}, lambda: ToolResult(True, "a"))
    slow = Tool("slow", "", {}, lambda: (time.sleep(0.5), ToolResult(True, "b"))[1], timeout_s=0.05)
    out = reg.run_many([(slow, {}), (fast, {})])
    assert [r.ok for r in out] == [False, True] and "timed out" in out[0].message


# ---- TTS warm-up
def test_speaker_warm_synthesizes_without_playing():
    from ev.audio.playback import Player
    from ev.tts.speaker import Speaker

    played = []

    class Eng:
        name, sample_rate = "fake", 22050

        def synthesize(self, text):
            yield np.zeros(100, np.float32)

    sp = Speaker(Eng(), Player(sink=lambda seg, sr: played.append(len(seg))))
    sp.warm()
    time.sleep(0.05)
    assert played == [] and not sp.is_speaking
    sp.close()
