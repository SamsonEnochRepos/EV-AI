"""A promised action with no tool call is not spoken; several requests in one breath all run; EV can say what it is built on."""
import datetime as dt

from ev.control.tools import _clicked
from ev.llm.brain import Brain, claims_action
from ev.llm.client import OllamaClient
from ev.llm.fastpath import match_compound, match_fast
from ev.safety.gate import SafetyGate
from ev.screen.uia import UIElement, find_element
from ev.tools.base import Tool, ToolResult

from .fakes import FakeOllama, ndjson, text_reply, tool_reply

NOW = lambda: dt.datetime(2026, 1, 2, 15, 4)  # noqa: E731


def _brain(cfg, memory, policy, registry, fake):
    client = OllamaClient(cfg, transport=fake.transport())
    gate = SafetyGate(policy, ask=lambda _r: False)
    return Brain(cfg, client, memory, registry, gate, persona="You are EV.", now=NOW)


def _recording(registry, name, params, required=()):
    """Replace a real tool (it would open an app or move the volume) with one that only records its calls."""
    calls: list[dict] = []

    def fn(**kw):
        calls.append(kw)
        return ToolResult(True, f"{name} done.")

    registry.register(Tool(name, name, params, fn, required))
    return calls


# ---- a claim without a tool call
def test_claims_are_recognised_and_offers_are_not():
    assert claims_action("Sure, I'll play the video by Mihul Mohan.", "x")
    assert claims_action("Sure thing, I'll switch to the VS Code tab for you.", "x")
    assert claims_action("Opening the dashboard.", "x")
    assert claims_action("I'm opening it now.", "x")
    assert not claims_action("Want me to open it?", "open it")
    assert not claims_action("I can open it if you like.", "x")
    assert not claims_action("Let me know if you need more.", "x")
    assert not claims_action("I'll be here.", "x")
    # a recap of something already done is only a claim when the user has just asked for something
    assert not claims_action("I opened YouTube earlier.", "what did you do")
    assert claims_action("I opened YouTube earlier.", "can you do that again")


def test_a_promise_with_no_tool_is_not_spoken_and_the_user_hears_the_truth(cfg, memory, policy, registry):
    fake = FakeOllama([text_reply("Sure, I'll play the video for you. It's ready to watch now.")])
    brain = _brain(cfg, memory, policy, registry, fake)
    spoken = []
    res = brain.respond("can you play the video by Mihul Mohan", on_sentence=spoken.append, allow_fast=False)
    assert len(fake.requests) == 2                                   # the first reply, then one nudge
    assert "no tool" in fake.requests[1]["messages"][-1]["content"]
    assert not any("I'll play" in s or "ready to watch" in s for s in spoken)
    assert res.text == "I wasn't able to do that one. Could you put it another way?"
    assert "I'll play" not in str(brain.history)                      # the false claim must not poison the next turns


def test_the_nudge_gets_the_tool_called(cfg, memory, policy, registry):
    opened = _recording(registry, "open_app", {"name": {"type": "string"}}, ("name",))
    fake = FakeOllama([text_reply("Sure, I'll open Notepad for you."), tool_reply("open_app", {"name": "notepad"}),
                       text_reply("Notepad is open.")])
    brain = _brain(cfg, memory, policy, registry, fake)
    spoken = []
    res = brain.respond("could you get notepad up", on_sentence=spoken.append, allow_fast=False)
    assert opened == [{"name": "notepad"}]
    assert spoken == ["Notepad is open."] and res.tool_log[0]["tool"] == "open_app"


def test_text_followed_by_a_tool_call_in_one_reply_is_just_a_tool_call(cfg, memory, policy, registry):
    opened = _recording(registry, "open_app", {"name": {"type": "string"}}, ("name",))
    both = ndjson({"message": {"role": "assistant", "content": "Sure, I'll open Notepad. "}, "done": False},
                  {"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "open_app", "arguments": {"name": "notepad"}}}]},
                   "done": False},
                  {"message": {"role": "assistant", "content": ""}, "done": True})
    fake = FakeOllama([both, text_reply("Done, it's open.")])
    brain = _brain(cfg, memory, policy, registry, fake)
    spoken = []
    brain.respond("could you get notepad up", on_sentence=spoken.append, allow_fast=False)
    assert opened == [{"name": "notepad"}] and len(fake.requests) == 2       # no nudge: the tool was called
    assert not any("I'll open" in s for s in spoken)


def test_ordinary_chat_is_untouched(cfg, memory, policy, registry):
    fake = FakeOllama([text_reply("Doing great, thanks. I'll be around if you need me.")])
    brain = _brain(cfg, memory, policy, registry, fake)
    res = brain.respond("how are you", allow_fast=False)
    assert len(fake.requests) == 1 and res.text.startswith("Doing great")


# ---- several requests in one breath
def test_compound_matching():
    names = lambda t: [(m.tool, m.args) for m in (match_compound(t) or [])]  # noqa: E731
    assert names("open spotify and turn the volume down") == [("open_app", {"name": "spotify"}), ("set_volume", {"direction": "down"})]
    assert names("open notepad and calculator") == [("open_app", {"name": "notepad"}), ("open_app", {"name": "calculator"})]
    assert [t for t, _ in names("close edge and then open vs code")] == ["close_app", "open_app"]
    assert [t for t, _ in names("set a timer for five minutes and open spotify")] == ["set_timer", "open_app"]
    # these stay with the existing matchers
    assert match_compound("open youtube and search for cats") is None             # a search needs the model afterwards
    assert match_compound("open notepad and type hello") is None                  # waits for the window: the screen agent
    assert match_compound("remind me in ten minutes to stop and close the laptop") is None
    assert match_compound("salt and pepper recipes") is None
    assert match_compound("open spotify") is None


def test_every_part_of_a_compound_request_runs_in_order(cfg, memory, policy, registry):
    opened = _recording(registry, "open_app", {"name": {"type": "string"}}, ("name",))
    volume = _recording(registry, "set_volume", {"direction": {"type": "string"}})
    fake = FakeOllama([text_reply("should not be used")])
    brain = _brain(cfg, memory, policy, registry, fake)
    spoken = []
    res = brain.respond("open spotify and turn the volume down", on_sentence=spoken.append)
    assert opened == [{"name": "spotify"}] and volume == [{"direction": "down"}]
    assert [r["tool"] for r in res.tool_log] == ["open_app", "set_volume"] and not fake.requests
    assert spoken == ["open_app done. set_volume done."]


def test_a_failed_part_does_not_stop_the_rest(cfg, memory, policy, registry):
    registry.register(Tool("open_app", "x", {"name": {"type": "string"}}, lambda name: ToolResult(False, f"No {name}."), ("name",)))
    volume = _recording(registry, "set_volume", {"direction": {"type": "string"}})
    brain = _brain(cfg, memory, policy, registry, FakeOllama([text_reply("unused")]))
    res = brain.respond("open spotify and turn the volume down")
    assert volume == [{"direction": "down"}] and res.text.startswith("No spotify.")


# ---- clicking: a log line that mentions the target is not the target
def test_a_paragraph_that_mentions_the_target_is_not_a_control():
    els = [UIElement("2026-11-09 17:23:12,598 INFO ev.app: heard (follow-up): Alright, can you click on the YouTube site? and more words here", "Text", (0, 0, 9, 9)),
           UIElement("YouTube", "Hyperlink", (0, 0, 9, 9))]
    assert find_element(els, "youtube").name == "YouTube"
    assert find_element(els[:1], "youtube site") is None


def test_a_long_clicked_label_is_not_read_out():
    assert _clicked("Search") == "Clicked Search."
    assert _clicked("x" * 90) == "Clicked it."


# ---- the 17:43 log: invented results, "Evie stop", search-then-play, phrasings that matched nothing
def test_invented_results_from_the_real_log_are_claims():
    for said, heard in [
        ('Sure thing! I opened YouTube in Microsoft Edge and clicked on the channel. The first video that comes up is "mr" by MrBeast.',
         "Now, within the YouTube, can you... So you just open the YouTube tab, right? Just click on it."),
        ("I opened YouTube in Firefox and clicked on the channel.", "You can close the Edge tab and switch to the Firefox browser"),
        ("I opened YouTube in Firefox and clicked on the channel.", "Evie Stop"),
        ("The first video that comes up is a good one.", "what is the weather"),
    ]:
        assert claims_action(said, heard), said
    # asking what happened earlier is still answerable
    assert not claims_action("I opened YouTube in Firefox.", "what did you open earlier?")
    assert not claims_action("I opened YouTube in Firefox.", "did you open it?")


def test_stop_words_are_a_command_not_a_question():
    from ev.llm.fastpath import is_stop

    for t in ["Evie Stop", "Evie Store.", "Evie, stop talking", "stop", "Stop it", "be quiet", "shut up", "EV, stock"]:
        assert is_stop(t), t
    for t in ["stop the music", "stop by the store", "Evie, stop eating", "what is a stop"]:
        assert not is_stop(t), t


def test_tab_phrasings_are_left_to_the_model_and_window_switches_match():
    assert match_fast("close the tab") is None and match_fast("switch to the firefox browser").args == {"title": "firefox"}
    from ev.stt.whisper import fix_names

    assert fix_names("close the makersoft edge tab") == "close the Microsoft edge tab"


def test_when_the_model_will_not_call_a_tool_the_user_hears_the_truth(cfg, memory, policy, registry):
    brain = _brain(cfg, memory, policy, registry, FakeOllama([text_reply("Sure, I'll set that up for you.")]))
    spoken = []
    brain.respond("Can you sort my life out", on_sentence=spoken.append, allow_fast=False)
    assert spoken == ["I wasn't able to do that one. Could you put it another way?"]
