"""A promised action with no tool call is not spoken; several requests in one breath all run; EV can say what it is built on."""
import datetime as dt

from ev.control.tools import _clicked
from ev.llm.brain import Brain, claims_action
from ev.llm.client import OllamaClient
from ev.llm.fastpath import match_compound, match_fast
from ev.llm.selfinfo import SelfInfo, looks_about_self, spoken_model, spoken_voice
from ev.safety.gate import SafetyGate
from ev.screen.ocr import OCR, OcrItem
from ev.screen.uia import UIElement, find_element
from ev.tools.base import Tool, ToolResult

from .fakes import FakeOllama, ndjson, text_reply, tool_reply

NOW = lambda: dt.datetime(2026, 1, 2, 15, 4)  # noqa: E731


def _brain(cfg, memory, policy, registry, fake, selfinfo=None):
    client = OllamaClient(cfg, transport=fake.transport())
    gate = SafetyGate(policy, ask=lambda _r: False)
    return Brain(cfg, client, memory, registry, gate, persona="You are EV.", now=NOW, selfinfo=selfinfo)


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


# ---- the play-a-pick phrasing now reaches the screen agent
def test_play_the_video_by_goes_to_the_screen_agent():
    m = match_fast("Can you play the video by Mihul Mohan?")
    assert m and m.tool == "do_on_screen" and "Mihul Mohan" in m.args["goal"]
    assert match_fast("play the next song") is None                              # not a pick on a page; it never was a command either
    assert match_fast("play").tool == "media_control"


# ---- self-knowledge
def test_model_and_voice_names_read_well_aloud():
    assert spoken_model("qwen2.5:3b") == "Qwen 2 point 5, 3 billion"
    assert spoken_model("llama3.2:latest") == "Llama 3 point 2"
    assert spoken_model("qwen2.5vl:3b") == "Qwen 2 point 5 VL, 3 billion"
    assert spoken_voice("en_US-hfc_female-medium") == "hfc female, medium quality"


def test_questions_about_itself_are_answered_from_the_config(cfg):
    topic = lambda t: (match_fast(t).args["topic"] if match_fast(t) else None)  # noqa: E731
    assert topic("What's your current text-to-speech model name?") == "tts"
    assert topic("which speech to text model do you use") == "stt"
    assert topic("what LLM are you running on") == "llm"
    assert topic("how were you built") == "build"
    assert topic("what technologies do you use") == "build"
    assert topic("how do you hear me") == "stt"
    assert topic("tell me about yourself") == "overview"
    assert topic("what is a language model") is None                             # a general question is not about EV
    info = SelfInfo(cfg, lambda: {"stt_model": "small", "stt_device": "cuda", "tts_engine": "piper", "llm_model": "qwen2.5:3b"})
    assert "Piper" in info.tts() and "hfc female" in info.tts()
    assert "Whisper small" in info.stt() and "graphics card" in info.stt()
    assert "Qwen 2 point 5, 3 billion" in info.llm() and "Ollama" in info.llm()
    assert "hey evie" in info.wake() and "openWakeWord" in info.wake()


def test_what_is_really_loaded_beats_the_config(cfg):
    info = SelfInfo(cfg, lambda: {"tts_engine": "kokoro", "stt_device": "cpu", "llm_model": "llama3.2:latest"})
    assert "Kokoro" in info.tts() and "processor" in info.stt() and info.llm().startswith("I think with Llama 3 point 2")


def test_a_broken_status_read_does_not_break_an_answer(cfg):
    def boom():
        raise RuntimeError("no stt yet")

    assert "Whisper" in SelfInfo(cfg, boom).stt()


def test_about_self_through_the_brain_is_a_gated_call_with_no_model(cfg, memory, policy, registry):
    info = SelfInfo(cfg, lambda: {"tts_engine": "piper"})                         # (this machine's ev.local.toml picks kokoro)
    registry.register(Tool("about_self", "x", {"topic": {"type": "string", "enum": ["tts", "overview"]}},
                           lambda topic="overview": ToolResult(True, info.answer(topic)), (), direct=True, llm_visible=False))
    fake = FakeOllama([text_reply("it's about Firefox")])
    brain = _brain(cfg, memory, policy, registry, fake, selfinfo=info)
    res = brain.respond("What's your current text-to-speech model name?")
    assert not fake.requests and "Piper" in res.text and res.tool_log[0]["decision"] == "allow"
    assert "about_self" not in str(registry.schemas())                           # costs the model's prompt nothing


def test_a_looser_self_question_gets_the_facts_beside_it(cfg, memory, policy, registry):
    fake = FakeOllama([text_reply("I run on a small local model.")])
    brain = _brain(cfg, memory, policy, registry, fake, selfinfo=SelfInfo(cfg, lambda: {"tts_engine": "piper"}))
    assert looks_about_self("why is your voice so quiet today")
    assert not looks_about_self("what's the weather")
    brain.respond("so what do you reckon your brain is made of", allow_fast=False)
    sent = fake.requests[0]["messages"][-1]["content"]
    assert "True facts about yourself" in sent and "Piper" in sent
    fake2 = FakeOllama([text_reply("Sunny.")])
    _brain(cfg, memory, policy, registry, fake2, selfinfo=SelfInfo(cfg)).respond("what's the weather like", allow_fast=False)
    assert "True facts about yourself" not in fake2.requests[0]["messages"][-1]["content"]


def test_about_self_has_a_policy_level(policy):
    assert policy.levels["about_self"] == "safe"


# ---- clicking: a log line that mentions the target is not the target
def test_a_paragraph_that_mentions_the_target_is_not_a_control():
    els = [UIElement("2026-11-09 17:23:12,598 INFO ev.app: heard (follow-up): Alright, can you click on the YouTube site? and more words here", "Text", (0, 0, 9, 9)),
           UIElement("YouTube", "Hyperlink", (0, 0, 9, 9))]
    assert find_element(els, "youtube").name == "YouTube"
    assert find_element(els[:1], "youtube site") is None
    items = [OcrItem("2026-11-09 17:23:12,598 INFO ev.app: heard (follow-up): can you click on the YouTube site? and more", (0, 0, 5, 5), 0.9)]
    assert OCR.find(items, "youtube site") is None


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


def test_a_search_then_play_is_two_steps(cfg, memory, policy, registry):
    steps = match_compound("Can you open YouTube in Microsoft Edge and search for Mr. Who's the Boss and play the first video that comes out?")
    assert [s.tool for s in steps] == ["open_website", "do_on_screen"]
    assert steps[0].args["query"] == "mr, who's the boss" and "first video" in steps[1].args["goal"]
    opened = _recording(registry, "open_website", {"site": {"type": "string"}, "query": {"type": "string"}, "browser": {"type": "string"}})
    picked = _recording(registry, "do_on_screen", {"goal": {"type": "string"}}, ("goal",))
    brain = _brain(cfg, memory, policy, registry, FakeOllama([text_reply("unused")]))
    brain.page_load_s = 0
    res = brain.respond("open YouTube and search for cats and then click the first result")
    assert opened == [{"site": "youtube", "query": "cats"}] and "first result" in picked[0]["goal"]
    assert [r["tool"] for r in res.tool_log] == ["open_website", "do_on_screen"]


def test_page_things_go_to_the_screen_agent_and_the_other_phrasings_match():
    for t in ["Can you open the YouTube channel?", "close the Edge tab", "switch to the VS Code tab", "open the first result",
              "play the video which says I tested most futuristic tech"]:
        assert match_fast(t).tool == "do_on_screen", t
    assert match_fast("close the tab") is None and match_fast("switch to the firefox browser").args == {"title": "firefox"}
    assert match_fast("What is the STD model we are using?").args == {"topic": "stt"}
    from ev.stt.whisper import fix_names

    assert fix_names("close the makersoft edge tab") == "close the Microsoft edge tab"


def test_when_the_model_will_not_call_a_tool_the_screen_agent_does_it(cfg, memory, policy, registry):
    picked = _recording(registry, "do_on_screen", {"goal": {"type": "string"}}, ("goal",))
    fake = FakeOllama([text_reply("Great! I opened the video and clicked on it.")])
    brain = _brain(cfg, memory, policy, registry, fake)
    spoken = []
    brain.respond("Can you play the video which says best tech", on_sentence=spoken.append, allow_fast=False)
    assert picked == [{"goal": "Can you play the video which says best tech"}]
    assert spoken == ["do_on_screen done."] and "clicked on it" not in str(brain.history)
    # nothing to do on screen: the plain truth
    brain2 = _brain(cfg, memory, policy, registry, FakeOllama([text_reply("Sure, I'll set that up for you.")]))
    spoken2 = []
    brain2.respond("Can you sort my life out", on_sentence=spoken2.append, allow_fast=False)
    assert spoken2 == ["I wasn't able to do that one. Could you put it another way?"]
