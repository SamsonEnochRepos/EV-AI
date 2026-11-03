"""What the dashboard can ask of EV. Every method here either reads state, tunes an allow-listed setting, or goes through
the same gate as a spoken request (skills, forgetting, confirming a fact). Nothing here edits policy.toml, and the only settings
file it writes is config/ev.local.toml."""
from __future__ import annotations

import copy
import dataclasses
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ev.core.config import Config
from ev.core.events import State
from ev.core.log import get_logger
from ev.dashboard import localconf
from ev.dashboard.server import DashboardError
from ev.dashboard.settings_schema import ALLOWED, SETTINGS, SettingError, coerce
from ev.skills.store import SkillError, canonical, validate
from ev.tts.base import make_engine

if TYPE_CHECKING:
    from ev.app import EVApp

log = get_logger("dashboard.backend")

KOKORO_PREFIXES = ("af_", "am_", "bf_", "bm_")      # American and British English; the language is fixed in the engine
MAX_SPEED = (0.6, 1.6)
# The only non-setting keys the dashboard writes (the voice picker). None of them reaches anything outside speech.
VOICE_KEYS = ("tts.engine", "tts.piper_voice", "tts.kokoro_voice", "tts.kokoro_speed", "tts.length_scale")


class AppBackend:
    def __init__(self, app: EVApp, local_file: Path | None = None):
        self.app = app
        self.local_file = local_file or app.cfg.root / "config" / "ev.local.toml"
        self._busy = threading.Lock()               # one dashboard action that talks or listens at a time

    # ---- live view
    def status(self) -> dict[str, Any]:
        a = self.app
        eng = a.speaker.engine if a.speaker else None
        proactive = a.proactive
        return {
            "state": a.sm.state.value, "paused": a.paused, "muted": a.muted, "display_mode": a.display_mode,
            "model": a.client.model, "voice": {"engine": getattr(eng, "name", None),
                                               "voice": getattr(eng, "voice", None) or getattr(eng, "voice_name", None)},
            "timers": [{"id": i, "seconds_left": round(s), "label": lab} for i, s, lab in a.scheduler.active_timers()],
            "reminders": [{"id": i, "text": t, "due": d} for i, t, d in a.memory.pending_reminders()],
            "proactive": None if proactive is None else {"snoozed_min": proactive.snooze_minutes_left(),
                                                         "rules": len(proactive.rules), "problems": list(proactive.problems)},
            "vault": None if a.vault is None else {"unconfirmed": len(a.vault.unconfirmed_facts())},
            "skills_problems": list(a.skills.problems),
        }

    def control(self, action: str) -> dict[str, Any]:
        a = self.app
        if action == "pause" and not a.paused or action == "resume" and a.paused:
            a.toggle_pause()
        elif action == "mute" and not a.muted or action == "unmute" and a.muted:
            a.toggle_mute()
        elif action == "kill":
            a.kill()
        elif action not in ("pause", "resume", "mute", "unmute"):
            raise DashboardError("unknown control")
        return {"ok": True, "paused": a.paused, "muted": a.muted}

    # ---- settings
    def _current(self, key: str) -> Any:
        s = ALLOWED[key]
        v = self.app.cfg.raw.get(s.section, {}).get(s.name)
        if v is None and self.app.proactive is not None and s.section == "proactive":
            g = self.app.proactive.guards
            q = g.quiet_hours
            v = {"quiet_start": q[0] if q else "", "quiet_end": q[1] if q else "", "max_spoken_per_hour": g.max_spoken_per_hour,
                 "max_idle_s": g.max_idle_s, "defer_minutes": g.defer_minutes}.get(s.name)
        return v

    def settings(self) -> dict[str, Any]:
        return {"settings": [{"key": s.key, "kind": s.kind, "label": s.label, "lo": s.lo, "hi": s.hi, "live": s.live,
                              "help": s.help, "value": self._current(s.key)} for s in SETTINGS],
                "file_only": "Anything that could widen what EV can reach or do (paths, web hosts, the builder, privacy, "
                             "policy.toml) is edited in the config files only."}

    def apply_settings(self, changes: dict[str, Any]) -> dict[str, Any]:
        try:
            clean = {k: coerce(k, v) for k, v in changes.items()}
        except SettingError as e:
            raise DashboardError(str(e), 403 if "can't be changed" in str(e) else 400) from e
        if not clean:
            raise DashboardError("nothing to change")
        localconf.update(self.local_file, clean)
        restart: list[str] = []
        for key, value in clean.items():
            if not self._apply_live(key, value):
                restart.append(key)
        return {"saved": sorted(clean), "restart_needed": sorted(restart)}

    def _apply_live(self, key: str, value: Any) -> bool:
        """Put a saved setting to work now where that is safe. False = it takes effect at the next start."""
        a = self.app
        s = ALLOWED[key]
        a.cfg.raw.setdefault(s.section, {})[s.name] = value
        if key == "llm.reply_max_sentences":
            a.brain.max_sentences = int(value)
        elif key == "llm.reply_max_sentences_detail":
            a.brain.max_sentences_detail = int(value)
        elif key == "llm.filler_after_s":
            a.brain.filler_after_s = float(value)
        elif key == "llm.temperature":
            a.client.opts["temperature"] = float(value)
        elif key == "tts.sentence_pause_ms":
            if a.speaker:
                a.speaker.pause_ms = int(value)
        elif key == "overlay.display_mode":
            a.display_mode = bool(value)
            a.memory.set_kv("display_mode", "1" if value else "0")
            a.overlay.send_display(bool(value))
        elif s.section == "conversation":
            pass                                    # read from the config at every turn
        elif key == "context.enabled":
            a.context.enabled = bool(value)
        elif s.section == "context":
            a.context.on[s.name] = bool(value)
        elif s.section == "proactive" and a.proactive is not None:
            g = a.proactive.guards
            if s.name in ("quiet_start", "quiet_end"):
                start = str(a.cfg.raw["proactive"].get("quiet_start") or (g.quiet_hours or ("22:30", "07:30"))[0])
                end = str(a.cfg.raw["proactive"].get("quiet_end") or (g.quiet_hours or ("22:30", "07:30"))[1])
                a.proactive.guards = dataclasses.replace(g, quiet_hours=(start, end))
            else:
                a.proactive.guards = dataclasses.replace(g, **{s.name: value})
        else:
            return False
        return True

    # ---- voice
    def _kokoro_voices(self) -> list[str]:
        f = self.app.cfg.path("tts", "kokoro_dir") / "voices-v1.0.bin"
        try:
            with np.load(f, allow_pickle=False) as z:
                return sorted(n for n in z.files if n.startswith(KOKORO_PREFIXES))
        except (OSError, ValueError):
            return []

    def _piper_voices(self) -> list[str]:
        d = self.app.cfg.path("tts", "piper_voice_dir")
        try:
            return sorted(p.name[:-5] for p in d.glob("*.onnx") if p.with_name(p.name + ".json").is_file())
        except OSError:
            return []

    def voices(self) -> dict[str, Any]:
        t = self.app.cfg["tts"]
        return {"engine": t["engine"], "piper": self._piper_voices(), "kokoro": self._kokoro_voices(),
                "piper_voice": t["piper_voice"], "kokoro_voice": t["kokoro_voice"], "kokoro_speed": t.get("kokoro_speed", 1.0),
                "length_scale": t.get("length_scale", 1.0), "speed_range": list(MAX_SPEED), "speaking": bool(self.app.speaker)}

    def set_voice(self, engine: str, voice: str, speed: float, preview: bool, save: bool) -> dict[str, Any]:
        a = self.app
        if a.speaker is None:
            raise DashboardError("EV has no voice running (check the log)", 409)
        if engine not in ("piper", "kokoro"):
            raise DashboardError("engine must be piper or kokoro")
        if voice not in (self._piper_voices() if engine == "piper" else self._kokoro_voices()):
            raise DashboardError("that voice isn't installed")
        if not MAX_SPEED[0] <= speed <= MAX_SPEED[1]:
            raise DashboardError(f"speed must be between {MAX_SPEED[0]} and {MAX_SPEED[1]}")
        data = copy.deepcopy(a.cfg.raw)
        t = data["tts"]
        t["engine"] = engine
        if engine == "piper":
            t["piper_voice"], t["length_scale"] = voice, round(1 / speed, 3)
        else:
            t["kokoro_voice"], t["kokoro_speed"] = voice, round(speed, 3)
        try:
            new = make_engine(Config(data, a.cfg.root), engine)
            a.speaker.swap_engine(new)              # loads and tries the voice first; on any failure the old one keeps speaking
        except Exception as e:                      # noqa: BLE001
            log.warning("voice change failed: %s", e)
            raise DashboardError(f"couldn't switch to that voice ({e}); the current voice is unchanged", 422) from e
        a.cfg.raw["tts"].update(t)
        if save:
            changes: dict[str, Any] = {"tts.engine": engine}
            if engine == "piper":
                changes |= {"tts.piper_voice": voice, "tts.length_scale": t["length_scale"]}
            else:
                changes |= {"tts.kokoro_voice": voice, "tts.kokoro_speed": t["kokoro_speed"]}
            assert set(changes) <= set(VOICE_KEYS)
            localconf.update(self.local_file, changes)
        if preview and a.sm.state is State.IDLE:
            a.speaker.say("Hi, this is how I sound now.")
        return {"engine": engine, "voice": voice, "saved": save}

    # ---- chat model
    def models(self) -> dict[str, Any]:
        a = self.app
        try:
            r = a.client.http.get("/api/tags")
            r.raise_for_status()
            found = r.json().get("models", [])
        except Exception as e:                      # noqa: BLE001
            raise DashboardError(f"Ollama isn't answering ({type(e).__name__})", 503) from e
        budget = float(a.cfg.get("llm", "vram_budget_gb", 3.2))
        out = [{"name": m["name"], "gb": round(m.get("size", 0) / 1e9, 1),
                "too_big": m.get("size", 0) / 1e9 > budget} for m in found if isinstance(m, dict) and "name" in m]
        return {"current": a.client.model, "models": out, "vram_budget_gb": budget}

    def set_model(self, model: str, save: bool) -> dict[str, Any]:
        a = self.app
        if model not in {m["name"] for m in self.models()["models"]}:
            raise DashboardError("Ollama doesn't have that model")
        a.client.model = model
        a.cfg.raw["llm"]["model"] = model
        if save:
            localconf.update(self.local_file, {"llm.model": model})
        threading.Thread(target=a.brain.refresh_prompt_cache, name="rewarm", daemon=True).start()
        return {"model": model, "saved": save}

    def snooze(self, minutes: float) -> dict[str, Any]:
        if self.app.proactive is None:
            raise DashboardError("proactive is off", 409)
        if not 0 <= minutes <= 720:
            raise DashboardError("minutes must be between 0 and 720")
        self.app.proactive.snooze(minutes)
        return {"snoozed_min": self.app.proactive.snooze_minutes_left()}

    # ---- skills
    def skills(self) -> dict[str, Any]:
        store = self.app.skills
        out = []
        for n in store.names():
            s = store.get(n)
            assert s is not None
            out.append({"name": s.name, "description": s.description, "triggers": s.triggers, "say": s.say, "params": s.params,
                        "steps": [{"tool": x.tool, "args": x.args, "optional": x.optional} for x in s.steps],
                        "reads_outside": s.reads_outside})
        return {"skills": out, "problems": list(store.problems),
                "tools": sorted(self.app.registry.names())}

    def _exclusive(self) -> None:
        """Skill runs and deletions can ask a spoken question, so they never overlap a conversation or each other."""
        if not self.app._turn_lock.acquire(blocking=False):
            raise DashboardError("EV is busy with something else; try again in a moment", 409)

    def run_skill(self, name: str) -> dict[str, Any]:
        if self.app.skills.get(name) is None:
            raise DashboardError("no such skill", 404)
        self._exclusive()
        try:
            ok, message = self.app.brain.run_skill_for_user(name)
        finally:
            self.app._turn_lock.release()
        return {"ok": ok, "message": message}

    def save_skill(self, spec: dict[str, Any]) -> dict[str, Any]:
        store = self.app.skills
        name = canonical(str(spec.get("name", "")))
        old = store.get(name)
        raw = {"name": name, "description": spec.get("description", ""), "triggers": spec.get("triggers", []),
               "say": spec.get("say", ""), "params": spec.get("params", {}), "steps": spec.get("steps", []),
               "instructions": old.instructions_file if old else ""}
        if not isinstance(raw["triggers"], list) or not isinstance(raw["steps"], list) or not isinstance(raw["params"], dict):
            raise DashboardError("triggers and steps must be lists and params an object")
        try:
            skill = validate(raw, old.instructions if old else "")
            missing = [s.tool for s in skill.steps if self.app.registry.get(s.tool) is None]
            if missing:
                raise SkillError(f"there is no tool called {missing[0]}")
            store.save(skill, overwrite=old is not None)
        except SkillError as e:
            raise DashboardError(str(e)) from e
        return {"saved": skill.name}

    def delete_skill(self, name: str) -> dict[str, Any]:
        if self.app.skills.get(name) is None:
            raise DashboardError("no such skill", 404)
        self._exclusive()
        try:
            res = self.app.gate.check("manage_skill", {"action": "delete", "name": canonical(name)})   # CONFIRM: EV asks out loud
            if not res.allowed:
                return {"deleted": False, "message": res.reason}
            self.app.skills.delete(name)
        except SkillError as e:
            raise DashboardError(str(e)) from e
        finally:
            self.app._turn_lock.release()
        return {"deleted": True}

    # ---- memory
    def facts(self) -> dict[str, Any]:
        v = self.app.vault
        if v is None:
            return {"facts": {}, "unconfirmed": [], "vault": False}
        return {"vault": True, "facts": v.profile_facts(200),
                "unconfirmed": [{"id": u.id, "key": u.key, "value": u.value} for u in v.unconfirmed_facts()]}

    def search(self, query: str, when: str) -> dict[str, Any]:
        idx = self.app.index
        if idx is None:
            raise DashboardError("the vault is off", 409)
        hits = idx.search(query, when or None, k=8) if query.strip() or when else []
        return {"hits": [{"path": h.path, "heading": h.heading, "text": h.text[:400], "date": h.date} for h in hits]}

    def _gated(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        self._exclusive()
        try:
            entry = self.app.brain._execute(tool, args, None, source="dashboard")
        finally:
            self.app._turn_lock.release()
        return {"ok": bool(entry.get("ok")), "message": str(entry.get("message", ""))}

    def forget(self, key: str) -> dict[str, Any]:
        # Not the fast path and not the user's recent words, so provenance makes EV ask for a spoken yes first.
        return self._gated("forget_fact", {"key": key[:60]})

    def keep(self, ident: str) -> dict[str, Any]:
        return self._gated("keep_fact", {"id": ident[:16]})     # CONFIRM in policy.toml: a spoken readback, never a click

    def policy_text(self) -> dict[str, Any]:
        p = self.app.cfg.path("general", "policy_file")
        return {"path": p.name, "read_only": True, "text": p.read_text(encoding="utf-8")}
