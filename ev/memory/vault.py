"""The vault: memory as a folder of Markdown files (an Obsidian vault). Obsidian doesn't have to be running.

EV writes only inside <vault>/<ev_folder> (default EV/): Facts.md, People/<name>.md, Projects/<name>.md, Daily/<date>.md.
The rest of the vault is read-only to it (see ev/memory/vault_index.py for reading).

Trust: the `source:` label in a fact line is for people. Anything that can write to the vault (a sync client, an Obsidian
plugin, pasted text) can type "(source: user said)", so EV never believes the file about itself. When `add_fact` writes a
line it also records the line's SHA-256 in SQLite (`fact_records`, outside the vault). A line reaches the system prompt only
if its exact text matches a record. A line without one (a hand edit, or anything else) is an "unconfirmed fact": it stays an
ordinary note, reachable only through `recall` wrapped as untrusted, until the user confirms it out loud.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ev.core.config import Config
from ev.core.log import get_logger
from ev.memory.store import MemoryStore, key_tokens, norm_key
from ev.safety.policy import Policy, clean_text

log = get_logger("vault")

FACTS_FILE = "Facts.md"

# "- key: value (source: user said, 2026-11-02)". The label tells people where a fact came from; EV doesn't read it back.
_LINE = re.compile(r"^- (?P<key>[a-z0-9_]{1,60}): (?P<value>.+) \(source: (?P<source>[^()]+?), (?P<date>\d{4}-\d{2}-\d{2})\)$")
_PAGE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,39}")

_README = """# EV

This folder belongs to EV, the voice assistant. It reads your whole vault to answer "what did I write about...", but it only
ever writes in here:

- `Facts.md`: things you told EV to remember.
- `People/` and `Projects/`: facts about a person or a project.
- `Daily/`: a short summary of what you talked about each day.

EV trusts a fact line only if it wrote that exact line itself. If you edit or add a line by hand it becomes an *unconfirmed
fact*: it stays here as a normal note, and EV ignores it as a fact until you say "review my unconfirmed facts" and agree.
"""

_FACTS_HEAD = """---
type: ev-facts
created: {today}
---
# What EV knows

Facts EV saved from what you told it. Edit freely, but a line you change by hand stops counting as a fact until you confirm it
by voice ("review my unconfirmed facts").

"""


class VaultError(ValueError):
    """Something EV can say out loud about why it didn't write to the vault."""


@dataclass(frozen=True)
class Fact:
    key: str
    value: str
    source: str
    file: str


@dataclass(frozen=True)
class Unconfirmed:
    id: str          # first 12 hex digits of the line's hash; changes whenever the line does
    key: str
    value: str
    line: str


def _sha(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


class Vault:
    def __init__(self, root: Path | str, memory: MemoryStore, *, ev_folder: str = "EV", fact_max_chars: int = 200,
                 injection_phrases: list[str] | tuple[str, ...] = (), policy: Policy | None = None,
                 now: Callable[[], dt.datetime] = dt.datetime.now):
        self.root = Path(root).expanduser()
        self.ev_dir = self.root / ev_folder
        self.memory = memory
        self.fact_max = int(fact_max_chars)
        self._inj = [re.compile(p, re.I) for p in injection_phrases]
        self.policy = policy
        self.now = now
        self._lock = threading.RLock()

    @classmethod
    def from_config(cls, cfg: Config, memory: MemoryStore, policy: Policy) -> Vault | None:
        v = cfg.raw.get("vault", {})
        if not v.get("enabled", False):
            return None
        return cls(v["path"], memory, ev_folder=str(v.get("ev_folder", "EV")),
                   fact_max_chars=int(cfg.get("memory", "fact_max_chars", 200)),
                   injection_phrases=policy.d.get("injection", {}).get("phrases", []), policy=policy)

    # ---- layout
    def ensure(self) -> None:
        """Create the EV folder, its README and an empty Facts.md on first run. Never touches anything outside the EV folder."""
        with self._lock:
            for sub in ("", "Daily", "People", "Projects"):
                self._path(sub).mkdir(parents=True, exist_ok=True)
            self._create_if_missing("README.md", _README)
            self._create_if_missing(FACTS_FILE, _FACTS_HEAD.format(today=f"{self.now():%Y-%m-%d}"))

    def _create_if_missing(self, rel: str, text: str) -> None:
        path = self._path(rel)
        if not path.exists():
            self._write_text(path, text)

    def _path(self, rel: str) -> Path:
        """A path inside the EV folder. Refuses anything that resolves outside it (a `..`, a symlink or junction) and anything
        the policy's write_dirs doesn't cover."""
        p = self.ev_dir / rel if rel else self.ev_dir
        base = Path(os.path.realpath(self.ev_dir))
        real = Path(os.path.realpath(p))
        if real != base and base not in real.parents:
            raise VaultError("I can only write inside my own folder in the vault.")
        if self.policy is not None and not self.policy.path_in_dirs(str(p), self.policy.write_dirs):
            raise VaultError("My policy doesn't let me write there.")
        return p

    # ---- file helpers
    @staticmethod
    def _read_lines(path: Path) -> list[str]:
        text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        return lines

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        tmp = path.with_name(path.name + ".evtmp")
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, path)

    def _write_lines(self, path: Path, lines: list[str]) -> None:
        self._write_text(path, "\n".join(lines) + "\n")

    # ---- sanitising
    def _clean_line(self, s: str, limit: int) -> str:
        """One plain line: no tags, wikilinks, code ticks or leading Markdown structure, so a stored value can't become a
        heading, a list item, a link, fake front matter or a fake source label."""
        s = clean_text(s)
        s = re.sub(r"<[^>\n]{0,200}>", " ", s)
        s = s.replace("[[", " ").replace("]]", " ").replace("`", "'")
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"^[#>*+|\-\s]+", "", s)
        s = s.replace("(source:", "( source:")
        return s[:limit].rstrip()

    def clean_fact(self, key: str, value: str) -> tuple[str, str]:
        k = norm_key(clean_text(key))[:60]
        if not k:
            raise VaultError("I need a name for that fact.")
        v = self._clean_line(value, self.fact_max)
        if not v:
            raise VaultError("I need something to remember.")
        raw = clean_text(f"{key} {value}")
        if any(rx.search(raw) or rx.search(k) or rx.search(v) for rx in self._inj):
            log.warning("refused to save a fact that reads like an instruction: key=%r", k)
            raise VaultError("I won't save that.")
        return k, v

    # ---- pages
    def _target(self, about: str) -> str:
        about = clean_text(about or "").strip()
        if not about:
            return FACTS_FILE
        if not _PAGE_NAME.fullmatch(about):
            raise VaultError("I can't use that name for a page.")
        return f"Projects/{about}.md" if (self.ev_dir / "Projects" / f"{about}.md").exists() else f"People/{about}.md"

    def _ensure_page(self, rel: str) -> Path:
        path = self._path(rel)
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel == FACTS_FILE:
            self._write_text(path, _FACTS_HEAD.format(today=f"{self.now():%Y-%m-%d}"))
        else:
            kind, name = rel.removesuffix(".md").split("/", 1)
            who = f"people: [{name}]" if kind == "People" else f"project: {name}"
            self._write_text(path, f"---\ndate: {self.now():%Y-%m-%d}\n{who}\nsource: EV\n---\n# {name}\n\n"
                                   "Part of [[Facts]]. EV trusts a line here only if it wrote that exact line.\n\n")
        return path

    # ---- facts
    def add_fact(self, key: str, value: str, source: str, about: str = "") -> Fact:
        """Write one fact line and record its hash. The only way a fact enters the vault (remember_fact, the migrations,
        keep_fact and later the dashboard all come through here)."""
        k, v = self.clean_fact(key, value)
        rel = self._target(about)
        return self._commit(rel, k, v, source)

    def _commit(self, rel: str, k: str, v: str, source: str, drop_sha: str | None = None) -> Fact:
        with self._lock:
            path = self._ensure_page(rel)
            line = f"- {k}: {v} (source: {source}, {self.now():%Y-%m-%d})"
            prev = next((r for r in self.memory.fact_records(rel) if r[0] == k), None)
            drop = {s for s in (prev[4] if prev else None, drop_sha) if s}
            lines = [ln for ln in self._read_lines(path) if _sha(ln) not in drop]
            lines.append(line)
            self.memory.upsert_fact_record(rel, k, v, source, _sha(line))
            try:
                self._write_lines(path, lines)
            except OSError as e:
                if prev:
                    self.memory.upsert_fact_record(rel, prev[0], prev[1], prev[2], prev[4], prev[3])
                else:
                    self.memory.delete_fact_record(rel, k)
                log.error("vault write failed: %s", e)
                raise VaultError("I couldn't write to the vault.") from e
            return Fact(k, v, source, rel)

    def forget_fact(self, key: str, about: str = "") -> bool:
        """Remove a fact EV wrote, and its record (so pasting the old line back doesn't bring it back). Only a line EV
        recorded can be forgotten this way; an unconfirmed line is an ordinary note for the user to edit."""
        rel = self._target(about)
        with self._lock:
            path = self.ev_dir / rel
            if not path.exists():
                return False
            recs = {r[0]: r for r in self.memory.fact_records(rel)}
            k = norm_key(clean_text(key))
            match = k if k in recs else self._fuzzy(k, list(recs))
            if match is None:
                return False
            sha = recs[match][4]
            self.memory.delete_fact_record(rel, match)
            try:
                self._write_lines(self._path(rel), [ln for ln in self._read_lines(path) if _sha(ln) != sha])
            except OSError as e:
                log.error("vault write failed: %s", e)
                raise VaultError("I stopped using that fact, but I couldn't tidy the file.") from e
            return True

    @staticmethod
    def _fuzzy(k: str, keys: list[str]) -> str | None:
        """A loose match ("dog name" for "dog_s_name"), but only when exactly one stored key fits."""
        q = key_tokens(k)
        if not q:
            return None
        hits = [c for c in keys if (t := key_tokens(c)) and (q <= t or t <= q)]
        return hits[0] if len(hits) == 1 else None

    def profile_facts(self, limit: int = 16) -> dict[str, str]:
        """The facts that may go into the system prompt: lines of Facts.md whose exact text matches a record EV wrote.
        Key, value and source come from the record; nothing is believed because the file says so."""
        try:
            present = {_sha(ln) for ln in self._read_lines(self.ev_dir / FACTS_FILE) if ln.startswith("- ")}
        except OSError:
            return {}
        out: dict[str, str] = {}
        for key, value, _source, _ts, sha in self.memory.fact_records(FACTS_FILE):      # newest first
            if sha in present and key not in out:
                out[key] = value
                if len(out) >= limit:
                    break
        return out

    def unconfirmed_facts(self) -> list[Unconfirmed]:
        """Bullet lines of Facts.md that EV has no record of: hand edits, forged labels, anything else."""
        try:
            lines = self._read_lines(self.ev_dir / FACTS_FILE)
        except OSError:
            return []
        known = {r[4] for r in self.memory.fact_records(FACTS_FILE)}
        out: list[Unconfirmed] = []
        for ln in lines:
            if not ln.startswith("- ") or _sha(ln) in known:
                continue
            body = re.sub(r"\s*\(source:[^)]*\)\s*$", "", ln[2:].strip())
            if not body:
                continue
            k, sep, v = body.partition(":")
            if not sep:
                k, v = " ".join(body.split()[:4]), body
            out.append(Unconfirmed(_sha(ln)[:12], norm_key(k.strip()) or "note", v.strip(), ln))
        return out

    def describe_unconfirmed(self, ident: str) -> str:
        """What EV reads out before keeping a hand-written line: cleaned, short, and clearly a quote."""
        item = next((u for u in self.unconfirmed_facts() if u.id == ident), None)
        if item is None:
            return "keep a note that is no longer there"
        return f"keep your note that {item.key.replace('_', ' ')} is {self._clean_line(item.value, 120)}"

    def keep_unconfirmed(self, ident: str) -> Fact:
        """The user said yes out loud: record this exact line as a fact (sanitised again on the way in)."""
        with self._lock:
            item = next((u for u in self.unconfirmed_facts() if u.id == ident), None)
            if item is None:
                raise VaultError("I can't find that note any more.")
            k, v = self.clean_fact(item.key, item.value)
            return self._commit(FACTS_FILE, k, v, "user confirmed", drop_sha=_sha(item.line))

    # ---- daily summaries (append-only)
    def append_daily(self, text: str, when: dt.datetime | None = None) -> Path | None:
        when = when or self.now()
        line = self._clean_line(text, 700)
        if not line:
            return None
        rel = f"Daily/{when:%Y-%m-%d}.md"
        try:
            with self._lock:
                path = self._path(rel)
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    self._write_text(path, f"---\ndate: {when:%Y-%m-%d}\nsource: EV conversation summaries\n---\n"
                                           f"# {when:%Y-%m-%d}\n\n")
                with path.open("a", encoding="utf-8", newline="\n") as f:
                    f.write(f"- {when:%H:%M} {line}\n")
        except (OSError, VaultError) as e:
            log.warning("daily summary not written: %s", e)
            return None
        return path
