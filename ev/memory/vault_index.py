"""Search over the whole vault: SQLite FTS5, ranked with bm25. The Markdown files are the source of truth; this index is a
cache that can be thrown away and rebuilt at any time (`python -m ev.memory.vault_index --rebuild`).

Everything it returns is the user's own notes, which may contain pasted web text, so `recall` hands results to the model
wrapped as untrusted data. Nothing here writes to the vault.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ev.core.log import get_logger

log = get_logger("vault_index")

_SCHEMA = """
-- lo..hi: the rowids of this note's chunks. Deleting FTS5 rows by the (unindexed) path is a full table scan, which made a
-- rebuild quadratic; a rowid range is direct.
CREATE TABLE IF NOT EXISTS docs (path TEXT PRIMARY KEY, mtime_ns INTEGER NOT NULL, size INTEGER NOT NULL, date TEXT NOT NULL,
                                 lo INTEGER NOT NULL DEFAULT 0, hi INTEGER NOT NULL DEFAULT -1);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(path UNINDEXED, heading, body, tokenize='porter unicode61');
"""

CHUNK_CHARS = 800
SNIPPET_CHARS = 300
MAX_QUERY_TOKENS = 12
_FRONT = re.compile(r"---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)", re.S)
_YAML_RISK = re.compile(r"(^|[\s\[,:{])[&*][A-Za-z0-9_-]|!!|!<")        # anchors, aliases and tags: expansion tricks
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_STOP = frozenset("""a an the and or but if of to in on at by for with from as is are was were be been am it its this that these
those i me my mine we our you your he she they them his her their there here so do does did have has had what when where which
who whom why how about into over up down out say said tell told write wrote note noted notes mention mentioned talk talked
please can could would should will just""".split())
_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
     "december"], 1)}


@dataclass(frozen=True)
class Hit:
    path: str        # relative to the vault, forward slashes
    heading: str
    text: str
    date: str        # YYYY-MM-DD
    score: float     # bm25 as returned by SQLite: lower (more negative) is a better match


def _is_link(path: str) -> bool:
    return os.path.islink(path) or getattr(os.path, "isjunction", lambda _p: False)(path)


def split_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """(front matter as a dict, the rest). Front matter is only parsed when it is small and has no YAML anchors, aliases or
    tags, because a note anyone can edit must not be able to make the indexer expand a billion-laughs document."""
    m = _FRONT.match(text) if text.startswith("---") else None
    if not m:
        return {}, text
    raw, body = m.group(1), text[m.end():]
    if len(raw) > 4096 or _YAML_RISK.search(raw):
        return {}, body
    try:
        import yaml

        meta = yaml.safe_load(raw)
    except Exception:                                  # noqa: BLE001  (bad YAML, or PyYAML not installed)
        d = re.search(r"^date:\s*(\S+)", raw, re.M)
        return ({"date": d.group(1)} if d else {}), body
    return (meta if isinstance(meta, dict) else {}), body


def note_date(meta: dict[str, object], rel: str, mtime_ns: int) -> str:
    d = meta.get("date")
    if isinstance(d, dt.date):                         # datetime is a date too
        return d.isoformat()[:10]
    if isinstance(d, str) and _DATE.match(d.strip()):
        return d.strip()[:10]
    m = _DATE.search(rel.rsplit("/", 1)[-1])
    if m:
        return m.group(0)
    return dt.datetime.fromtimestamp(mtime_ns / 1e9).strftime("%Y-%m-%d")


def make_chunks(title: str, body: str) -> list[tuple[str, str]]:
    """(heading, text) pieces of at most CHUNK_CHARS, split on headings first and on lines after that."""
    sections: list[tuple[str, list[str]]] = []
    heading = title
    buf: list[str] = []
    for line in body.split("\n"):
        m = re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            sections.append((heading, buf))
            heading, buf = (m.group(1).strip()[:120] or title), []
        else:
            buf.append(line)
    sections.append((heading, buf))
    out: list[tuple[str, str]] = []
    for head, lines in sections:
        cur = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            while len(line) > CHUNK_CHARS:                       # one enormous line: cut it on whitespace
                cut = line.rfind(" ", 0, CHUNK_CHARS)
                cut = cut if cut > 0 else CHUNK_CHARS
                if cur:
                    out.append((head, cur))
                    cur = ""
                out.append((head, line[:cut].strip()))
                line = line[cut:].strip()
            if cur and len(cur) + len(line) + 1 > CHUNK_CHARS:
                out.append((head, cur))
                cur = ""
            cur = f"{cur}\n{line}" if cur else line
        if cur:
            out.append((head, cur))
    return out


def query_tokens(query: str) -> list[str]:
    """Plain words only. Quoting each one later means a query can't smuggle in FTS operators (NEAR, OR, column filters...)."""
    seen: list[str] = []
    for w in re.findall(r"[^\W_]+", query.lower()):
        if w not in _STOP and (len(w) > 1 or w.isdigit()) and w not in seen:
            seen.append(w)
    return seen[:MAX_QUERY_TOKENS]


def parse_when(when: str | None, today: dt.date) -> tuple[str, str] | None:
    """A date range (inclusive ISO dates) for words like "yesterday", "last week", "october", or None if not understood."""
    if not when or not when.strip():
        return None
    w = re.sub(r"\s+", " ", when.strip().lower())
    iso = lambda d: d.isoformat()                      # noqa: E731
    if w in ("today", "now", "tonight"):
        return iso(today), iso(today)
    if w in ("yesterday", "last night"):
        y = today - dt.timedelta(days=1)
        return iso(y), iso(y)
    monday = today - dt.timedelta(days=today.weekday())
    if w == "this week":
        return iso(monday), iso(today)
    if w == "last week":
        return iso(monday - dt.timedelta(days=7)), iso(monday - dt.timedelta(days=1))
    if w == "this month":
        return iso(today.replace(day=1)), iso(today)
    if w == "last month":
        end = today.replace(day=1) - dt.timedelta(days=1)
        return iso(end.replace(day=1)), iso(end)
    if m := re.fullmatch(r"(?:last|past) (\d{1,3}) days", w):
        return iso(today - dt.timedelta(days=int(m.group(1)))), iso(today)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", w):
        return w, w
    if m := re.fullmatch(r"(\d{4})-(\d{2})", w):
        return _month(int(m.group(1)), int(m.group(2)))
    if m := re.fullmatch(r"([a-z]+)(?: (\d{4}))?", w):
        mon = _MONTHS.get(m.group(1))
        if mon:
            year = int(m.group(2)) if m.group(2) else today.year - (1 if mon > today.month else 0)
            return _month(year, mon)
    return None


def _month(year: int, month: int) -> tuple[str, str] | None:
    try:
        first = dt.date(year, month, 1)
    except ValueError:
        return None
    nxt = dt.date(year + (month == 12), month % 12 + 1, 1)
    return first.isoformat(), (nxt - dt.timedelta(days=1)).isoformat()


class VaultIndex:
    def __init__(self, root: Path | str, db_path: Path | str, *, max_note_kb: int = 512, poll_s: float = 15,
                 now: Callable[[], dt.datetime] = dt.datetime.now):
        self.root = Path(root).expanduser()
        self.max_bytes = int(max_note_kb) * 1024
        self.poll_s = float(poll_s)
        self.now = now
        self._lock = threading.RLock()
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        if "lo" not in {r[1] for r in self.db.execute("PRAGMA table_info(docs)")} and self.doc_count_raw():
            self.db.executescript("DROP TABLE docs; DROP TABLE chunks;")      # an index from before lo/hi: it is only a cache
        self.db.executescript(_SCHEMA)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def close(self) -> None:
        self.stop()
        with self._lock:
            self.db.close()

    # ---- keeping the index fresh
    def _scan(self) -> dict[str, tuple[int, int]]:
        found: dict[str, tuple[int, int]] = {}
        if not self.root.is_dir():
            return found
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "node_modules"
                           and not _is_link(os.path.join(dirpath, d))]       # never follow a link out of the vault
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                if not fn.lower().endswith(".md") or _is_link(p):
                    continue
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                if st.st_size <= self.max_bytes:
                    found[os.path.relpath(p, self.root).replace("\\", "/")] = (st.st_mtime_ns, st.st_size)
        return found

    def refresh(self) -> tuple[int, int, int]:
        """Index notes that are new or changed, drop ones that are gone. Returns (added, updated, removed)."""
        found = self._scan()
        with self._lock:
            cur = {r[0]: (r[1], r[2]) for r in self.db.execute("SELECT path, mtime_ns, size FROM docs")}
        gone = [p for p in cur if p not in found]
        changed = [p for p, st in found.items() if cur.get(p) != st]
        added = updated = 0
        for n, p in enumerate([*gone, *changed]):
            with self._lock:                       # one note at a time, so a recall during a big first index isn't kept waiting
                if p in found:
                    if self._index_file(p, *found[p]):
                        added, updated = (added + 1, updated) if p not in cur else (added, updated + 1)
                else:
                    self._drop(p)
                if n % 100 == 99:
                    self.db.commit()
        with self._lock:
            self.db.commit()
        if gone or changed:
            log.info("vault index: %d added, %d updated, %d removed", added, updated, len(gone))
        return added, updated, len(gone)

    def rebuild(self) -> int:
        with self._lock:
            self.db.execute("DELETE FROM chunks")
            self.db.execute("DELETE FROM docs")
            self.db.commit()
        self.refresh()
        return self.doc_count()

    def doc_count(self) -> int:
        with self._lock:
            return int(self.db.execute("SELECT COUNT(*) FROM docs").fetchone()[0])

    def doc_count_raw(self) -> int:
        try:
            return int(self.db.execute("SELECT COUNT(*) FROM docs").fetchone()[0])
        except sqlite3.OperationalError:
            return 0

    def _drop(self, rel: str) -> None:
        row = self.db.execute("SELECT lo, hi FROM docs WHERE path = ?", (rel,)).fetchone()
        if row:
            self.db.execute("DELETE FROM chunks WHERE rowid BETWEEN ? AND ?", row)
            self.db.execute("DELETE FROM docs WHERE path = ?", (rel,))

    def _index_file(self, rel: str, mtime_ns: int, size: int) -> bool:
        try:
            text = (self.root / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        meta, body = split_frontmatter(text)
        title = next((m.group(1).strip() for ln in body.split("\n") if (m := re.match(r"^#\s+(.*)", ln))),
                     rel.rsplit("/", 1)[-1].removesuffix(".md"))
        self._drop(rel)
        rows = [(rel, h, t) for h, t in make_chunks(title, body)]
        lo, hi = 0, -1
        if rows:
            self.db.executemany("INSERT INTO chunks(path, heading, body) VALUES (?,?,?)", rows)
            hi = int(self.db.execute("SELECT last_insert_rowid()").fetchone()[0])
            lo = hi - len(rows) + 1
            # The range is only valid if the new rows got consecutive ids. If SQLite ever does otherwise, delete by path (slow
            # but correct) rather than a range that could take another note's chunks with it.
            if self.db.execute("SELECT COUNT(*) FROM chunks WHERE rowid BETWEEN ? AND ?", (lo, hi)).fetchone()[0] != len(rows):
                log.warning("vault index: chunk ids for %s are not consecutive; rebuilding the slow way", rel)
                self.db.execute("DELETE FROM chunks WHERE path = ?", (rel,))
                self.db.executemany("INSERT INTO chunks(path, heading, body) VALUES (?,?,?)", rows)
                hi = int(self.db.execute("SELECT last_insert_rowid()").fetchone()[0])
                lo = hi - len(rows) + 1
        self.db.execute("INSERT INTO docs(path, mtime_ns, size, date, lo, hi) VALUES (?,?,?,?,?,?)",
                        (rel, mtime_ns, size, note_date(meta, rel, mtime_ns), lo, hi))
        return True

    # ---- searching
    def search(self, query: str, when: str | None = None, k: int = 4) -> list[Hit]:
        """Best-matching snippets, at most two per note. With no usable words but a time range ("what did we talk about
        yesterday"), the newest notes in that range."""
        toks = query_tokens(query)
        rng = parse_when(when, self.now().date())
        if not toks and rng is None:
            return []
        where, params = [], []
        if toks:
            where.append("chunks MATCH ?")
            params.append(" OR ".join(f'"{t}"' for t in toks))
        if rng:
            where.append("docs.date BETWEEN ? AND ?")
            params += list(rng)
        order = "bm25(chunks)" if toks else "docs.date DESC, chunks.rowid"
        # Only fixed fragments are formatted into the SQL; the query words and dates travel as ? parameters.
        sql = (f"SELECT chunks.path, chunks.heading, chunks.body, docs.date, {'bm25(chunks)' if toks else '0.0'} "  # noqa: S608
               f"FROM chunks JOIN docs ON docs.path = chunks.path WHERE {' AND '.join(where)} ORDER BY {order} LIMIT ?")
        with self._lock:
            rows = self.db.execute(sql, [*params, max(k, 1) * 4]).fetchall()
        hits: list[Hit] = []
        per_note: dict[str, int] = {}
        for path, heading, body, date, score in rows:
            if per_note.get(path, 0) >= 2:
                continue
            per_note[path] = per_note.get(path, 0) + 1
            hits.append(Hit(path, heading, _snippet(body), date, float(score)))
            if len(hits) >= k:
                break
        return hits

    # ---- background polling (no watchdog dependency)
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="vault-index", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=3.0)

    def _loop(self) -> None:
        while True:
            try:
                self.refresh()
            except Exception:                          # noqa: BLE001  (a bad note must not kill the poller)
                log.exception("vault index refresh failed")
            if self._stop.wait(self.poll_s):
                return


def _snippet(body: str) -> str:
    s = re.sub(r"\s+", " ", body).strip()
    if len(s) <= SNIPPET_CHARS:
        return s
    cut = s.rfind(" ", 0, SNIPPET_CHARS)
    return s[:cut if cut > 0 else SNIPPET_CHARS] + "..."


def main(argv: list[str] | None = None) -> int:
    from ev.core.config import load_config

    ap = argparse.ArgumentParser(prog="python -m ev.memory.vault_index")
    ap.add_argument("--rebuild", action="store_true", help="throw the index away and rebuild it from the Markdown files")
    ap.add_argument("--query", help="search the index and print the hits")
    ap.add_argument("--when", help="limit --query to a time range: today, yesterday, last week, october, 2026-10-30...")
    a = ap.parse_args(argv)
    cfg = load_config()
    v = cfg["vault"]
    idx = VaultIndex(v["path"], cfg.data_path(v["index_file"]), max_note_kb=int(v.get("max_note_kb", 512)))
    if a.rebuild:
        print(f"rebuilt: {idx.rebuild()} notes indexed")
    else:
        added, updated, removed = idx.refresh()
        print(f"refreshed: {added} added, {updated} updated, {removed} removed; {idx.doc_count()} notes indexed")
    if a.query is not None:
        for h in idx.search(a.query, a.when, int(v.get("recall_k", 4))):
            print(f"{h.date}  {h.path}  [{h.heading}]  {h.text}")
    idx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
