"""SQLite memory: facts, notes, reminders, rolling summary. No vector DB (RAM is tight)."""
from __future__ import annotations

import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    key TEXT PRIMARY KEY, value TEXT NOT NULL, updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, due REAL NOT NULL,
    fired INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
-- What EV itself wrote into the vault's fact files. A vault line is only trusted if its exact text hashes to a row here.
CREATE TABLE IF NOT EXISTS fact_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT, file TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
    source TEXT NOT NULL, ts REAL NOT NULL, line_sha256 TEXT NOT NULL, UNIQUE(file, key)
);
CREATE INDEX IF NOT EXISTS fact_records_sha ON fact_records(line_sha256);
"""


def _norm(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


norm_key = _norm


_STOP = {"s", "my", "the", "a", "an", "of", "user", "users"}


def _tokens(key: str) -> set[str]:
    return {t for t in _norm(key).split("_") if t and t not in _STOP}


key_tokens = _tokens


class MemoryStore:
    def __init__(self, db_path: Path | str):
        self._lock = threading.Lock()
        self._closed = False
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.executescript(_SCHEMA)
        self.db.commit()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self.db.close()

    def _read(self, sql: str, params: tuple = (), one: bool = False):  # noqa: ANN202
        with self._lock:
            if self._closed:
                return None if one else []
            cur = self.db.execute(sql, params)
            return cur.fetchone() if one else cur.fetchall()

    # facts
    def remember(self, key: str, value: str) -> str:
        k = _norm(key)
        if not k or not value.strip():
            raise ValueError("empty fact")
        with self._lock:
            if self._closed:
                return k
            self.db.execute(
                "INSERT INTO facts(key,value,updated) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated=excluded.updated",
                (k, value.strip()[:300], time.time()),
            )
            self.db.commit()
        return k

    def recall(self, key: str) -> str | None:
        row = self._read("SELECT value FROM facts WHERE key=?", (_norm(key),), one=True)
        return row[0] if row else None

    def forget(self, key: str) -> bool:
        """Delete a fact by key. Falls back to a fuzzy match, but only when exactly one stored key fits."""
        with self._lock:
            if self._closed:
                return False
            cur = self.db.execute("DELETE FROM facts WHERE key=?", (_norm(key),))
            if cur.rowcount:
                self.db.commit()
                return True
            q = _tokens(key)
            if not q:
                return False
            hits = [r[0] for r in self.db.execute("SELECT key FROM facts").fetchall()
                    if (t := _tokens(r[0])) and (q <= t or t <= q)]
            if len(hits) != 1:
                return False
            self.db.execute("DELETE FROM facts WHERE key=?", (hits[0],))
            self.db.commit()
            return True

    def facts(self, limit: int = 50) -> dict[str, str]:
        rows = self._read("SELECT key,value FROM facts ORDER BY updated DESC LIMIT ?", (limit,))
        return dict(rows)

    # records of the fact lines EV wrote into the vault (see ev/memory/vault.py)
    def upsert_fact_record(self, file: str, key: str, value: str, source: str, line_sha256: str,
                           ts: float | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            self.db.execute(
                "INSERT INTO fact_records(file,key,value,source,ts,line_sha256) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(file,key) DO UPDATE SET value=excluded.value, source=excluded.source, "
                "ts=excluded.ts, line_sha256=excluded.line_sha256",
                (file, key, value, source, ts if ts is not None else time.time(), line_sha256))
            self.db.commit()

    def fact_records(self, file: str) -> list[tuple[str, str, str, float, str]]:
        """(key, value, source, ts, line_sha256) for one vault file, newest first."""
        return self._read("SELECT key,value,source,ts,line_sha256 FROM fact_records WHERE file=? ORDER BY ts DESC, key",
                          (file,))

    def delete_fact_record(self, file: str, key: str) -> bool:
        with self._lock:
            if self._closed:
                return False
            cur = self.db.execute("DELETE FROM fact_records WHERE file=? AND key=?", (file, key))
            self.db.commit()
            return cur.rowcount > 0

    # notes
    def add_note(self, text: str) -> int:
        with self._lock:
            cur = self.db.execute(
                "INSERT INTO notes(text,created) VALUES(?,?)", (text.strip()[:2000], time.time())
            )
            self.db.commit()
        return int(cur.lastrowid or 0)

    def notes(self, limit: int = 20) -> list[str]:
        rows = self._read("SELECT text FROM notes ORDER BY id DESC LIMIT ?", (limit,))
        return [r[0] for r in rows]

    # reminders
    def add_reminder(self, text: str, due: float) -> int:
        with self._lock:
            cur = self.db.execute("INSERT INTO reminders(text,due) VALUES(?,?)", (text, due))
            self.db.commit()
        return int(cur.lastrowid or 0)

    def due_reminders(self, now: float | None = None) -> list[tuple[int, str]]:
        now = now if now is not None else time.time()
        return self._read("SELECT id,text FROM reminders WHERE fired=0 AND due<=? ORDER BY due", (now,))

    def pending_reminders(self) -> list[tuple[int, str, float]]:
        return self._read("SELECT id,text,due FROM reminders WHERE fired=0 ORDER BY due")

    def mark_fired(self, rid: int) -> None:
        with self._lock:
            if self._closed:
                return
            self.db.execute("UPDATE reminders SET fired=1 WHERE id=?", (rid,))
            self.db.commit()

    # key/value (rolling summary etc.)
    def get_kv(self, key: str, default: str = "") -> str:
        row = self._read("SELECT value FROM kv WHERE key=?", (key,), one=True)
        return row[0] if row else default

    def set_kv(self, key: str, value: str) -> None:
        with self._lock:
            if self._closed:
                return
            self.db.execute(
                "INSERT INTO kv(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value)
            )
            self.db.commit()

    # the last file EV created (screenshot, note...), so "open it" has something to open
    def set_last_artifact(self, path: Path | str, kind: str) -> None:
        self.set_kv("last_artifact", str(path))
        self.set_kv("last_artifact_kind", kind)

    def last_artifact(self) -> tuple[str, str]:
        return self.get_kv("last_artifact"), self.get_kv("last_artifact_kind")

    def snapshot(self) -> dict[str, Any]:
        return {"facts": self.facts(), "summary": self.get_kv("summary")}
