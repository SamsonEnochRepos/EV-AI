"""One-time memory migrations that run at start-up."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ev.core.config import Config, load_toml
from ev.core.log import get_logger
from ev.memory.store import MemoryStore, norm_key
from ev.memory.vault import FACTS_FILE, Vault, VaultError
from ev.memory.vault_index import VaultIndex
from ev.safety.policy import Policy

log = get_logger("migrations")

SEED_FLAG = "migrated.seed_facts"
VAULT_FLAG = "migrated.vault_facts"


def seed_from_persona(memory: MemoryStore, seed_file: Path) -> int:
    """Put the facts that used to be written into persona.md into memory, once.

    The persona is now only Evie's character and rules; what she knows about the user lives in memory, where the user can
    change it by voice. A fact that is already stored is never overwritten (the user may have corrected it since), and
    placeholder values like "[FILL IN]" are skipped. Returns how many facts were added."""
    if memory.get_kv(SEED_FLAG) == "1":
        return 0
    if not seed_file.exists():
        log.warning("seed facts file %s not found; skipping", seed_file)
        return 0
    added = 0
    for key, value in load_toml(seed_file).get("facts", {}).items():
        value = str(value).strip()
        if not value or "[FILL IN]" in value or memory.recall(key) is not None:
            continue
        memory.remember(key, value)
        added += 1
    memory.set_kv(SEED_FLAG, "1")
    log.info("seeded %d facts from %s", added, seed_file.name)
    return added


SEEDED_KEYS = "seed.keys"
# The keys in the first release of seed_facts.toml. An install that predates the SEEDED_KEYS record is treated as having
# seeded these, so a fact the user deliberately forgot is not brought back by a later seed update.
_FIRST_SEED_KEYS = frozenset({"name", "country", "work", "building", "values", "laptop", "other_projects", "interests",
                              "not_to_confuse"})


def apply_seed_updates(memory: MemoryStore, vault: Vault, seed_file: Path) -> int:
    """Add facts that were added to config/seed_facts.toml after this install first seeded itself.

    Only keys that have never been seeded here are added, each through `Vault.add_fact` (so each gets a record, source
    "seeded"). An existing fact is never overwritten (the user may have edited or corrected it), and a fact the user forgot
    is not resurrected, because every key ever seeded is remembered in kv. Returns how many were added."""
    if not seed_file.exists() or memory.get_kv(SEED_FLAG) != "1":
        return 0                                         # only an install that was seeded to begin with gets seed updates
    wanted = {norm_key(k): str(v).strip() for k, v in load_toml(seed_file).get("facts", {}).items()}
    raw = memory.get_kv(SEEDED_KEYS)
    seen: set[str] = set(json.loads(raw)) if raw else set(_FIRST_SEED_KEYS)
    if not raw:
        seen |= {r[0] for r in memory.fact_records(FACTS_FILE)} | {u.key for u in vault.unconfirmed_facts()}
    added = 0
    for key, value in wanted.items():
        if key in seen or not value or "[FILL IN]" in value:
            continue
        try:
            vault.add_fact(key, value, "seeded")
            added += 1
        except VaultError as e:
            log.warning("seed fact %r not added: %s", key, e)
            if "couldn't write" in str(e):
                continue                                  # try again next start
        seen.add(key)
    memory.set_kv(SEEDED_KEYS, json.dumps(sorted(seen)))
    if added:
        log.info("added %d new seed fact(s) to the vault", added)
    return added


def open_vault(cfg: Config, memory: MemoryStore, policy: Policy) -> tuple[Vault | None, VaultIndex | None]:
    """Set the vault up at start-up: create the EV folder on first run, move any old SQLite facts in, open the search index.
    Any failure here (folder not writable, disk error) falls back to the SQLite facts, so memory trouble never stops EV."""
    vault = Vault.from_config(cfg, memory, policy)
    if vault is None:
        return None, None
    try:
        vault.ensure()
        migrate_facts_to_vault(memory, vault, cfg.path("memory", "seed_file"))
        apply_seed_updates(memory, vault, cfg.path("memory", "seed_file"))
        v = cfg["vault"]
        index = VaultIndex(vault.root, cfg.data_path(v["index_file"]), max_note_kb=int(v.get("max_note_kb", 512)),
                           poll_s=float(v.get("poll_s", 15)))
    except (OSError, VaultError, sqlite3.Error) as e:
        log.error("vault unavailable (%s); using the SQLite facts instead", e)
        return None, None
    return vault, index


def migrate_facts_to_vault(memory: MemoryStore, vault: Vault, seed_file: Path | None = None) -> int:
    """Move the SQLite facts table into the vault, once. Every fact goes through `Vault.add_fact`, so each one gets a record
    and is sanitised on the way (one that reads like an instruction, or can't be written, is skipped and logged). The old
    table is left alone and unused. Source is "seeded" for keys from the seed file and "migrated" for everything else.

    The flag is only set when nothing failed for a reason that could go away (the vault folder not writable), so a start-up
    with the vault unavailable tries again next time instead of silently losing the facts."""
    if memory.get_kv(VAULT_FLAG) == "1":
        return 0
    seeded = {norm_key(k) for k in load_toml(seed_file).get("facts", {})} if seed_file and seed_file.exists() else set()
    moved = 0
    retry = False
    for key, value in sorted(memory.facts(1000).items()):
        try:
            vault.add_fact(key, value, "seeded" if key in seeded else "migrated")
            moved += 1
        except VaultError as e:
            log.warning("fact %r not moved to the vault: %s", key, e)
            retry = retry or "couldn't write" in str(e)
        except OSError as e:
            log.warning("fact %r not moved to the vault: %s", key, e)
            retry = True
    if not retry:
        memory.set_kv(VAULT_FLAG, "1")
    log.info("moved %d facts into the vault", moved)
    return moved
