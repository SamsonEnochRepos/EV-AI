from __future__ import annotations

from pathlib import Path

from ev.core.config import ROOT
from ev.memory.migrations import SEED_FLAG, seed_from_persona


def write_seed(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "seed.toml"
    p.write_text(body, encoding="utf-8")
    return p


def test_seeds_once_and_skips_placeholders(memory, tmp_path):
    seed = write_seed(tmp_path, '[facts]\nname = "Sam"\ninterests = "[FILL IN]"\nblank = "  "\n')
    assert seed_from_persona(memory, seed) == 1
    assert memory.facts() == {"name": "Sam"}
    assert memory.get_kv(SEED_FLAG) == "1"


def test_second_run_adds_nothing_even_after_the_user_forgot_a_fact(memory, tmp_path):
    seed = write_seed(tmp_path, '[facts]\nname = "Sam"\n')
    seed_from_persona(memory, seed)
    memory.forget("name")
    assert seed_from_persona(memory, seed) == 0            # forgetting must stick, not be re-seeded at every start
    assert memory.facts() == {}


def test_never_overwrites_a_fact_the_user_already_has(memory, tmp_path):
    memory.remember("location", "Pune")
    seed = write_seed(tmp_path, '[facts]\nlocation = "India"\nname = "Sam"\n')
    assert seed_from_persona(memory, seed) == 1
    assert memory.recall("location") == "Pune"


def test_missing_seed_file_is_not_an_error_and_not_marked_done(memory, tmp_path):
    assert seed_from_persona(memory, tmp_path / "nope.toml") == 0
    assert memory.get_kv(SEED_FLAG) == ""


def test_shipped_seed_file_loads_and_matches_what_the_persona_used_to_say(memory):
    n = seed_from_persona(memory, ROOT / "config" / "seed_facts.toml")
    facts = memory.facts(50)
    assert n == len(facts) and n >= 8
    assert "Samson" in facts["name"] and "India" in facts["country"] and "Ryzen 5 7535HS" in facts["laptop"]
    assert not {"location", "city", "home_city", "hometown", "home", "lives_in", "current_city", "place"} & facts.keys()
    assert "Suryaprasanth" in facts["not_to_confuse"]
    assert "interests" not in facts                        # the [FILL IN] placeholder is not stored
    assert all(len(v) <= 200 for v in facts.values())      # Phase 2 caps a fact at 200 characters


def test_seed_facts_fit_the_prompt_and_say_nothing_invented():
    from ev.core.config import load_toml

    facts = load_toml(ROOT / "config" / "seed_facts.toml")["facts"]
    real = {k: v for k, v in facts.items() if "[FILL IN]" not in v}
    from ev.core.config import load_config

    room = int(load_config()["memory"]["max_facts_in_prompt"])
    assert len(real) <= room - 4, "seeded facts leave fewer than four slots for facts the user adds"
    assert all(len(v) <= 200 for v in real.values())
    blob = " ".join(real.values()).lower()
    for invented in ("dog", "biscuit", "bistic", "cat ", "wife", "girlfriend", "birthday", "phone number", "password"):
        assert invented not in blob, f"seed facts mention {invented!r}"


def apply_updates_env(memory, tmp_path, body):
    from ev.memory.vault import Vault

    memory.set_kv(SEED_FLAG, "1")                        # an install that was seeded at its first start-up
    v = Vault(tmp_path / "vault", memory)
    v.ensure()
    seed = write_seed(tmp_path, body)
    return v, seed


def test_an_install_that_never_seeded_gets_no_seed_updates(memory, tmp_path):
    from ev.memory.migrations import apply_seed_updates
    from ev.memory.vault import Vault

    v = Vault(tmp_path / "vault", memory)
    v.ensure()
    assert apply_seed_updates(memory, v, write_seed(tmp_path, '[facts]\nprojects = "EV"\n')) == 0
    assert v.profile_facts() == {}


def test_seed_updates_add_only_new_keys(memory, tmp_path):
    from ev.memory.migrations import SEEDED_KEYS, apply_seed_updates

    v, seed = apply_updates_env(memory, tmp_path, '[facts]\nname = "Sam"\n')
    assert apply_seed_updates(memory, v, seed) == 0              # "name" is a first-release key: nothing to add
    seed.write_text('[facts]\nname = "Sam"\nprojects = "EV and Cell Intel"\nhobby = "[FILL IN]"\n', encoding="utf-8")
    assert apply_seed_updates(memory, v, seed) == 1
    assert v.profile_facts()["projects"] == "EV and Cell Intel"
    assert {r[0]: r[2] for r in memory.fact_records("Facts.md")}["projects"] == "seeded"
    assert apply_seed_updates(memory, v, seed) == 0              # nothing new the second time
    assert "projects" in memory.get_kv(SEEDED_KEYS) and "hobby" not in memory.get_kv(SEEDED_KEYS)   # placeholders wait
    seed.write_text('[facts]\nname = "Sam"\nprojects = "EV and Cell Intel"\nhobby = "chess"\n', encoding="utf-8")
    assert apply_seed_updates(memory, v, seed) == 1 and v.profile_facts()["hobby"] == "chess"       # filled in later: added


def test_seed_updates_never_overwrite_or_resurrect(memory, tmp_path):
    from ev.memory.migrations import apply_seed_updates

    v, seed = apply_updates_env(memory, tmp_path, '[facts]\nname = "Sam"\nlaptop = "old laptop"\n')
    v.add_fact("name", "Samson, told by the user", "user said")                         # the user's own wording
    v.add_fact("laptop", "old laptop", "seeded")
    apply_seed_updates(memory, v, seed)
    assert v.forget_fact("laptop")                                                      # the user forgot it
    seed.write_text('[facts]\nname = "Different"\nlaptop = "old laptop"\nhobby = "chess"\n', encoding="utf-8")
    assert apply_seed_updates(memory, v, seed) == 1
    facts = v.profile_facts()
    assert facts["name"] == "Samson, told by the user" and "laptop" not in facts and facts["hobby"] == "chess"


def test_an_install_from_before_the_seed_record_does_not_get_its_original_keys_back(memory, tmp_path):
    """The real first release seeded name/country/work/...; a user who forgot one before this mechanism existed keeps it forgotten."""
    from ev.memory.migrations import apply_seed_updates

    v, seed = apply_updates_env(memory, tmp_path, '[facts]\nname = "Sam"\nlaptop = "x"\nprojects = "EV"\n')
    v.add_fact("name", "Sam", "seeded")                  # laptop was seeded once and then forgotten: no record, no line
    assert apply_seed_updates(memory, v, seed) == 1      # only the genuinely new key
    assert set(v.profile_facts()) == {"name", "projects"}


def test_persona_no_longer_carries_user_facts():
    persona = (ROOT / "config" / "persona.md").read_text(encoding="utf-8")
    for needle in ("Samson", "IGME", "Ryzen", "Suryaprasanth", "India", "[FILL IN]"):
        assert needle not in persona
