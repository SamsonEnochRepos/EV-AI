"""python -m ev [--debug] [--repl]"""
from __future__ import annotations

import argparse
import os
import sys

from ev.core.config import load_config

# No telemetry, and no model-hub traffic at runtime (models are downloaded by scripts/download_models.py).
for _k, _v in {"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
               "ANONYMIZED_TELEMETRY": "False", "TRANSFORMERS_OFFLINE": "1"}.items():
    os.environ.setdefault(_k, _v)


def main() -> int:
    ap = argparse.ArgumentParser(prog="ev", description="EV: local voice-first assistant")
    ap.add_argument("--debug", action="store_true", help="log to console too")
    ap.add_argument("--repl", action="store_true", help="developer text REPL (not the product UI)")
    ap.add_argument("--config", help="path to an alternate ev.toml")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()
    if a.repl:
        from ev.repl import run_repl

        return run_repl(cfg, a.debug)
    from ev.app import EVApp

    EVApp(cfg, a.debug).run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
