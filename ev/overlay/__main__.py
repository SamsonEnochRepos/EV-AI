"""python -m ev.overlay --demo   |   python -m ev.overlay --serve (spawned by EV)"""
from __future__ import annotations

import argparse
import sys

from ev.core.config import load_config


def main() -> int:
    ap = argparse.ArgumentParser(prog="ev.overlay")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--demo", action="store_true", help="cycle through every state with synthetic audio")
    g.add_argument("--serve", action="store_true", help="internal: run as EV's overlay process")
    a = ap.parse_args()
    cfg = load_config()
    toolkit = cfg["overlay"].get("toolkit", "qt")
    if toolkit == "tk" and a.demo:     # Tk is demo/benchmark-only (see docs/DECISIONS.md D5)
        from . import tk_app

        return tk_app.run_demo(cfg)
    from . import qt_app

    return qt_app.run_demo(cfg) if a.demo else qt_app.run_server(cfg)


if __name__ == "__main__":
    sys.exit(main())
