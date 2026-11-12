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
    ap.add_argument("--display", action="store_true", help="with --demo: show display mode (the text card)")
    a = ap.parse_args()
    cfg = load_config()
    from . import qt_app

    return qt_app.run_demo(cfg, a.display) if a.demo else qt_app.run_server(cfg)


if __name__ == "__main__":
    sys.exit(main())
