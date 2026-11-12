"""A tiny hook for agents that only run commands (Codex): forwards the request on stdin to EV and prints EV's answer.

No decision from EV (it is off, paused, or nobody answered) prints nothing and exits 0, so the agent asks the user as usual.
"""
import os
import sys
import urllib.request

port = os.environ.get("EV_AGENT_PORT", "47831")
token = os.environ.get("EV_AGENT_TOKEN", "")
agent = os.environ.get("EV_AGENT_NAME", "codex")
try:
    body = sys.stdin.buffer.read()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/permission/{agent}", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=60) as r:       # noqa: S310  (a fixed http://127.0.0.1 URL)
        out = r.read()
        if r.status == 200 and out:
            sys.stdout.write(out.decode("utf-8"))
except Exception:                                            # noqa: BLE001, S110
    pass
