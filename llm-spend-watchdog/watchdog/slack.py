"""Delivery with a person in the loop.

Nothing is posted on its own. The default is a dry run: the message is printed
and saved to outbox/. With --approve, the message is shown and posted only
after someone types y. Replays never post.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

from .alerts import redact

OUTBOX = Path(__file__).resolve().parent.parent / "outbox"


def save(message: str, name: str) -> Path:
    OUTBOX.mkdir(exist_ok=True)
    path = OUTBOX / (re.sub(r"[^0-9A-Za-z_-]+", "-", name).strip("-") + ".md")
    path.write_text(message + "\n")
    return path


def post(message: str, webhook: str) -> None:
    req = urllib.request.Request(webhook, data=json.dumps({"text": message}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        if r.status >= 300:
            raise RuntimeError(f"Slack returned {r.status}")


def deliver(message: str, name: str, approve: bool = False) -> str:
    """Returns what happened: drafted, posted or declined."""
    message = redact(message)
    path = save(message, name)
    print("\n" + message + "\n")
    if not approve:
        print(f"Dry run: saved to {path}. Nothing was posted.")
        return "drafted"
    webhook = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook:
        print(f"No SLACK_WEBHOOK_URL set, so nothing was posted. Saved to {path}.")
        return "drafted"
    if input("Post this to Slack? [y/N] ").strip().lower() != "y":
        print("Not posted.")
        return "declined"
    post(message, webhook)
    print("Posted to Slack.")
    return "posted"
