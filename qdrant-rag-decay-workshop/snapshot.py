"""Backup/restore the show collections via Qdrant snapshots. Camera never opens this.

Take a snapshot of the prepared show state once (after ingest.py && prep.py), then any
destructive run — verify_arc.py, a rehearsal, the live dedup — is reverted in seconds
with `restore` instead of re-running the slow ingest + backfill.

`restore` prefers the snapshot already on the cluster. If the cluster has none (fresh
cluster), it uploads the newest local `data/{collection}-*.snapshot` file instead, which
creates the collection directly — no ingest.py, no prep.py.

    uv run python snapshot.py backup     # snapshot pokemon_webinar + pokemon_viz
    uv run python snapshot.py restore    # revert both (cluster snapshot, else local file)
    uv run python snapshot.py download   # save the cluster snapshots to data/ for sharing
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from helpers import config

DATA = Path(__file__).resolve().parent / "data"


def restore(client: QdrantClient, col: str) -> None:
    snaps = client.list_snapshots(col) if client.collection_exists(col) else []
    if snaps:
        latest = max(snaps, key=lambda s: s.creation_time)
        client.recover_snapshot(
            col,
            location=f"file:///qdrant/snapshots/{col}/{latest.name}",
            priority=models.SnapshotPriority.SNAPSHOT,
        )
        print(f"{col}: restored cluster snapshot {latest.name} "
              f"-> {client.count(col).count} points")
        return

    local = sorted(DATA.glob(f"{col}-*.snapshot"))
    if not local:
        raise SystemExit(
            f"{col}: no snapshot on the cluster and no data/{col}-*.snapshot file. "
            "Get the snapshot file from Dylan, or rebuild with "
            "`ingest.py && prep.py && snapshot.py backup`.")
    path = local[-1]
    print(f"{col}: uploading {path.name} ({path.stat().st_size / 1e6:.0f} MB)…")
    with path.open("rb") as f:
        r = requests.post(
            f"{os.environ['QDRANT_URL']}/collections/{col}/snapshots/upload",
            params={"priority": "snapshot"},
            headers={"api-key": os.environ["QDRANT_API_KEY"]},
            files={"snapshot": (path.name, f)},
            timeout=1800,
        )
    r.raise_for_status()
    print(f"{col}: restored from local file -> {client.count(col).count} points")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["backup", "restore", "download"])
    args = ap.parse_args()

    load_dotenv()
    client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"],
                          timeout=300)
    for col in (config.COLLECTION, config.VIZ_COLLECTION):
        if args.command == "backup":
            snap = client.create_snapshot(col)
            print(f"{col}: created {snap.name} ({snap.size / 1e6:.0f} MB)")
        elif args.command == "restore":
            restore(client, col)
        else:
            snaps = client.list_snapshots(col)
            if not snaps:
                print(f"{col}: no cluster snapshot to download")
                continue
            latest = max(snaps, key=lambda s: s.creation_time)
            out = DATA / latest.name
            with requests.get(
                f"{os.environ['QDRANT_URL']}/collections/{col}/snapshots/{latest.name}",
                headers={"api-key": os.environ["QDRANT_API_KEY"]},
                stream=True, timeout=1800,
            ) as r:
                r.raise_for_status()
                with out.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            print(f"{col}: downloaded {out.name} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
