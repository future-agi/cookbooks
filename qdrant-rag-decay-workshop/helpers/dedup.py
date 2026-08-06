"""Find duplicate point ids for the dedup fix. Camera never opens this file.

A point is a duplicate if another point already carries the same (doc_id, chunk_index) —
i.e. the same fragment of the same document, re-ingested from an overlapping crawl. We
keep the lowest id in each group and return the rest for deletion.
"""

from __future__ import annotations

from qdrant_client import QdrantClient

from helpers import config

SCROLL_BATCH = 256


def find_duplicate_ids(client: QdrantClient, collection: str = config.COLLECTION) -> list[int]:
    seen: set[tuple[str, int]] = set()
    dup_ids: list[int] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=["doc_id", "chunk_index"],
            with_vectors=False,
        )
        for p in points:
            key = (p.payload["doc_id"], p.payload["chunk_index"])
            if key in seen:
                dup_ids.append(p.id)
            else:
                seen.add(key)
        if offset is None:
            break
    return dup_ids
