"""Word-boundary chunker. Camera never opens this file.

The broken baseline over-chunks with a tiny size so flavor text fragments across several
points. The fragmentation is planted and deliberately never fixed — dedup removes copies,
not truncation — so truncated flavor answers read as broken data in the before-state.
Type-chart docs are never chunked (the cold open depends on them staying whole).
"""

from __future__ import annotations

# Small on purpose: splits flavor text (~150-250 chars) into 2-3 truncated points — a
# plausible bad config, not a cartoon. The broken state fix #1 exposes.
BROKEN_CHUNK_CHARS = 110
BROKEN_OVERLAP_CHARS = 25


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split into <=size-char chunks on word boundaries with a char overlap."""
    words = text.split()
    if not words:
        return []
    chunks, cur, cur_len = [], [], 0
    for w in words:
        if cur and cur_len + 1 + len(w) > size:
            chunks.append(" ".join(cur))
            # start next chunk with a tail overlap
            back, blen = [], 0
            for pw in reversed(cur):
                if blen + len(pw) > overlap:
                    break
                back.insert(0, pw)
                blen += len(pw) + 1
            cur, cur_len = back, sum(len(x) + 1 for x in back)
        cur.append(w)
        cur_len += 1 + len(w)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


if __name__ == "__main__":
    # ponytail: check the broken chunk size actually fragments a flavor-length string.
    s = ("Charizard: Spits fire that is hot enough to melt boulders. "
         "Known to cause forest fires unintentionally. When expelling a blast of "
         "superhot fire, the red flame at the tip of its tail burns more intensely.")
    cs = chunk_text(s, BROKEN_CHUNK_CHARS, BROKEN_OVERLAP_CHARS)
    assert len(cs) >= 2, cs
    assert all(len(c) <= BROKEN_CHUNK_CHARS + 15 for c in cs), cs
    print(f"{len(cs)} chunks:", cs)
