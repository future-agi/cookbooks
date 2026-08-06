"""Shared constants: collection, named vectors, models, top-k.

Change defaults HERE, never ad hoc in ingest / agent / notebook, so all three surfaces
agree on vector names and models. The workshop's whole story depends on it.
"""

import os
from pathlib import Path

# Qdrant collection the whole workshop runs against.
COLLECTION = "pokemon_webinar"

# Small sibling collection for the Qdrant Web UI point-cloud beat (fix #1). Sized so the
# Visualize tab's sample is near-complete and the duplicate clusters actually show.
VIZ_COLLECTION = "pokemon_viz"

# Cross-process retrieval switch: the notebook writes it, the Streamlit app reads it per
# question. ingest.py resets it so a stale rehearsal state can never leak into the show.
STATE_FILE = Path(__file__).resolve().parent.parent / "data" / ".retrieval_state.json"
DEFAULT_STATE = {"mode": "minilm", "current_only": False}

# Named vectors on the collection. The baseline uses DENSE_WEAK only; each fix adds one.
DENSE_WEAK = "dense_weak"       # fix #2 migrates away from this
DENSE_STRONG = "dense_strong"   # fix #2 target
SPARSE = "sparse"               # fix #3 hybrid prefetch
COLBERT = "colbert"             # fix #3 late-interaction rerank

# FastEmbed model ids (identifiers verified against the installed fastembed).
MODEL_DENSE_WEAK = "sentence-transformers/all-MiniLM-L6-v2"   # 384d
# bge-large holds recall as the haystack + duplicates grow; mxbai's high similarity floor
# collapsed under the duplicate crowding on this corpus (verified 2026-07-20).
MODEL_DENSE_STRONG = "BAAI/bge-large-en-v1.5"                 # 1024d
MODEL_SPARSE = "Qdrant/minicoil-v1"                           # miniCOIL, Qdrant-differentiated
MODEL_COLBERT = "colbert-ir/colbertv2.0"                      # MAX_SIM multivector

DIM_DENSE_WEAK = 384
DIM_DENSE_STRONG = 1024
DIM_COLBERT = 128  # colbertv2.0 token dimension

# Retrieval defaults. Small top-k so duplicates actually crowd out unique chunks (fix #1).
TOP_K = 5
# The panel shows deeper than the agent reads, so the audience can see the right doc
# sitting below the cutoff (cold open: 8 stale steel-gen5 copies, current gen6 at rank 9).
PANEL_K = 10

# Generator (strict grounding prompt lives in agent.py), as a "provider:model" string for
# langchain's init_chat_model — set GENERATOR_MODEL in .env to run on the API key you have
# ("google_vertexai:gemini-2.5-flash", "openai:gpt-5.5", ...). Small/fast is deliberate:
# the demo depends on the model NOT answering Pokemon questions from parametric memory.
# Whichever model produces a scored run also owns its numbers — judged answer quality is
# not portable across generators, so re-run the golden set after changing this.
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "anthropic:claude-haiku-4-5")
