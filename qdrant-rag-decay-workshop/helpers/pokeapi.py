"""PokeAPI fetching with an on-disk cache. Camera never opens this file.

Fetches the raw JSON the corpus builder needs: per-Pokemon data (types, past_types,
stats, sprite) and species data (flavor text per game version), plus type-chart
damage relations. Everything is cached to data/corpus_cache.json so re-runs are free.
"""

from __future__ import annotations

import json
from pathlib import Path

import requests

BASE = "https://pokeapi.co/api/v2"
CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "corpus_cache.json"

# The full Pokedex: species ids 1..1025 (through Gen 9).
ALL_IDS = range(1, 1026)

# Static version -> generation map. PokeAPI flavor entries reference a game version
# ("red", "sword"); the generation is stable game data, so we map it directly
# instead of walking version -> version-group -> generation per entry.
VERSION_TO_GEN = {
    "red": 1, "blue": 1, "yellow": 1,
    "gold": 2, "silver": 2, "crystal": 2,
    "ruby": 3, "sapphire": 3, "emerald": 3, "firered": 3, "leafgreen": 3,
    "diamond": 4, "pearl": 4, "platinum": 4, "heartgold": 4, "soulsilver": 4,
    "black": 5, "white": 5, "black-2": 5, "white-2": 5,
    "x": 6, "y": 6, "omega-ruby": 6, "alpha-sapphire": 6,
    "sun": 7, "moon": 7, "ultra-sun": 7, "ultra-moon": 7,
    "lets-go-pikachu": 7, "lets-go-eevee": 7,
    "sword": 8, "shield": 8, "brilliant-diamond": 8, "shining-pearl": 8,
    "legends-arceus": 8,
    "scarlet": 9, "violet": 9, "legends-z-a": 9,
}

# generation-i -> 1, etc. Covers the resource-url form PokeAPI uses in nested objects.
_GEN_NAME_TO_NUM = {f"generation-{r}": n for n, r in enumerate(
    ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix"], start=1)}


def gen_name_to_num(name: str) -> int:
    """'generation-ii' -> 2."""
    return _GEN_NAME_TO_NUM[name]


def _slim(url: str, data: dict) -> dict:
    """Keep only the fields the corpus builder reads. The raw /pokemon JSON carries the
    full move list (~100KB each); at 1,025 species the unslimmed cache would be ~350MB."""
    if "/pokemon-species/" in url:
        return {
            "name": data["name"],
            "generation": data["generation"],
            "flavor_text_entries": [
                {"flavor_text": e["flavor_text"], "language": e["language"],
                 "version": e["version"]}
                for e in data["flavor_text_entries"]
                if e["language"]["name"] == "en"
            ],
        }
    if "/pokemon/" in url:
        return {
            "name": data["name"],
            "types": data["types"],
            "past_types": data.get("past_types", []),
            "stats": data["stats"],
            "sprites": {"front_default": data["sprites"]["front_default"]},
        }
    if "/type/" in url:
        return {
            "name": data["name"],
            "damage_relations": data["damage_relations"],
            "past_damage_relations": data.get("past_damage_relations", []),
        }
    return data


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Re-slim everything on save so a legacy full-JSON cache shrinks in place.
    CACHE_PATH.write_text(json.dumps({url: _slim(url, d) for url, d in cache.items()}))


def _get(session: requests.Session, cache: dict, url: str) -> dict:
    if url in cache:
        return cache[url]
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    cache[url] = _slim(url, resp.json())
    return cache[url]


def fetch_pokedex(ids=ALL_IDS) -> tuple[list[dict], list[dict]]:
    """Return (pokemon, species) raw JSON lists for the given ids.

    pokemon[i] carries types/past_types/stats/sprites; species[i] carries flavor text
    plus the generation the species was introduced in.
    """
    cache = _load_cache()
    session = requests.Session()
    pokemon, species = [], []
    try:
        for pid in ids:
            pokemon.append(_get(session, cache, f"{BASE}/pokemon/{pid}"))
            species.append(_get(session, cache, f"{BASE}/pokemon-species/{pid}"))
    finally:
        _save_cache(cache)
    return pokemon, species


def fetch_types(type_names: list[str]) -> dict[str, dict]:
    """Return {type_name: raw JSON} for the given types (damage relations + past)."""
    cache = _load_cache()
    session = requests.Session()
    out = {}
    try:
        for name in type_names:
            out[name] = _get(session, cache, f"{BASE}/type/{name}")
    finally:
        _save_cache(cache)
    return out
