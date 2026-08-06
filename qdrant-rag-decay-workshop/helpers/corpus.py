"""Build the workshop corpus as documents with the locked payload schema.

Every document is a dict with exactly these keys (the payload schema from CLAUDE.md):
    doc_id      stable id, e.g. "magnemite-gen2-types"  (gold labels reference this)
    name        Pokemon or type name
    generation  int 1..9 — the generation the document's facts describe
    intro_gen   int 1..9 — the generation the species entered the Pokedex (1 for
                type-chart docs). The scaling curve stages real growth on this.
    doc_type    "pokedex_entry" | "types" | "stats" | "type_chart"
    sprite_url  front sprite (empty for type_chart docs)
    text        the searchable/citable text
    is_current  False for a superseded types/type_chart snapshot, True otherwise. The
                payload-filter fix (arc step 7) is one line: keep is_current == True.

doc_id is stable across re-ingestion and re-embedding, so IR gold labels survive
fix #1 (re-ingest) and fix #2 (re-embed). Never key gold labels on point ids.
"""

from __future__ import annotations

from helpers import pokeapi

STAT_LABELS = {
    "hp": "HP", "attack": "Attack", "defense": "Defense",
    "special-attack": "Sp. Atk", "special-defense": "Sp. Def", "speed": "Speed",
}


def _flavor_docs(pkmn: dict, species: dict) -> list[dict]:
    """One flavor doc per generation, deduping identical text across game versions."""
    name = pkmn["name"]
    sprite = pkmn["sprites"]["front_default"] or ""
    seen_per_gen: dict[int, str] = {}
    for entry in species["flavor_text_entries"]:
        if entry["language"]["name"] != "en":
            continue
        gen = pokeapi.VERSION_TO_GEN.get(entry["version"]["name"])
        if gen is None or gen in seen_per_gen:
            continue
        text = " ".join(entry["flavor_text"].split())  # strip \n\f and collapse spaces
        seen_per_gen[gen] = text
    return [
        {
            "doc_id": f"{name}-gen{gen}-entry",
            "name": name,
            "generation": gen,
            "doc_type": "pokedex_entry",
            "sprite_url": sprite,
            "text": f"{name.capitalize()}: {text}",
        }
        for gen, text in sorted(seen_per_gen.items())
    ]


def _type_docs(pkmn: dict, intro_gen: int) -> list[dict]:
    """Per-generation type docs, reconstructed from types + past_types.

    past_types lists the types a Pokemon HAD in a past generation; current types apply
    from the generation after the last past snapshot. Magnemite -> gen1 Electric,
    gen2 Electric/Steel: the cold-open conflict.
    """
    name = pkmn["name"]
    sprite = pkmn["sprites"]["front_default"] or ""
    docs = []
    max_past_gen = 0
    for past in pkmn.get("past_types", []):
        gen = pokeapi.gen_name_to_num(past["generation"]["name"])
        max_past_gen = max(max_past_gen, gen)
        types = [t["type"]["name"] for t in past["types"]]
        docs.append(_one_type_doc(name, sprite, gen, types))
    current = [t["type"]["name"] for t in pkmn["types"]]
    current_gen = max(max_past_gen + 1, intro_gen)
    docs.append(_one_type_doc(name, sprite, current_gen, current))
    return docs


def _one_type_doc(name: str, sprite: str, gen: int, types: list[str]) -> dict:
    typing = "/".join(t.capitalize() for t in types)
    return {
        "doc_id": f"{name}-gen{gen}-types",
        "name": name,
        "generation": gen,
        "doc_type": "types",
        "sprite_url": sprite,
        # Generation lives in the payload, NOT the text: the stale and current typings
        # must read as flat contradictions so the payload filter is what resolves them.
        "text": f"{name.capitalize()} is a {typing}-type Pokemon.",
    }


def _stats_doc(pkmn: dict, intro_gen: int) -> dict:
    """Single current stats doc — PokeAPI exposes no past_stats, so no history."""
    name = pkmn["name"]
    stats = {s["stat"]["name"]: s["base_stat"] for s in pkmn["stats"]}
    total = sum(stats.values())
    parts = ", ".join(f"{lbl} {stats[k]}" for k, lbl in STAT_LABELS.items())
    return {
        "doc_id": f"{name}-stats",
        "name": name,
        "generation": intro_gen,
        "doc_type": "stats",
        "sprite_url": pkmn["sprites"]["front_default"] or "",
        "text": f"{name.capitalize()} base stats: {parts}. Total: {total}.",
    }


def _render_type_chart(tname: str, rel: dict) -> str:
    def names(key):
        return [x["name"].capitalize() for x in rel[key]]
    resists = names("half_damage_from")
    weak = names("double_damage_from")
    immune = names("no_damage_from")
    # Generation stays in the payload, not the text (same reason as type docs).
    text = f"The {tname.capitalize()} type resists {', '.join(resists)}."
    if immune:
        text += f" It takes no damage from {', '.join(immune)}."
    text += f" It is weak to {', '.join(weak)}."
    return text


def _type_chart_docs(type_json: dict) -> list[dict]:
    """Per-generation type-chart docs for one type: current + each past snapshot.

    Steel -> gen5 doc (resists Ghost/Dark) + current doc (does not): the cold-open
    fallback when the generator leaks Magnemite's typing from memory.
    """
    tname = type_json["name"]
    docs = []
    max_past_gen = 0
    for past in type_json.get("past_damage_relations", []):
        gen = pokeapi.gen_name_to_num(past["generation"]["name"])
        max_past_gen = max(max_past_gen, gen)
        docs.append({
            "doc_id": f"typechart-{tname}-gen{gen}",
            "name": tname,
            "generation": gen,
            "doc_type": "type_chart",
            "sprite_url": "",
            "text": _render_type_chart(tname, past["damage_relations"]),
        })
    current_gen = max_past_gen + 1 if max_past_gen else 1
    docs.append({
        "doc_id": f"typechart-{tname}-gen{current_gen}",
        "name": tname,
        "generation": current_gen,
        "doc_type": "type_chart",
        "sprite_url": "",
        "text": _render_type_chart(tname, type_json["damage_relations"]),
    })
    return docs


# All 18 standard types (every type-chart doc the corpus carries).
TYPE_NAMES = [
    "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison",
    "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon", "steel",
    "fairy", "dark",
]


def build_corpus() -> list[dict]:
    """Fetch and assemble the full clean corpus (no duplicates, no chunking yet)."""
    pokemon, species = pokeapi.fetch_pokedex()
    types = pokeapi.fetch_types(TYPE_NAMES)
    docs: list[dict] = []
    for pkmn, spec in zip(pokemon, species):
        intro_gen = pokeapi.gen_name_to_num(spec["generation"]["name"])
        species_docs = (
            _flavor_docs(pkmn, spec)
            + _type_docs(pkmn, intro_gen)
            + [_stats_doc(pkmn, intro_gen)]
        )
        for doc in species_docs:
            doc["intro_gen"] = intro_gen
        docs.extend(species_docs)
    for tjson in types.values():
        chart_docs = _type_chart_docs(tjson)
        for doc in chart_docs:
            doc["intro_gen"] = 1  # charts are reference docs, present from the start
        docs.extend(chart_docs)

    # is_current: only types / type_chart have superseded snapshots. Within each such
    # (name, doc_type) group the highest generation is current; older ones are stale.
    for doc in docs:
        doc["is_current"] = True
    for dtype in ("types", "type_chart"):
        groups: dict[str, list[dict]] = {}
        for doc in docs:
            if doc["doc_type"] == dtype:
                groups.setdefault(doc["name"], []).append(doc)
        for group in groups.values():
            latest = max(d["generation"] for d in group)
            for doc in group:
                doc["is_current"] = doc["generation"] == latest
    return docs


if __name__ == "__main__":
    # ponytail: self-check that the corpus builds and the cold-open docs exist.
    corpus = build_corpus()
    by_type: dict[str, int] = {}
    by_gen: dict[int, int] = {}
    for d in corpus:
        assert set(d) == {"doc_id", "name", "generation", "intro_gen", "doc_type",
                          "sprite_url", "text", "is_current"}
        by_type[d["doc_type"]] = by_type.get(d["doc_type"], 0) + 1
        by_gen[d["intro_gen"]] = by_gen.get(d["intro_gen"], 0) + 1
    ids = {d["doc_id"] for d in corpus}
    assert len(ids) == len(corpus), "doc_ids must be unique"
    # Cold-open conflict must be present as two generation-tagged docs.
    assert "magnemite-gen1-types" in ids and "magnemite-gen2-types" in ids
    # Steel type-chart fallback pair.
    assert "typechart-steel-gen5" in ids
    print(f"corpus: {len(corpus)} docs  {by_type}")
    print(f"docs by intro_gen: {dict(sorted(by_gen.items()))}")
    for did in ["magnemite-gen1-types", "magnemite-gen2-types",
                "typechart-steel-gen5", "drowzee-gen1-entry"]:
        print(" ", next(d["text"] for d in corpus if d["doc_id"] == did))
