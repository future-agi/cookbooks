# Runbook: 60 Minutes

Surfaces: **App** (Streamlit), **NB** (`workshop.ipynb`), **FI** (Future AGI dashboard), **QUI** (Qdrant Cloud Web UI). Times are ceilings. Every beat below says who drives, what to do, what to type, and what numbers must land.

Every FI number below is the **live account's dashboard aggregate** (Haiku runs of 2026-08-04, run files committed in `data/`). Retrieval-cell numbers reproduce on this cluster to the digit; collection counts track the PokéAPI crawl date (this cache: 22,566 / 251).

If running late: shorten beat 12's recap, compress fix #2 to flip → score → commit, skip the second Web UI look after dedup, and trim beat 5's eval walkthrough to the judge criterion alone. Never cut the multi-hop trace or the cold-open close.

## Before The Show

1. Restore if needed: `uv run python snapshot.py restore`. Rebuild with `ingest.py` + `prep.py` only if the corpus or golden set changed.
2. `cat data/.retrieval_state.json` → `{"mode": "minilm", "current_only": false}` (missing file = same default).
3. `uv run python verify_arc.py --baseline-only` → all flaws red: fix1 dup-rate 0.67, fix2 recall 0.64, fix3 NDCG/MRR 0.22/0.21, cold open False.
4. Collections: `pokemon_webinar` = 22,566 points, `pokemon_viz` = 251 (older cache prints 22,946 / 240 — what must hold is the ~2.7× collapse at dedup).
5. Start the app, let models warm, ask one throwaway question, confirm the badge reads `minilm`, restart for clean history.
6. Restart the notebook kernel, run only the setup cell. Keep the executed backup notebook open in a second tab.
7. FI: logged in; the five stage projects bookmarked (`pokedex-webinar`, `…-fix1-dedup`, `…-fix2-bge`, `…-fix3-hybrid`, `…-fix4-filter`) plus **`pokedex-webinar-live`** — every live app/notebook question lands there (`FI_PROJECT_NAME` in `.env`); the five scored projects stay frozen (only `run_golden.py --stage` writes to them, and it is not run live — the golden cells in the notebook are commented out with "pre-run before the session"). Experiments view ready.
8. QUI: `pokemon_viz` Visualize tab open with `{"limit": 1000}`.
9. Screen: App + notebook on the shared display; FI ready to swap in; bookmarks hidden; 125% zoom; notifications off.

## Run Of Show

**1 · 0:00–0:05 · Context slides — Rishav, deck**
- Action: the four intro slides — agents decay quietly · green dashboards vs wrong answers · evals are the debugger (the contrast thesis) · tonight's patient + the 0.57 → 0.92 arc teaser.
- Land the handoff line from slide 4: "First: the patient's anatomy. Dylan — walk us through the RAG setup."

**2 · 0:05–0:11 · The patient — Dylan, App**
- Action: RAG architecture in two minutes (Qdrant collection, LangGraph agent, the retrieval panel), then one working question.
- Type: `Which Pokemon puts its prey to sleep and then eats their dreams?`
- Expect: correct answer citing [drowzee]; panel shows chunks with sprites, generation badges, ranks.

**3 · 0:11–0:16 · Cold open + decay — Dylan, App → NB §1**
- Action: ask the question that will run through the whole show. Point at the panel, not the answer: 8 copies of `typechart-steel-gen5` fill the agent's top-5; the current gen6 chart sits at rank 9, below the cutoff line. Then the pre-generated decay curve.
- Type: `Does the Steel type resist Ghost and Dark attacks?`
- Expect: confident wrong answer — "Yes! [Steel] resists both…"; curve: recall@5 0.67 → 0.45 → 0.39 at Gen 1 / Gen 1–4 / full dex.

**4 · 0:16–0:18 · Baseline golden run — Dylan, NB §1**
- Action: show the runner (the commented `run_golden.py --stage iter1` cell): 37 golden questions through the traced agent, one project per stage. The run was executed pre-show; the traces are sitting in `pokedex-webinar`.
- Hand back: "37 questions, 37 traces. Rishav — what do the numbers say?"

**5 · 0:18–0:24 · Eval setup + judge calibration — Rishav, FI**
- Action: on `pokedex-webinar`, walk the eval configuration: the five trace evals mapped to the root span (input / output / retrieved_context), then the answer-correctness judge with its criterion on screen (including "an answer that says it depends is not a correct answer"). Then open the cold-open trace.
- Staging note: the project is pre-scored (evals configured pre-show so 37 rows carry scores); this beat *walks* the configuration rather than creating it live — scoring 37 traces on camera would be dead air. Say what you're doing: "this is exactly the setup screen; we ran it before the session so we don't watch paint dry."

**6 · 0:24–0:27 · Baseline read — Rishav, FI**
- Action: the aggregates, then the Snorlax trace (row 4: same wrong Delcatty entry ×4).
- Expect: answer correctness 0.57 (16 of 37 fail, zero refusals), 2.0 searches per question (17 of 37 need 2+; one paraphrase takes five). Relevance reads a healthy-looking 0.92 while four questions in ten fail — groundedness (0.68) is the one early tell. Diagnosis out loud: 36/37 queries carry a duplicate in top-5 — duplication in the data, not a model problem. Hand to Dylan: dedup.

**7 · 0:27–0:34 · Fix #1 dedup — Dylan → Rishav, NB §2 → QUI → App → FI**
- Action: run the dup-rate cell, the dedup cell, the dup-rate cell again; refresh QUI Visualize; ask the app; Rishav reads `…-fix1-dedup`.
- Expect from the cells: Gengar dup-rate 80% → 0%; `pokemon_webinar` 22,566 → 8,415; `pokemon_viz` 251 → 95; QUI clusters visibly thin.
- Type in the app: `Tell me the Pokedex entry for Gengar`
- Expect in the app: before, five copies of one doc; after, five distinct documents and a richer answer.
- Expect on FI: answer correctness 0.57 → 0.76 (failures 16 → 9); attribution 0.89 → 1.00; duplicates in top-5 36/37 → 2/37 queries; searches 2.00 → 1.46. Say the surprise out loud: chunk utilization DIDN'T move (0.85 → 0.85) — five distinct chunks score the same as five copies of one; it never measured duplication. And the Steel question now HEDGES ("depends on the generation") — both charts reached the top-5. Hold it.

**8 · 0:34–0:40 · Fix #2 embedding migration — Dylan → Rishav, NB §3 → App → FI**
- Action: run the flip + A/B cell; ask the app; Rishav reads `…-fix2-bge`.
- Expect from the cell: prints four named vectors live on one collection, then recall@5 MiniLM 0.64 → bge 1.00. Narrate: rollback would be the same one-line flip.
- Type in the app: `the electric mouse Pokemon that stores electricity in the pouches on its cheeks`
- Expect in the app: ANSWER FLIPS — Pichu (wrong, re-verified) → Pikachu.
- Expect on FI: answer correctness 0.76 → 0.92 (failures 9 → 3); groundedness peaks at 0.92; searches 1.46 → 1.24. Nine queries outside the fix-2 slice get their gold doc into the top-5 too (eight of them fix-3 paraphrases). Steel is back to confidently wrong.
- Worth saying: the grounding prompt forbids guessing a name and searching to confirm it; even so the agent re-searches around rot — say "fewer retrieval loops", not milliseconds or dollars (we measured search counts, not latency or tokens).

**9 · 0:40–0:47 · Fix #3 hybrid + rerank — Dylan → Rishav, NB §4 → App → FI**
- Action: run the Marowak cell, then the flip + score cell; ask the app; Rishav reads `…-fix3-hybrid`.
- Expect from the cells: Marowak absent from dense top-30 → hybrid rank 1; recall 0.78 → 0.89, NDCG@5 0.61 → 0.77, MRR 0.55 → 0.73. Wobble: hybrid recall can print 0.94 (one gold doc flips rank 5↔7 on RRF ties) — both are a win, don't re-run chasing a number.
- Type in the app: `A Pokemon that lives in dark caves and uses sound waves to navigate and hunt.`
- Expect in the app: no answer flip (several bats genuinely match); the tell is the panel filling with echolocation bats.
- Expect on FI: answer correctness DIPS 0.92 → 0.86 — say it straight: the agent was already rescuing mis-ranked docs with extra rewrites, so fixing rank barely moves the answer column; the wins are rank metrics + fewer searches (1.24 → 1.14). No-hallucination hits its arc low here (0.81 → 0.70) with groundedness passing the flagged rows — confident ranking exposes generator embellishment. Attribution talking point: the fused pool at rank 20 holds the gold doc for 0.94 of queries, fusion-only top-5 is 0.72 (below pure dense 0.78) — the ColBERT rerank delivers the win. Steel hedges again: both charts at ranks 1–2.

**10 · 0:47–0:51 · Cold-open close — Dylan, NB §5 → App**
- Action: run the filter cell (index `is_current`, flip `current_only`), then the ask cell; repeat in the app.
- Expect from the cells: steel docs go from `[gen5, gen6, …]` to `[gen6]` only; the agent answers "No".
- Type in the app: `Does the Steel type resist Ghost and Dark attacks?`
- Expect in the app: wrong "Yes" becomes correct "No"; badge reads `current-only`.
- Expect on FI (`…-fix4-filter`): the judge that stayed red through four retrieval upgrades goes green (its reasoning reads its own criterion: commits to the same core claim as the reference); answer correctness lands at 0.92. Say it: four ranking upgrades never fixed this — currency lives in metadata.

**11 · 0:51–0:54 · Multi-hop trace — Rishav, NB §6 → FI**
- Action: run the Drowzee cell; open its trace.
- Type (the cell types it): `What does Drowzee eat, and is that Pokemon weak to Bug-type attacks?`
- Expect: 2+ `search_pokedex` retriever spans under one root span. If it answers in one hop, use the bookmarked rehearsal trace — do not retry live.

**12 · 0:54–0:57 · Experiments view — Rishav, FI**
- Action: before/after across the five projects.
- Expect: answer correctness 0.57 → 0.76 → 0.92 → 0.86 → 0.92; searches per question 2.00 → 1.46 → 1.24 → 1.14 → 1.16. All five stages fully scored on the live account — no pending columns.

**13 · 0:57–1:00 · Close + Q&A — both**
- Action: one line, then questions: measure → locate the failing layer → fix that layer → re-run the same queries → verify the number moved.

## Fallbacks

- Notebook cell fails: switch to the executed backup tab and narrate from output.
- Cold open answers correctly: show that stale retrieval still won; call it a generator lucky break and continue.
- Qdrant Cloud or Anthropic hiccup: narrate from the backup notebook and the fallback deck in `images/`.
- FI score behaves oddly: Rishav uses the notebook's local gold-label number and reconciles later.
- FI dashboard cannot score at all: narrate every read from the notebook's §7 recap table (transcribed from committed score files) and say that's what you're doing.
- Multi-hop collapses to one retrieval: use the bookmarked rehearsal trace; do not retry live.

## Rehearsal Gates

- Run `verify_arc.py` (full, destructive) before the dry run; `snapshot.py restore` afterward.
- Save the successful executed notebook as the backup tab.
- Bookmark a rehearsal trace where the Drowzee question produces two-plus retriever spans.
- Check `pokemon_viz` in QUI: duplicate clusters visible before dedup, thinner after.
- Confirm dashboard reads with Rishav: does the platform score traces while the standalone-eval endpoint is down, and can Experiments compare across the five projects?
- Recapture the fallback deck in `images/` on the live account (committed images show older runs): baseline multi-hop traces with the wrong Steel answer; the post-dedup Steel hedge trace and its eval row (trace metrics passing, judge red); cheek-pouch lookalikes and its Pikachu-first counterpart; the judge red/green rows; the Drowzee multi-hop tree. Plus a `pokemon_viz` point-cloud before/after pair for the Visualize fallback.
