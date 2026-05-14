# Google ADK Production Eval Loop

Companion notebook for the blog post [How to Evaluate Google ADK Agents with FutureAGI](https://futureagi.com/blog/evaluate-google-adk-agents).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/future-agi/cookbooks/blob/main/google_adk_eval_loop/Google_ADK_Eval_Loop.ipynb)

## What this covers

Three runnable steps from the 6-step ADK Production Eval Loop:

1. **Instrument** Google ADK with `traceai-google-adk` so every agent invocation, tool call, and Gemini completion lands in FutureAGI Observe as structured OpenTelemetry spans.
2. **Score** the agent's output with the unified `fi.evals.evaluate()` API.
3. **Auto-enrich** spans with the score so traces in Observe carry their evaluation results inline.

The CI gate (Step 4), simulate (Step 5), and optimize (Step 6) steps are documented but not run inline — Step 4 is meant for pytest, Step 5 (`fi.simulate`) is voice-only, and Step 6 has its own dedicated notebook.

## Prerequisites

- A `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/app/apikey)
- A `FI_API_KEY` and `FI_SECRET_KEY` from your [FutureAGI dashboard](https://app.futureagi.com/)

Both have free tiers that comfortably cover this notebook.

## Files

- `Google_ADK_Eval_Loop.ipynb` — the notebook
- `README.md` — this file
