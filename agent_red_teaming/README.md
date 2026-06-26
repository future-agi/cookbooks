# Agent Red-Teaming with FutureAGI Guardrails

Fires a battery of adversarial inputs at a simple support agent, screens every
turn with the FutureAGI guardrail pipeline at input and output, and traces the
run in FutureAGI Observe.

## What this covers

- A guardrail pipeline (`create_default_pipeline`) wired with the jailbreak,
  code-injection, and secrets scanners.
- Pre-screening user input so malicious turns never reach the agent.
- Post-screening agent output for secret and PII leakage.
- Full OpenTelemetry tracing of the run in FutureAGI Observe.

The adversarial inputs mirror the adversarial simulation personas: prompt
injection, jailbreak, secret exfiltration, and PII extraction, plus a benign
control.

## Coverage notes

| Attack | Where blocked |
|---|---|
| `prompt_injection` | **Input** — jailbreak scanner catches override framing |
| `jailbreak` | **Input** — jailbreak scanner catches DAN / roleplay framing |
| `secret_exfiltration` | **Output** — social-engineering request passes input scan; secrets scanner catches any credentials the agent might echo back |
| `pii_extraction` | **Output** — same; the agent's reply is screened for leaked data |
| `benign_control` | Passes both scans — traces cleanly in Observe |

This two-layer model (input block + output leakage scan) reflects real-world
deployment: you can't always identify social engineering from the request alone,
so output screening is the second line of defense.

## Prerequisites

- `FI_API_KEY` and `FI_SECRET_KEY` from your [FutureAGI dashboard](https://app.futureagi.com/) (Settings → API keys)
- `OPENAI_API_KEY` from [OpenAI](https://platform.openai.com/api-keys)

All have free tiers that cover this example. The guardrail scanners run locally.

## Run

```bash
pip install -r requirements.txt
export FI_API_KEY=... FI_SECRET_KEY=... OPENAI_API_KEY=...
python red_team_guardrails.py
```

Expected output:

```
attack                     input      detail
------------------------------------------------------------
prompt_injection           BLOCKED    blocked_by=['jailbreak']
jailbreak                  BLOCKED    blocked_by=['jailbreak']
secret_exfiltration        passed     output clean
pii_extraction             passed     output clean
benign_control             passed     output clean

Open FutureAGI Observe to see every turn as a traced span: https://app.futureagi.com
```

## Files

- `red_team_guardrails.py` — the runnable example
- `requirements.txt` — dependencies
- `README.md` — this file
