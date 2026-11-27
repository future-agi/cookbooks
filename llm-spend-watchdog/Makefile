PYTHON ?= python3
PY := .venv/bin/python

.PHONY: setup data demo replay check digest eval eval-cloud test clean

setup: .venv/.installed
.venv/.installed: requirements.txt
	$(PYTHON) -m venv .venv
	.venv/bin/pip install -q --upgrade pip
	.venv/bin/pip install -q -r requirements.txt
	@test -f .env || cp .env.example .env
	@touch $@

data: setup
	$(PY) data/generate.py

# One command: install, make the synthetic month, replay it, write the weekly digest.
demo: data
	$(PY) -m watchdog replay
	$(PY) -m watchdog digest

replay: data
	$(PY) -m watchdog replay

# One hourly read at the moment the runaway key is first flagged (dry run).
check: data
	$(PY) -m watchdog check --reset --at 2026-11-18T22:00Z

digest: data
	$(PY) -m watchdog digest

eval: setup
	$(PY) -m evals.run_evals

eval-cloud: setup
	$(PY) -m evals.run_evals --llm --cloud

test: setup
	$(PY) -m unittest discover -s tests -t .

clean:
	rm -rf outbox .state data/sample
