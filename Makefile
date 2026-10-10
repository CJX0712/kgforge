.PHONY: install lint format test cov demo benchmark clean gen

PY ?= python

install:
	$(PY) -m pip install -r requirements.lock.txt

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

test:
	$(PY) -m pytest -q -W ignore::UserWarning

cov:
	$(PY) -m pytest -q -W ignore::UserWarning --cov=kgforge --cov-report=term

demo:
	$(PY) examples/run_demo.py

benchmark:
	$(PY) -m kgforge.cli benchmark --seeds 7,42,123

gen:
	$(PY) -m kgforge.cli gen --out kg_synthetic.tsv

clean:
	rm -rf .pytest_cache __pycache__ .coverage htmlcov
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
