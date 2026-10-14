# Override PYTHON for Windows: make data PYTHON=.venv/Scripts/python.exe
PYTHON ?= python

.PHONY: data test
data:
	$(PYTHON) run_pipeline.py --subset FD001 --stage data

test:
	$(PYTHON) -m pytest -q
