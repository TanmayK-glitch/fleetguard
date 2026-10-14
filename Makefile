# Override PYTHON for Windows: make data PYTHON=.venv/Scripts/python.exe
PYTHON ?= python

.PHONY: data train test
data:
	$(PYTHON) run_pipeline.py --subset FD001 --stage data

train:
	$(PYTHON) run_pipeline.py --subset FD001 --stage train

test:
	$(PYTHON) -m pytest -q
