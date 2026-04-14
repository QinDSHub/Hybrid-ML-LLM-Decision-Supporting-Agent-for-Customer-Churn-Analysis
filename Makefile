PYTHON ?= python
CONFIG ?= configs/default.toml

install:
	$(PYTHON) -m pip install -r requirements.txt

preprocess:
	$(PYTHON) scripts/preprocess.py --config $(CONFIG)

predict:
	$(PYTHON) scripts/predict.py --config $(CONFIG)

evaluate:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG)

run:
	$(PYTHON) scripts/run_pipeline.py --config $(CONFIG)

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

lint:
	ruff check src tests
