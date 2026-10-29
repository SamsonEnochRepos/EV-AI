PY=.venv/Scripts/python.exe
.PHONY: check lint type test unit integration eval security
check: lint type test
lint:
	$(PY) -m ruff check ev tests
type:
	$(PY) -m mypy ev
test:
	$(PY) -m pytest -q
unit:
	$(PY) -m pytest -q -m "not integration"
integration:
	$(PY) -m pytest -q -m integration
eval:
	$(PY) -m ev.evals.run all
security:
	$(PY) -m ev.evals.run security
