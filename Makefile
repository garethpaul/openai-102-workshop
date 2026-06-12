.PHONY: all audit build check lint lock lock-check lock-upgrade run runtime-check smoke static-check test verify

PYTHON ?= python3
UV ?= uv

# Build the app (compile maintained Python modules)
build:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -c "import pathlib; [compile(pathlib.Path(path).read_text(), path, 'exec') for path in ('components/common.py', 'scripts/check-runtime-imports.py', 'scripts/smoke-streamlit.py', 'test_app.py', 'utils/crawler.py', 'utils/generate.py', 'utils/token.py')]"

# Run the app locally
run:
	streamlit run 👋_Hello.py

# Test the app
test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest -q test_app.py

static-check:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/check-workshop-baseline.py

lock:
	$(UV) pip compile requirements.in --python-version 3.12 --universal --quiet --output-file requirements.txt
	$(UV) pip compile requirements-test.in --python-version 3.12 --universal --quiet --output-file requirements-test.txt

lock-upgrade:
	$(UV) pip compile requirements.in --python-version 3.12 --universal --upgrade --quiet --output-file requirements.txt
	$(UV) pip compile requirements-test.in --python-version 3.12 --universal --upgrade --quiet --output-file requirements-test.txt

lock-check: lock
	git diff --exit-code -- requirements.txt requirements-test.txt

audit:
	pip-audit -r requirements-test.txt
	pip-audit -r requirements.txt

runtime-check:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/check-runtime-imports.py

smoke:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/smoke-streamlit.py

lint: static-check

verify: lint test

check: verify

# Build test and run the app
all: build test run
