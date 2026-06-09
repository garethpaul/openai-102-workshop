.PHONY: all build check lint run static-check test verify

PYTHON ?= python3

# Build the app (compile maintained Python modules)
build:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -c "import pathlib; [compile(pathlib.Path(path).read_text(), path, 'exec') for path in ('components/common.py', 'test_app.py', 'utils/crawler.py', 'utils/generate.py')]"

# Run the app locally
run:
	streamlit run 👋_Hello.py

# Test the app
test:
	$(PYTHON) -m pytest -q test_app.py

static-check:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/check-workshop-baseline.py

lint: static-check

verify: lint test

check: verify

# Build test and run the app
all: build test run
